import torch
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
from huggingface_hub import PyTorchModelHubMixin
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import optim
from data_prep import metnet_dataloader, prepare_data_MetNet
from metnet.layers import ConditionTime, ConvGRU, DownSampler, MetNetPreprocessor, TimeDistributed, ConvLSTM
from torch.utils.data import DataLoader, random_split
import numpy as np
import math
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import f1_score
import PIL
import matplotlib as mpl
from matplotlib import colors
from metnet.layers.utils import get_conv_layer

class MetNetPylight(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str = "downsampler", #4 CNN layers
        file_name: str = "data3_500k.h5",
        input_channels: int = 12, # radar channels + longitude channels + time encoding channels = 15 (excluding lead time encoding)
        n_samples: int = 1000, # number of radar snapshots to preprocess
        sat_channels: int = 0, # ignore
        input_size: int = 256, # height = width = input_size = 112
        output_channels: int = 512, # number of rain bins
        rain_step: int = 0.2, # size of each rain bin in millimeters
        hidden_dim: int = 384, # hidden dimensions in RNN layer
        kernel_size: int = 3, # Kernel sizes in Downsampler
        num_layers: int = 1, # ignore
        num_att_layers: int = 4, #Number of attention layers, 8 original paper.
        forecast_steps: int = 240, # Number of lead times
        temporal_dropout: float = 0.2, # Dropout
        num_workers: int = 32,
        batch_size: int = 8,
        momentum: float = 0.9,
        att_heads: int = 8,
        plot_every: int = 10,
        keep_biggest: float = 0.8,
        
        learning_rate: int = 1e-2, 
        leadtime_spacing: int = 1,
        **kwargs,
    ):
        super(MetNetPylight, self).__init__()
        pl.seed_everything(42, workers = True)
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)
        sat_channels = self.config["sat_channels"]
        input_size = self.config["input_size"]
        input_channels = self.config["input_channels"]
        temporal_dropout = self.config["temporal_dropout"]
        image_encoder = self.config["image_encoder"]
        forecast_steps = self.config["forecast_steps"]
        hidden_dim = self.config["hidden_dim"]
        kernel_size = self.config["kernel_size"]
        num_layers = self.config["num_layers"]
        num_att_layers = self.config["num_att_layers"]
        output_channels = self.config["output_channels"]

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_samples = n_samples
        self.file_name = file_name
        self.workers = num_workers
        self.rain_step = rain_step
        self.learning_rate = learning_rate        
        self.batch_size = batch_size
        self.plot_every = plot_every
        self.momentum = momentum
        self.att_heads = att_heads
        lead_time_keys = list(range(forecast_steps))
        lead_time_counts = [0 for i in range(forecast_steps)]
        self.lead_time_histogram = dict(zip(lead_time_keys,lead_time_counts))
        self.keep_biggest = keep_biggest
        self.weights = None
        self.leadtime_spacing = leadtime_spacing
        
        if str(self.device)== "cuda:0":
            self.printer = True
        else:
            self.printer = False
        '''
        self.preprocessor = MetNetPreprocessor(
            sat_channels=sat_channels, crop_size=input_size, use_space2depth=True, split_input=True
        )
        # Update number of input_channels with output from MetNetPreprocessor
        new_channels = sat_channels * 4  # Space2Depth
        new_channels *= 2  # Concatenate two of them together
        input_channels = input_channels - sat_channels + new_channels'''
        self.drop = nn.Dropout(temporal_dropout)
        '''if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")'''
        image_encoder = DownSampler(input_channels + forecast_steps)
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(forecast_steps)
        self.temporal_enc = TemporalEncoder(
            image_encoder.output_channels, hidden_dim, ks=kernel_size, n_layers=num_layers
        )
        self.position_embedding = AxialPositionalEmbedding(dim=self.temporal_enc.out_channels, shape = (input_size // 4, input_size // 4))
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=self.att_heads, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )
        '''conv2d = get_conv_layer(conv_type="standard")
        self.conv_agg = nn.Sequential(
            conv2d(hidden_dim, hidden_dim, kernel_size=(28,1), padding="same"),
            conv2d(hidden_dim, hidden_dim, kernel_size=(1,28), padding="same"),
            conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=1),
            #nn.MaxPool2d((2, 2), stride=2),
            # antialiased_cnns.BlurPool(160, stride=2) if antialiased else nn.Identity(),
            nn.BatchNorm2d(hidden_dim),
            conv2d(hidden_dim, hidden_dim, kernel_size=(28,1)),
            conv2d(hidden_dim, hidden_dim, kernel_size=(1,28)),
            conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=1),           
            #nn.MaxPool2d((2, 2), stride=2),
            # antialiased_cnns.BlurPool(output_channels, stride=2) if antialiased else nn.Identity(),
        )'''
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))  # Reduces to mask
        self.double()
        
        self.save_hyperparameters()

    def encode_timestep(self, x, fstep=1, lead_times = []):
        #print("\n shape before preprocess: ", x.shape)
        # Preprocess Tensor
        #x = self.preprocessor(x)
        #print("\n shape after preprocess: ", x.shape)
        # Condition Time
        #plot_channels(x, 1, tit_add = "INPUT")
        if lead_times:
            bs, t, c, w, h = x.shape
            x_temp = torch.empty((bs, t, c+self.forecast_steps, w, h), device = self.device)
            for i,lead_time in enumerate(lead_times):
                
                x_temp[i] = self.ct(x[i:i+1], lead_time)
        else:
            x_temp = self.ct(x, fstep)

        x = x_temp.double()
        
        

        #print("\n shape after ct: ", x.shape)

        ##CNN
        x = self.image_encoder(x)
        #plot_channels(x, 1, tit_add = "after image_encoder")
        #print("\n shape after image_encoder: ", x.shape)

        # Temporal Encoder
        #_, state = self.temporal_enc(self.drop(x))
        dropped = self.drop(x)
        _, state = self.temporal_enc(dropped)
        embedded = self.position_embedding(state)
        #plot_channels(state, 1, tit_add = "after temporal_enc")
        #print("\n shape after temp enc: ", state.shape)

        agg = self.temporal_agg(state)
        #agg = self.conv_agg(state)
        #plot_channels(x, 1, tit_add = "after agg")
        #print("\n shape after temporal_agg: ", agg.shape)
        return agg
    
    def forward(self, imgs,lead_time = 0, lead_times = []):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        - lead_time #random int between 0,self.forecast_steps
        """

        # Compute all timesteps, probably can be parallelized
        #print("in forward")
        #print_channels(imgs[0,0])
        #plot_channels(imgs, 2, tit_add = "input")
        #print(imgs.shape)
        x = self.encode_timestep(imgs, lead_time, lead_times)
        #plot_channels(x, 2, tit_add = "after encode")
        #print("shape before head: ", x.shape)
        out = self.head(x)
        #print("shape after head: ", out.shape)
        #plot_channels(out, 1, tit_add = "after head")
        #soft = torch.softmax(out,dim=1)
        #print("shape after softmax: ", soft.shape)
        #plot_channels(out, 1, tit_add = "after softmax")
        #plot_bins(out[0], " after head",soft[0], "after softmax")
        return out
        

    def training_step(self, batch, batch_idx):
        
        x, y, rainy_leads = batch
        #print("training_step len: ", x.shape[0])
        
        bs = x.shape[0]
        lead_times = [int(np.random.choice(leads.cpu().detach().numpy())) for leads in rainy_leads]
        
        #w = torch.tensor(self.weights,device = self.device)
        L = CrossEntropyLoss()
        y_hat = self(x.float(),lead_times=lead_times)
        
        #y_leads = torch.empty(y[:,lead_time].shape, device = self.device)
        #y_leads = torch.tensor([y[i,lead_times[i]] for i in range(self.batch_size)], device = self.device)
        loss = L(y_hat, y[torch.arange(bs), lead_times], )
        
        
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        '''if batch_idx == 0:
            
            self.train_batch = (x[0:1],y[0:1])'''
            
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
    
        x, y = batch
        #print("validation_step len: ", x.shape[0])
        #log_img = (batch_idx == 0 and str(self.device)== "cuda:0")
        lead_times = list(range(self.forecast_steps))
        
        loss = 0
        
        L = CrossEntropyLoss()
        for lead_time in lead_times:
            y_hat = self(x.float(),lead_time)
            
            
            loss += L(y_hat, y[:,lead_time])

            '''y_img, y_hat_img = thresh_imgs(y[0,lead_time], y_hat[0], thresh_bin = 1)
            if log_img:
                self.logger.experiment.log({f"val_{lead_time}":[wandb.Image(y_img.cpu(), caption=f"y leadtime {lead_time}"), wandb.Image(y_hat_img.cpu(), caption=f"y_hat leadtime {lead_time}")]})'''
        loss /= self.forecast_steps
        
        self.log("validation/loss_epoch", loss, on_step=False, on_epoch=True)
        
        return {"loss": loss}
        
    def test_step(self, batch, batch_idx):
        x,y = batch
        
        
        
        # ---------- calculate test_loss ---------- 
        loss = 0
        L = CrossEntropyLoss()
        y_hat = torch.empty(y.shape)
        for lead_time in range(self.forecast_steps):
            y_hat_here= self(x,lead_time)
            loss += L(y_hat_here, y[:,lead_time])
            y_hat[:,lead_time] = y_hat_here
        loss /= self.forecast_steps
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        # ---------- calculate test_loss ---------- 
        
     
        
        thresh = 1
        no_rain = torch.zeros(y[0,0].shape, device = self.device)
        no_rain[0] = 1

        
            
            
        '''    
        #for i in range(y.shape[0]):
        for i in range(1):
            temp3 = x[i,-1,0:4].cpu().numpy()
            temp3 = np.mean(temp3, axis = 0)
            temp3 = (temp3 + np.min(temp3))/(np.max(temp3)-np.min(temp3))
            temp3 = temp3*255
            pil_im_x = PIL.Image.fromarray(np.uint8(temp3))
            self.logger.log_image(key=f"val_{lead_time}", images=[pil_im_x], caption = ["x"])
            
            self.f1_count += 1
            imgs = []
            capts = []
            for lead_time in range(0,self.forecast_steps,10):
                
                y_img, y_hat_img, f1,f1_control = thresh_imgs(y[i,lead_time], y_hat[i,lead_time], no_rain, thresh_bin = thresh)
                self.avg_y_img[lead_time]+=(torch.mean(y_img.cpu()))
                self.avg_y_hat_img[lead_time]+=(torch.mean(y_hat_img.cpu()))
                self.f1s[lead_time] += f1
                self.f1s_control[lead_time] += f1_control
                
                if lead_time in list(range(0,60,10)):
                    
                    temp1 = y_img.cpu().numpy()
                    pil_im_y = PIL.Image.fromarray(np.uint8(temp1)*255)
                    imgs.append(pil_im_y)
                    mean_y = str(np.mean(temp1))

                    capts.append(f"y {lead_time}, mean={mean_y[0:4]}")
                    temp2 = y_hat_img.cpu().numpy()
                    pil_im_y = PIL.Image.fromarray(np.uint8(temp2)*255)
                    imgs.append(pil_im_y)
                    capts.append(f"y_hat {lead_time}, f1={round(f1,4)}")
                    
                    
                    
                    fig,axs = plt.subplots(1,2)
                    im0 = axs[0].imshow(y_img, vmin=0, vmax=1)
                    axs[0].set_title("y")
                    im0 = axs[1].imshow(y_hat_img, vmin=0, vmax=1)
                    axs[1].set_title("y_hat")
                    fig.suptitle(f"Threshhold: {thresh}, lead time: {lead_time}, with f1-score: {f1}")
                    fig.subplots_adjust(right=0.8)
                    
                    fig.colorbar(im0, ax=axs.ravel().tolist())
                    plt.show()
            self.logger.log_image(key=f"val_{lead_time}", images=imgs, caption = capts)
        '''
        for i in range(x.shape[0]):
            softed = torch.softmax(y_hat[i],dim=1)
            plot_probabillity(y[i],softed,[kk for kk in range(0,self.forecast_steps,self.forecast_steps//3)],increment = 0.2, spacing = self.leadtime_spacing)
            #plot_categories(y[i,0],softed,increment = 0.2)
        
        #plot bins:
        '''for i in range(x.shape[0]):
            softed = torch.softmax(y_hat[i,0],dim=0)
            plot_bins(x[i], softed[0:9], "x" ,y[i,0,0:9], "y")
        #plot category:
        for i in range(x.shape[0]):
            plot_category(y[i,0],y_hat[i,0],self.output_channels,self.rain_step,self.device)'''
    def on_train_epoch_end(self):
        imgs = []
        capts = []
        thresh = 1
        
        '''for lead_time in range(0,self.forecast_steps,10):
            y_hat = self(self.train_batch[0],lead_time)
            y = self.train_batch[1]
            no_rain = torch.zeros(y[0,0].shape, device = self.device)
            no_rain[0] = 1
            y_img, y_hat_img, f1,f1_control = thresh_imgs(y[0,lead_time], y_hat[0], no_rain, thresh_bin = thresh)
            
            
            
            temp1 = y_img.cpu().numpy()
            pil_im_y = PIL.Image.fromarray(np.uint8(temp1)*255)
            imgs.append(pil_im_y)
            capts.append(f"y {lead_time}")
            temp2 = y_hat_img.cpu().numpy()
            pil_im_y = PIL.Image.fromarray(np.uint8(temp2)*255)
            imgs.append(pil_im_y)
            capts.append(f"y_hat {lead_time}")
        self.logger.log_image(key=f"train_end_{lead_time}", images=imgs, caption = capts)'''
        
    def validation_epoch_end(self, val_step_outputs):
        '''wombo = self.logger.experiment
        hist_scores = [[s] for s in self.lead_time_histogram.values()]
        table = wandb.Table(data = hist_scores, columns = ["lead_times"])
        wombo.log({"leadtimes_histogram": wombo.plot.histogram(table, "Lead times", title="Lead times histogram")})'''
        outs = tuple([x["loss"] for x in val_step_outputs])

        avg_val_loss = torch.tensor(outs, device = self.device).mean()

        return {"val_loss":avg_val_loss}
    def test_epoch_end(self,test_step_outputs):
        f1mean = np.array(self.f1s)/self.f1_count
        f1_control_mean = np.array(self.f1s_control)/self.f1_count
        plt.plot(f1mean,"b", label="f1 meaned")
        plt.plot(f1_control_mean,"--g", label="no rain controll")
        plt.legend()
        plt.title(f"f1-scores over leadtimes")
        plt.xlabel("lead_time")
        plt.ylabel("f1")
        plt.show()
        
        y_means = np.array(self.avg_y_img)/self.f1_count
        y_hat_means = np.array(self.avg_y_hat_img)/self.f1_count
        plt.plot(y_means,"b", label="y means")
        plt.plot(y_hat_means,"g", label="y_hat means")
        plt.legend()
        plt.title(f"y and yhat above threshhold at different leadtimes")
        plt.xlabel("lead_time")
        plt.ylabel("mean")
        plt.show()
        
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=self.learning_rate, momentum = self.momentum)
        return optimizer

    def setup(self, stage = None):
        self.train_data = metnet_dataloader.MetNetDataset("train", N = self.n_samples , keep_biggest = self.keep_biggest, leadtime_spacing = self.leadtime_spacing, lead_times = self.forecast_steps)
        self.val_data = metnet_dataloader.MetNetDataset("val", N = None, keep_biggest = 1, leadtime_spacing = self.leadtime_spacing)
        self.test_data = metnet_dataloader.MetNetDataset("test", N = self.n_samples, keep_biggest = 1, leadtime_spacing = self.leadtime_spacing)
        '''if self.train_data.weights is not None:

            self.weights = self.train_data.weights'''
       
        
       
        print(f"Training data samples = {len(self.train_data)}")
        
        print(f"Validation data samples = {len(self.val_data)}")
        print(f"Test data samples = {len(self.test_data)}")
        #print("SHAPES: ", self.train_data.shape, self.val_data.shape, self.test_data.shape)
        #print([i.split("/")[-1][0:6] for i in self.train_data.file_names])
        print("FIXING IMBALANCE")
        self.class_balancing = torch.from_numpy(np.load("/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/class_imbalance.npy")).to(self.device)
        self.class_balancing_half = torch.from_numpy(np.load("/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/class_imbalance0.15.npy")).to(self.device)
        
        a = torch.divide(self.class_balancing, self.class_balancing_half).cpu().numpy()
        a = np.nan_to_num(a)
        b = np.arange(0,len(a))
        idx = np.where(a!=0)
        #print(len(a))
        #print(a)
        #plt.plot(self.class_balancing.cpu(), "--r")
        '''plt.scatter(b[idx],a[idx])
        plt.title("")
        #plt.yscale("log")
        plt.show()'''
        self.rain_bins = torch.zeros((self.output_channels,))
        '''for i, batch in enumerate(self.train_data):
            x,y = batch
            
            self.rain_bins += torch.sum(y.cpu(),dim = [0,1,3,4])
            if i%50==0: 
                print(f"{i}/{len(self.train_data)}")
        np.save(f"/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/class_imbalance{self.keep_biggest}.npy",self.rain_bins.numpy())
        plt.plot(self.rain_bins)
        plt.yscale("log")
        plt.show()'''
        
    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_data,batch_size=self.batch_size, num_workers =  self.workers, shuffle = True)
        return train_loader
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.batch_size, num_workers =  self.workers)
        return val_loader
    def test_dataloader(self):
        test_loader = DataLoader(self.test_data,batch_size=self.batch_size, num_workers =  self.workers)
        return test_loader
        
    '''def on_train_epoch_start(self):
        print("INSIDE NOW")
        self.rain_bins = torch.zeros((self.output_channels,))
        for i, batch in enumerate(self.train_data):
            x,y = batch
            self.rain_bins += torch.sum(y.cpu(),dim = [0,1,3,4])
            if i%50==0: 
                print(f"{i}/{len(self.train_data)}")
        plt.plot(self.rain_bins)
        plt.yscale("log")
        plt.show()'''
        
                
def thresh_imgs(y, y_hat, after_five, thresh_bin = 1):
    bins, w, h = y_hat.shape
    
    y_below = torch.sum(y[0:thresh_bin], dim=0)
    y_above = torch.sum(y[thresh_bin:], dim=0)
    y_outcome = torch.ones((w, h))
    y_idx_less_rain = torch.where(y_below<y_above)
    y_outcome[y_idx_less_rain] = 0
    
    after_five_below = torch.sum(after_five[0:thresh_bin], dim=0)
    after_five_above = torch.sum(after_five[thresh_bin:], dim=0)
    after_five_outcome = torch.ones((w, h))
    after_five_idx_less_rain = torch.where(after_five_below<after_five_above)
    after_five_outcome[after_five_idx_less_rain] = 0
    
    
    y_hat_soft = torch.softmax(y_hat,dim=0)
    y_hat_below = torch.sum(y_hat_soft[0:thresh_bin], dim=0)
    y_hat_above = torch.sum(y_hat_soft[thresh_bin:], dim=0)
    y_hat_outcome = torch.ones((w, h))
    y_hat_idx_less_rain = torch.where(y_hat_below<y_hat_above)
    y_hat_outcome[y_hat_idx_less_rain] = 0
    
    a = torch.flatten(y_outcome)
    b = torch.flatten(y_hat_outcome)
    control = torch.flatten(after_five_outcome)
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    control = control.cpu().detach().numpy()
    
    f1 = f1_score(a, b)
    f1_control = f1_score(a,control)

    
    return y_outcome, y_hat_outcome, f1, f1_control
       
def thresh_F1(y_hat, y,thresh_bin = 1):
    #input: pred shape (None, lead_times, bins, 28, 28), 
    #y shape (None, bins, 28, 28), 
    #thresh_bin: groups bin below (excluding thresh_bin) into one
    n, bins, w, h = y_hat.shape
    
    y_below = torch.sum(y[:,0:thresh_bin], dim=1)
    y_above = torch.sum(y[:,thresh_bin:], dim=1)
    y_outcome = torch.ones((n, w, h))
    y_idx_less_rain = torch.where(y_below<y_above)
    y_outcome[y_idx_less_rain] = 0
    
    
    y_hat_soft = torch.softmax(y_hat,dim=1)
    y_hat_below = torch.sum(y_hat_soft[:,0:thresh_bin], dim=1)
    y_hat_above = torch.sum(y_hat_soft[:,thresh_bin:], dim=1)
    y_hat_outcome = torch.ones((n, w, h))
    y_hat_idx_less_rain = torch.where(y_hat_below<y_hat_above)
    y_hat_outcome[y_hat_idx_less_rain] = 0
    
    f1_here = 0
    for y_binned, y_hat_binned in zip(y_outcome,y_hat_outcome):
        a = torch.flatten(y_binned)
        b = torch.flatten(y_hat_binned)
        a = a.cpu().detach().numpy()
        b = b.cpu().detach().numpy()
        f1_here += f1_score(a, b)
    return f1_here/y_outcome.shape[0]
        
    
class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.out_channels = out_channels
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)
        #self.rnn = ConvLSTM(in_channels, out_channels, ks, n_layers)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])
        
def rain_transform(x):
    rain = (10**(x / 10.0) / 200.0)**(1.0 / 1.6)
    return rain
   
def plot_category(y,y_hat,bins,increment,title,thresh_bin = 2):
    #accepts y.shape = (bins,w,h)
    print(title, y.shape)
    print(title, y_hat.shape)

    _, w, h = y.shape
    a1 = torch.empty((w,h))
    a2 = torch.empty((w,h))
    part_1_a = torch.sum(y[0:thresh_bin], dim = 0)
    part_1_b = torch.sum(y[thresh_bin:], dim = 0)
    part_2_a = torch.sum(y_hat[0:thresh_bin], dim = 0)
    part_2_b = torch.sum(y_hat[thresh_bin:], dim = 0)
    idx_1_below = torch.where(part_1_a==1)
    idx_1_above = torch.where(part_1_b==1)
    idx_2_below = torch.where(part_2_a<=part_2_b)
    idx_2_above = torch.where(part_2_a>part_2_b)
    a1[idx_1_below] = 0
    a1[idx_1_above] = 1
    a2[idx_2_below] = 0
    a2[idx_2_above] = 1
    y = a1.cpu().detach().numpy()
    y_hat = a2.cpu().detach().numpy()
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle(str(title))
    ax[0].imshow(y)
    ax[1].imshow(y_hat)
    plt.show()

def plot_categories(y,y_hat,increment = 0.2):
    #accepts y.shape = (bins,w,h)
    
    _, w, h = y.shape
    categories = [(0,1), (1,5), (5,10), (10, y.shape[0]-1)]
    labels = {}
    bounds = [i for i in range(len(categories)+1)]
    rain_img_y = torch.zeros((w,h))
    rain_img_y_hat = torch.zeros((w,h))
    part_y_hat_probs = rain_img_y_hat[:]
    for i, (low,high) in enumerate(categories):
        low_str = str(increment*low)
        high_str = str(increment*high)
        labels[i] = low_str + "-" + high_str
        
        
        
        part_y = torch.sum(y[low:high], dim = 0)
       
        part_y_hat = torch.sum(y_hat[low:high], dim = 0)
        
        idx_y = torch.where(part_y==1)
        idx_y_hat = torch.where(part_y_hat>part_y_hat_probs)
        part_y_hat_probs[idx_y_hat] = part_y_hat[idx_y_hat]
        rain_img_y[idx_y] = i
        rain_img_y_hat[idx_y_hat] = i
        
    fig, ax = plt.subplots(1,2)
    rain_img_y = rain_img_y/torch.max(rain_img_y)
    rain_img_y_hat = rain_img_y_hat/torch.max(rain_img_y_hat)
    ax[0].imshow(rain_img_y, cmap = "hot")
    ax[0].set_title("Ground truth rain")
    ax[1].imshow(rain_img_y_hat, cmap = "hot")
    ax[1].set_title("Predicted rain")
    plt.show()
    
def plot_probabillity(y,y_hat,lead_times ,increment = 0.2,spacing = 1):
    #accepts y.shape = (leads, bins,w,h)
    
    _, _, w, h = y.shape
    fig, ax = plt.subplots(len(lead_times),2)
    
    
    bounds = [0, 0.2, 1,3]
    prob_bounds = [0, 0.2, 0.4, 0.6, 0.8, 1]
    
    
    ii = np.arange(0,28)
    jj = np.arange(0,28)
    xx, yy = np.meshgrid(ii, jj)
    
    divnorm_bounds = colors.TwoSlopeNorm(vmin=0, vcenter=0.1, vmax=3)
    
    for j, lead_time in enumerate(lead_times):
        rain_img_y = torch.zeros((w,h))
        
        for i in range(y.shape[1]):
            idx_y_rain = torch.where( y[lead_time,i] == 1)
            rain_img_y[idx_y_rain] = i*increment
        
        #zz = rain_img_y[xx,yy]
        
        rain_img_y_hat = torch.sum(y_hat[lead_time,1:], dim = 0)
        
        #np.save("att_visualisera.npy", rain_img_y.cpu().detach().numpy())
        zz = rain_img_y.cpu().detach().numpy()
        
        zz_hat = rain_img_y_hat.cpu().detach().numpy()
        im1 = ax[j,0].contourf(xx,yy,zz,bounds,cmap = "Reds", extend="both", norm = divnorm_bounds)
        
        ax[j,0].set_title(f"Lead time:{(lead_time*5+5)*spacing} min")
        im2 = ax[j,1].contourf(xx,yy,zz_hat,prob_bounds,cmap = "Greens")
        ax[j,1].set_title(f"Lead time:{(lead_time*5+5)*spacing} min")
        ax[j,0].get_xaxis().set_visible(False)
        ax[j,0].get_yaxis().set_visible(False)
        ax[j,1].get_xaxis().set_visible(False)
        ax[j,1].get_yaxis().set_visible(False)
        
    #fig.subplots_adjust(left=0.2)
    
    
    cb1 = fig.colorbar(im1, ax=ax[:,0])
    cb1.set_label("Rain [mm/h]")
    cb2 = fig.colorbar(im2, ax=ax[:,1])
    cb2.set_label("Probabillity of rain>0.2mm/h")
    fig.suptitle("Ground truth rainfall (left) vs. Prediction probabillity (right)")
    plt.show()
def plot_bins(x, y1, title1="", y2 = 0, title2=""):
    
    N = y1.shape[0]
    side = int(N**0.5)
    if side**2<N: side += 1
    fig1, ax = plt.subplots(1,1)
    x = x.cpu().detach().numpy()
    x = np.mean(x[6,4:8,:,:], axis=0)
    ax.imshow(x)
    fig1.suptitle("input 5 minutes before")
    
    
    for y, title in zip([y1, y2],[title1,title2]):
        fig, axs = plt.subplots(side,side)
        fig.suptitle("All bins " + title)
        axs = axs.reshape(-1)
        y = y.cpu().detach().numpy()
        vmax = np.max(y)
        vmin = np.min(y)
        for i,array in enumerate(y):
            
            im = axs[i].imshow(array,vmax = vmax, vmin = vmin)
            axs[i].set_title(str(i) + " mean: " +f'{np.mean(array):.2f}')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    plt.show()
       
def plot_channels(x, maxN=None,tit_add=""):
    print("CHANNEL SHAPE: ", x.shape)
    
    if len(x.shape)==5:
        if not maxN or maxN>x.shape[2]: maxN = x.shape[2]
        channels = np.random.choice(x.shape[2],maxN, replace = False)
        for channel in channels:
            
            plt.imshow(x[0,0,0,:,:].cpu().detach().numpy())
            plt.colorbar()
            plt.title(str(channel)+" " + tit_add)
            plt.show()
    else:
        if not maxN or maxN>x.shape[1]: maxN = x.shape[1]
        channels = np.random.choice(x.shape[1],maxN, replace = False)
        for channel in channels:
            
            plt.imshow(x[0,0,:,:].cpu().detach().numpy())
            plt.colorbar()
            plt.title(str(channel)+" " + tit_add)
            plt.show()


          
def print_channels(x):
    x = x.cpu().detach().numpy()
    for channel, array in enumerate(x):
        mean = np.mean(array)
        std = np.std(array)
        print(f"Mean: {mean} Std: {std} for channel {channel}")
class RainfieldCallback(pl.Callback):
    def __init__(self, val_samples, num_samples=1):
        super().__init__()
        self.val_imgs, self.val_Y = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_Y = self.val_Y[:num_samples]
        
          
    def on_validation_epoch_end(self, trainer, pl_module, lead_times = 60):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        for lead_time in range(lead_times):
            y_hat = pl_module(val_imgs)
            preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })

'''
def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)'''
