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
        #self.drop = nn.Dropout(temporal_dropout)
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
        _, state = self.temporal_enc(x)
        embedded = self.position_embedding(state)
        #plot_channels(state, 1, tit_add = "after temporal_enc")
        #print("\n shape after temp enc: ", state.shape)

        agg = self.temporal_agg(state)
        #plot_channels(x, 1, tit_add = "after agg")
        #print("\n shape after temporal_agg: ", agg.shape)
        return state

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
        
        x, y = batch
        #print("training_step len: ", x.shape[0])
        
        bs = x.shape[0]
        lead_times = [np.random.randint(0,self.forecast_steps) for _ in range(bs)]
        
        L = CrossEntropyLoss()
        y_hat = self(x.float(),lead_times=lead_times)
        #y_leads = torch.empty(y[:,lead_time].shape, device = self.device)
        #y_leads = torch.tensor([y[i,lead_times[i]] for i in range(self.batch_size)], device = self.device)
        loss = L(y_hat, y[torch.arange(bs), lead_times])
        if self.plot_every:
            if (self.global_step)%self.plot_every == 0 and str(self.device)== "cuda:0":
                for lead_time in range(self.forecast_steps):
                    y_hat = self(x.float(),lead_time = lead_time)
                    plot_bins(x[0], y_hat[0], " y_hat leadtime: "+str(lead_time),y[0,lead_time], " y leadtime: "+str(lead_time))
        
        #loss = L(y_hat, y[:,lead_time])
        
        self.log("train/loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
    
        x, y = batch
        #print("validation_step len: ", x.shape[0])
        lead_times = list(range(self.forecast_steps))
        
        loss = 0
        L = CrossEntropyLoss()
        for lead_time in lead_times:
            y_hat = self(x.float(),lead_time)
            
            loss += L(y_hat, y[:,lead_time])
        loss /= self.forecast_steps
        self.log("validation/loss_epoch", loss, on_step=False, on_epoch=True)
        #if self.global_step%100 == 1:
            #plot_category(y[0,0],y_hat[0],self.output_channels,self.rain_step,self.device)
        return {"loss": loss}
        
    def test_step(self, batch, batch_idx):
        x,y = batch
        # ---------- calculate test_loss ---------- 
        loss = 0
        L = CrossEntropyLoss()
        for lead_time in range(self.forecast_steps):
            y_hat= self(x,lead_time)
            loss += L(y_hat, y[:,lead_time])
        loss /= self.forecast_steps
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        # ---------- calculate test_loss ---------- 
        
        # ---------- get all lead times  ---------- 
        y_hat = torch.empty(y.shape)
        for lead_time in range(self.forecast_steps):
            y_hat[:,lead_time] = self(x,lead_time)
        # ---------- get all lead times  ----------    
        
        #plot bins:
        for i in range(x.shape[0]):
            plot_bins(x[i], y_hat[i,0], "x" ,y[i,0], "y")
        #plot category:
        for i in range(x.shape[0]):
            plot_category(y[i,0],y_hat[i,0],self.output_channels,self.rain_step,self.device)

    def validation_epoch_end(self, val_step_outputs):
        '''wombo = self.logger.experiment
        hist_scores = [[s] for s in self.lead_time_histogram.values()]
        table = wandb.Table(data = hist_scores, columns = ["lead_times"])
        wombo.log({"leadtimes_histogram": wombo.plot.histogram(table, "Lead times", title="Lead times histogram")})'''
        outs = tuple([x["loss"] for x in val_step_outputs])
        avg_val_loss = torch.tensor(outs, device = self.device).mean()
        return {"val_loss":avg_val_loss}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=self.learning_rate, momentum = self.momentum)
        return optimizer

    def setup(self, stage = None):
        data = metnet_dataloader.MetNetDataset(self.file_name, N = self.n_samples, lead_times = self.forecast_steps, rain_step = self.rain_step, n_bins = self.output_channels, keep_biggest = self.keep_biggest, printer = self.printer)
        nsamples = len(data)
        split_list = [math.floor(nsamples*0.7),math.floor(nsamples*0.15),math.floor(nsamples*0.15)]
        split_list[1] = max(1,split_list[1])
        split_list[2] = max(1,split_list[2])
        split_list[0] += nsamples-sum(split_list)
        self.train_data, self.val_data, self.test_data= random_split(data, split_list)
        #print("SHAPES: ", self.train_data.shape, self.val_data.shape, self.test_data.shape)
        
        
    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_data,batch_size=self.batch_size, num_workers =  self.workers)
        return train_loader
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.batch_size, num_workers =  self.workers)
        return val_loader
    def test_dataloader(self):
        test_loader = DataLoader(self.test_data,batch_size=self.batch_size, num_workers =  self.workers)
        return test_loader
        
    def thresh_F1(self, y_hat, y,thresh = 0.2):
        #input: pred shape (None, lead_times, bins, 28, 28), y shape (None, lead_times, bins, 28, 28)
        n, leads, bins, w, h = y_hat.shape
        thresh_bin = thresh//self.rain_step 
        y_thresh = torch.zeros(n, leads, 2, w, h) #binary threshhold
        y_thresh[:,:,1,:,:] = 1
        y_hat_thresh = torch.copy(y_thresh)
        idx_y = torch.where(y[:,:,:thresh_bin,:,:] == 1)
        idx_y_hat = torch.where(y_hat[:,:,:thresh_bin,:,:] == 1)
        y_thresh[:,:,0,:,:][idx_y] = 1
        y_thresh[:,:,1,:,:][idx_y] = 0
        
        
    
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
