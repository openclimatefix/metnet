import torch
import torch.nn as nn
from axial_attention import AxialAttention
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

class MetNetPylight(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str = "downsampler", #4 CNN layers
        file_name: str = "data3_500k.h5",
        input_channels: int = 12,
        n_samples: int = 1000,
        sat_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 512,
        hidden_dim: int = 384,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 1,
        forecast_steps: int = 240,
        temporal_dropout: float = 0.2,
        num_workers: int = 32,
        batch_size: int = 8,
        rain_step: int = 0.2, #in millimeters
        learning_rate: int = 1e-2, 
        **kwargs,
    ):
        super(MetNetPylight, self).__init__()
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
        '''
        self.preprocessor = MetNetPreprocessor(
            sat_channels=sat_channels, crop_size=input_size, use_space2depth=True, split_input=True
        )
        # Update number of input_channels with output from MetNetPreprocessor
        new_channels = sat_channels * 4  # Space2Depth
        new_channels *= 2  # Concatenate two of them together
        input_channels = input_channels - sat_channels + new_channels'''
        self.drop = nn.Dropout(temporal_dropout)
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(forecast_steps)
        self.temporal_enc = TemporalEncoder(
            image_encoder.output_channels, hidden_dim, ks=kernel_size, n_layers=num_layers
        )
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=8, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )

        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))  # Reduces to mask
        self.double()
        
        self.save_hyperparameters()

    def encode_timestep(self, x, fstep=1):
        #print("\n shape before preprocess: ", x.shape)
        # Preprocess Tensor
        #x = self.preprocessor(x)
        #print("\n shape after preprocess: ", x.shape)
        # Condition Time
        plot_channels(x, 1, tit_add = "before ct")
        x = self.ct(x, fstep)
        
        x = x.double()
        
        

        #print("\n shape after ct: ", x.shape)

        ##CNN
        x = self.image_encoder(x)
        #plot_channels(x, 1, tit_add = "after downsampler")
        #print("\n shape after image_encoder: ", x.shape)

        # Temporal Encoder
        #_, state = self.temporal_enc(self.drop(x))
        _, state = self.temporal_enc(x)
        #plot_channels(x, 1, tit_add = "after temporal_enc")
        #print("\n shape after temp enc: ", state.shape)

        agg = self.temporal_agg(state)
        plot_channels(x, 1, tit_add = "after agg")
        #print("\n shape after temporal_agg: ", agg.shape)
        return agg

    def forward(self, imgs,lead_time):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        - lead_time #random int between 0,self.forecast_steps
        """

        # Compute all timesteps, probably can be parallelized
        print("in forward")
        #plot_channels(imgs, 2, tit_add = "input")
        x = self.encode_timestep(imgs, lead_time)
        #plot_channels(x, 2, tit_add = "after encode")
        print("shape before head: ", x.shape)
        out = self.head(x)
        print("shape after head: ", out.shape)
        plot_channels(x, 1, tit_add = "after head")
        out = torch.softmax(out,dim=1)
        print("shape after softmax: ", out.shape)
        plot_channels(out, 1, tit_add = "after softmax")
        
        return out

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        print("training_step len: ", x.shape[0])

        lead_time = np.random.randint(0,self.forecast_steps)
        L = CrossEntropyLoss()
        y_hat = self(x.float(),lead_time)
        for i in range(x.shape[0]):
            plot_bins(y_hat[i], "y_hat")
            plot_bins(y[i,0], "y")
        loss = L(y_hat, y[:,lead_time])
        self.log("train/loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
    
        x, y = batch
        print("validation_step len: ", x.shape[0])
        lead_times = list(range(self.forecast_steps))
        
        loss = 0
        L = CrossEntropyLoss()
        for lead_time in lead_times:
            y_hat = self(x.float(),lead_time)
            
            loss += L(y_hat, y[:,lead_time])
        self.log("validation/loss_epoch", loss, on_step=False, on_epoch=True)
        if self.global_step%100 == 1:
            plot_category(y[0,0],y_hat[0],self.output_channels,self.rain_step,self.device)
        return {"loss": loss}

    def validation_epoch_end(self, val_step_outputs):
        outs = tuple([x["loss"] for x in val_step_outputs])
        avg_val_loss = torch.tensor(outs).mean()
        return {"val_loss":avg_val_loss}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=self.learning_rate)
        return optimizer

    def setup(self, stage = None):
        data = metnet_dataloader.MetNetDataset(self.file_name, N = self.n_samples, lead_times = self.forecast_steps, rain_step = self.rain_step, n_bins = self.output_channels)
        nsamples = len(data)
        split_list = [math.floor(nsamples*0.7),math.floor(nsamples*0.3)]
        split_list[0] += nsamples-sum(split_list)
        self.train_data, self.val_data = random_split(data, split_list)
        
    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_data,batch_size=self.batch_size, num_workers =  self.workers)
        return train_loader
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.batch_size, num_workers =  self.workers)
        return val_loader
        
    def thresh_F1(self, pred, target,thresh = 0.2):
        thresh_bin = thresh//self.rain_step 
        norms =  torch.sum(pred, dim=1)

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
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

def plot_bins(y, title=""):
    N = y.shape[0]
    side = int(N**0.5)
    if side**2<N: side += 1
    fig, axs = plt.subplots(side,side)
    fig.suptitle("All bins " + title)
    axs = axs.reshape(-1)
    for i,img in enumerate(y):
        axs[i].imshow(img.cpu().detach().numpy())
        axs[i].set_title(str(i))
    plt.show()
       
def plot_channels(x, maxN=None,tit_add=""):
    print("CHANNEL SHAPE: ", x.shape)
    
    if len(x.shape)==5:
        if not maxN or maxN>x.shape[2]: maxN = x.shape[2]
        channels = np.random.choice(x.shape[2],maxN, replace = False)
        for channel in channels:
            
            plt.imshow(x[0,0,channel,:,:].cpu().detach().numpy())
            plt.title(str(channel)+" " + tit_add)
            plt.show()
    else:
        if not maxN or maxN>x.shape[1]: maxN = x.shape[1]
        channels = np.random.choice(x.shape[1],maxN, replace = False)
        for channel in channels:
            
            plt.imshow(x[0,channel,:,:].cpu().detach().numpy())
            plt.title(str(channel)+" " + tit_add)
            plt.show()
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
