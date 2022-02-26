import torch
import torch.nn as nn
from axial_attention import AxialAttention
from huggingface_hub import PyTorchModelHubMixin
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from data_prep import metnet_dataloader, prepare_data_MetNet
from metnet.layers import ConditionTime, ConvGRU, DownSampler, MetNetPreprocessor, TimeDistributed, ConvLSTM
from torch.utils.data import DataLoader, random_split
import numpy as np
import math

class MetNetPylight(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str = "downsampler", #4 CNN layers
        file_name: str = "combination_all_pn157.h5",
        input_channels: int = 12,
        n_samples: int = 1000,
        sat_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 12,
        hidden_dim: int = 384,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 4,
        forecast_steps: int = 48,
        temporal_dropout: float = 0.2,
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
        '''
        self.preprocessor = MetNetPreprocessor(
            sat_channels=sat_channels, crop_size=input_size, use_space2depth=True, split_input=True
        )
        # Update number of input_channels with output from MetNetPreprocessor
        new_channels = sat_channels * 4  # Space2Depth
        new_channels *= 2  # Concatenate two of them together
        input_channels = input_channels - sat_channels + new_channels'''
        #self.drop = nn.Dropout(temporal_dropout)
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

    def encode_timestep(self, x, fstep=1):
        #print("\n shape before preprocess: ", x.shape)
        # Preprocess Tensor
        #x = self.preprocessor(x)
        #print("\n shape after preprocess: ", x.shape)
        # Condition Time

        x = self.ct(x, fstep)
        x = x.double()

        #print("\n shape after ct: ", x.shape)

        ##CNN
        x = self.image_encoder(x)
        #print("\n shape after image_encoder: ", x.shape)

        # Temporal Encoder
        #_, state = self.temporal_enc(self.drop(x))
        _, state = self.temporal_enc(x)
        #print("\n shape after temp enc: ", state.shape)

        agg = self.temporal_agg(state)
        #print("\n shape after temporal_agg: ", agg.shape)
        return agg

    def forward(self, imgs,lead_time):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        - lead_time #random int between 0,self.forecast_steps
        """

        # Compute all timesteps, probably can be parallelized
        res = []

        x_i = self.encode_timestep(imgs, lead_time)

        out = self.head(x_i)
        res.append(out)
        res = torch.stack(res, dim=1)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch


        lead_time = np.random.randint(0,self.forecast_steps)

        y_hat = self(x.float(),lead_time)
        loss = F.mse_loss(y_hat, y[:,lead_time])
        pbar = {"training_loss": loss}
        return {"loss": loss, "progress_bar": pbar}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        lead_times = list(range(self.forecast_steps))
        loss = 0
        for lead_time in lead_times:
            y_hat = self(x.float(),lead_time)
            loss += F.mse_loss(y_hat, y[:,lead_time])

        return {"loss": loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in val_step_outputs]).mean()
        pbar = {"avg_val_loss":avg_val_loss}

        return {"val_loss":avg_val_loss, "progress_bar":pbar}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=1e-2)
        return optimizer

    def setup(self, stage = None):
        data = metnet_dataloader.MetNetDataset(self.file_name, N = self.n_samples, lead_times = self.forecast_steps)
        nsamples = len(data)
        split_list = [math.floor(nsamples*0.7),math.floor(nsamples*0.3)]
        split_list[0] += nsamples-sum(split_list)
        self.train_data, self.val_data = random_split(data, split_list)
    def train_dataloader(self):


        train_loader = DataLoader(dataset=self.train_data,batch_size=4)
        return train_loader
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data)
        return val_loader


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        #self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)
        self.rnn = ConvLSTM(in_channels, out_channels, ks, n_layers)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])

'''
def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)'''
