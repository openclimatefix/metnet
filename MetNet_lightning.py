from metnet.models.metnet_pylight import MetNetPylight
import torch
import torch.nn.functional as F
from data_prep.prepare_data_MetNet import load_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import DeviceStatsMonitor

wandb.login()

model = MetNetPylight(
        hidden_dim=128, #384 original paper
        forecast_steps=60, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=30, #512
        input_size=112, # 112
        n_samples = 50000,
        num_workers = 32,
        batch_size = 5,
        learning_rate = 1e-2,
        num_att_layers = 4,
        plot_every = 100, #Plot every global_step
        rain_step = 0.1,
        momentum = 0.9,
        )
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
print(model)
wandb_logger = WandbLogger(project="lit-wandb")

trainer = pl.Trainer(track_grad_norm = 2, max_epochs=1000, gpus=-1,log_every_n_steps=1, logger = wandb_logger,strategy="ddp")

trainer.fit(model)
wandb.finish()
