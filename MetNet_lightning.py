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
        hidden_dim=8, #384 original paper
        forecast_steps=1, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=6, #512
        input_size=112, # 112
        n_samples = 100,
        num_workers = 8,
        batch_size = 1,
        learning_rate = 1e-2
        )
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
print(model)
wandb_logger = WandbLogger(project="lit-wandb")

trainer = pl.Trainer(track_grad_norm = 2, max_epochs=5000, gpus=-1,log_every_n_steps=1, logger = wandb_logger,strategy="ddp_find_unused_parameters_false")

trainer.fit(model)
wandb.finish()
