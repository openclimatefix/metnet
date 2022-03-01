from metnet.models.metnet_pylight import MetNetPylight
import torch
import torch.nn.functional as F
from data_prep.prepare_data_MetNet import load_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.login()

model = MetNetPylight(
        hidden_dim=256, #384 original paper
        forecast_steps=60, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=51, #512
        input_size=112, # 112
        n_samples = 1000,
        )
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
print(model)
wandb_logger = WandbLogger(project="lit-wandb")

trainer = pl.Trainer(max_epochs=10, gpus=-1,log_every_n_steps=20, logger = wandb_logger)

input("train? press enter to continue...")
trainer.fit(model)
wandb.finish()
