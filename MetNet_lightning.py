from metnet.models.metnet_pylight import MetNetPylight
import torch
import torch.nn.functional as F
from data_prep.prepare_data_MetNet import load_data
import pytorch_lightning as pl


model = MetNetPylight(
        hidden_dim=128, #384 original paper
        forecast_steps=6, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=51, #512
        input_size=112, # 112
        n_samples = 500,
        )
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
print(model)

trainer = pl.Trainer(fast_dev_run= True)
input("train? press enter to continue...")
trainer.fit(model)
