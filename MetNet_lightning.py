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
# MetNet expects original HxW to be 4x the input size
#x = torch.randn((1, 7, 16, 128, 128))
print(model)

trainer = pl.Trainer(fast_dev_run= True)
input("train? press enter to continue...")
trainer.fit(model)
'''x = torch.randn((1, 7, 15, 112, 112))
#x = torch.cat([torch.zeros(1, 7, 10, 32, 32),torch.randn(1,7,8,32,32)], dim=2)
out = model(x)
# MetNet creates predictions for the center 1/4th
y = torch.randn((1, 6, 51, 28, 28))
print(F.mse_loss(out, y))
F.mse_loss(out, y).backward()'''
