from metnet import MetNet, MetNet2
from torchinfo import summary
import torch
import torch.nn.functional as F
from ocf_datapipes.training.metnet_national import metnet_national_datapipe
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Dataprep
datapipe = metnet_national_datapipe("national.yaml")

dataloader = DataLoader(dataset=datapipe, batch_size=4, pin_memory=True, num_workers=8)

class LitModel(pl.LightningModule):
    def __init__(self, use_metnet2: bool = False, input_channels=12, center_crop_size=64, input_size=256, forecast_steps=96, lr=1e-4):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.lr = lr
        if use_metnet2:
            self.model = MetNet2(output_channels=1, input_channels=input_channels, center_crop_size=center_crop_size, input_size=input_size, forecast_steps=forecast_steps) # every half hour for 48 hours
        else:
            self.model = MetNet(output_channels=1, input_channels=input_channels, center_crop_size=center_crop_size, input_size=input_size, forecast_steps=forecast_steps)

    def forward(self, x, forecast_step):
        return self.model(x, forecast_step)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, 0)
        # TODO Maybe weight loss by forecast step? So sooner ones are more important than later ones?
        loss = F.mse(torch.sum(y_hat), y[0])
        for f in range(self.forecast_steps):
            y_hat = self(x,f)
            loss += F.mse(torch.sum(y_hat), y[f])
        return loss / self.forecast_steps

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

batch = next(iter(datapipe))
input_channels = batch[0].shape[1] # [Time, Channel, Width, Height] for now assume square
trainer = pl.Trainer(max_epochs=50)
model = LitModel(input_channels=input_channels, input_size=batch[0].shape[2])

trainer.fit(model, train_dataloaders=dataloader)
torch.save(model.model, "metnet_uk_national")
