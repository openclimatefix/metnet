from metnet import MetNet, MetNet2
from torchinfo import summary
import torch
import torch.nn.functional as F
import datetime
from ocf_datapipes.training.metnet_national import metnet_national_datapipe
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse

class LitModel(pl.LightningModule):
    def __init__(self, use_metnet2: bool = False, input_channels=12, center_crop_size=64, input_size=256, forecast_steps=96, lr=1e-4):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.learning_rate = lr
        if use_metnet2:
            self.model = MetNet2(output_channels=1, input_channels=input_channels, center_crop_size=center_crop_size, input_size=input_size, forecast_steps=forecast_steps, use_preprocessor=False) # every half hour for 48 hours
        else:
            self.model = MetNet(output_channels=1, input_channels=input_channels, center_crop_size=center_crop_size, input_size=input_size, forecast_steps=forecast_steps, use_preprocessor=False)

    def forward(self, x, forecast_step):
        return self.model(x, forecast_step)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, 0)
        # TODO Maybe weight loss by forecast step? So sooner ones are more important than later ones?
        loss_fn = torch.nn.MSELoss()
        print(f"Y-hat: {y_hat.shape} After Sum: {torch.sum(y_hat, dim=0).shape} Target: {y[:][0].shape} Zeroed: {y[:][0][0].shape}")
        loss = loss_fn(torch.sum(y_hat, dim=0), y[:][0][0])
        for f in range(self.forecast_steps):
            y_hat = self(x,f)
            loss += loss_fn(torch.sum(y_hat, dim=0), y[:][f][0])
        return loss / self.forecast_steps

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_2", action="store_true", help="Use MetNet-2")
    parser.add_argument("--config", default="national.yaml")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accumulate", type=int, default=1)
    args = parser.parse_args()
    # Dataprep
    datapipe = metnet_national_datapipe(args.config, start_time=datetime.datetime(2020,1,1), end_time=datetime.datetime(2020,12,31))
    val_datapipe = metnet_national_datapipe(args.config, start_time=datetime.datetime(2021,1,1), end_time=datetime.datetime(2022,12,31))
    dataloader = DataLoader(dataset=datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset=val_datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers)
    # Get the shape of the batch
    batch = next(iter(datapipe))
    input_channels = batch[0].shape[1] # [Time, Channel, Width, Height] for now assume square
    print(f"Number of input channels: {input_channels}")
    # Validation steps
    model_checkpoint = ModelCheckpoint(monitor="loss")
    early_stopping = EarlyStopping(monitor="loss")
    trainer = pl.Trainer(max_epochs=args.epochs,
                         precision=16 if args.fp16 else 32,
                         devices=args.num_gpu,
                         accelerator="auto",
                         auto_select_gpus=True,
                         auto_lr_find=False,
                         log_every_n_steps=1,
                         limit_val_batches=400*args.accumulate,
                         limit_train_batches=8000*args.accumulate,
                         accumulate_grad_batches=args.accumulate,
                         callbacks=[model_checkpoint, early_stopping])
    model = LitModel(input_channels=input_channels, input_size=batch[0].shape[2], use_metnet2=args.use_2)
    # trainer.tune(model)
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    torch.save(model.model, f"metnet{'-2' if args.use_2 else ''}_uk_national")
