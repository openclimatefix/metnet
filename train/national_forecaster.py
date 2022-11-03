from metnet import MetNet, MetNet2
from torchinfo import summary
import torch
import torch.nn.functional as F
from ocf_datapipes.training.metnet_national import metnet_national_datapipe
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import datetime
import numpy as np


class LitModel(pl.LightningModule):
    def __init__(
        self,
        use_metnet2: bool = False,
        input_channels=12,
        center_crop_size=64,
        input_size=256,
        forecast_steps=96,
        lr=1e-4,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.learning_rate = lr
        if use_metnet2:
            self.model = MetNet2(
                output_channels=1,
                input_channels=input_channels,
                center_crop_size=center_crop_size,
                input_size=input_size,
                forecast_steps=forecast_steps,
                use_preprocessor=False,
            )  # every half hour for 48 hours
        else:
            self.model = MetNet(
                output_channels=1,
                input_channels=input_channels,
                center_crop_size=center_crop_size,
                input_size=input_size,
                forecast_steps=forecast_steps,
                use_preprocessor=False,
            )

    def forward(self, x, forecast_step):
        return self.model(x, forecast_step)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.half()
        y = y.half()
        f = np.random.randint(1, skip_num + 1)  # Index 0 is the current generation
        y_hat = self(x, f-1) # torch.tensor(f-1).long().type_as(x))
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f, 0])
        total_num = 1
        for i, f in enumerate(
            range(f + 1, 97, skip_num)
        ):  # Can only do 12 hours, so extend out to 48 by doing every 4th one
            y_hat = self(x, f-1) # torch.tensor(f-1).long().type_as(x))
            loss += loss_fn(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f, 0])
            total_num += 1
        return loss / total_num

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.half()
        y = y.half()
        f = np.random.randint(1, skip_num + 1)  # Index 0 is the current generation
        y_hat = self(x, torch.tensor(f).long().type_as(x))
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f, 0])
        total_num = 1
        for i, f in enumerate(
            range(f + 1, 97, skip_num)
        ):  # Can only do 12 hours, so extend out to 48 by doing every 4th one
            y_hat = self(x, torch.tensor(f-1).long().type_as(x))
            loss += loss_fn(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f, 0])
            total_num += 1
        return loss / total_num

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_2", action="store_true", help="Use MetNet-2")
    parser.add_argument("--config", default="national.yaml")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--nwp", action="store_true")
    parser.add_argument("--sat", action="store_true")
    parser.add_argument("--hrv", action="store_true")
    parser.add_argument("--pv", action="store_true")
    parser.add_argument("--sun", action="store_true")
    parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=96, help="Number of forecast steps per pass")
    parser.add_argument("--size", type=int, default=256, help="Input Size in pixels")
    parser.add_argument("--center_size", type=int, default=64, help="Center Crop Size")
    parser.add_argument("--cpu", action="store_true", help="Force run on CPU")
    parser.add_argument("--accumulate", type=int, default=1)
    args = parser.parse_args()
    skip_num = int(96 / args.steps)
    # Dataprep
    datapipe = metnet_national_datapipe(
        args.config,
        start_time=datetime.datetime(2020, 1, 1),
        end_time=datetime.datetime(2020, 12, 31),
        use_sun=args.sun,
        use_nwp=args.nwp,
        use_sat=args.sat,
        use_hrv=args.hrv,
        use_pv=args.pv,
    )
    #val_datapipe = metnet_national_datapipe(
    #    args.config,
    #    start_time=datetime.datetime(2021, 1, 1),
    #    end_time=datetime.datetime(2022, 12, 31),
    #    use_sun=args.sun,
    #    use_nwp=args.nwp,
    #    use_sat=args.sat,
    #    use_hrv=args.hrv,
    #    use_pv=args.pv,
    #)
    dataloader = DataLoader(
        dataset=datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers
    )
    #val_dataloader = DataLoader(
    #    dataset=val_datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers
    #)
    # Get the shape of the batch
    batch = next(iter(dataloader))
    input_channels = batch[0].shape[
        2
    ]  # [Batch. Time, Channel, Width, Height] for now assume square
    print(f"Number of input channels: {input_channels}")
    # Validation steps
    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        dirpath=f"/mnt/storage_ssd_4tb/metnet_models/metnet{'_2' if args.use_2 else ''}_inchannels{input_channels}"
                f"_step{args.steps}"
                f"_size{args.size}"
                f"_sun{args.sun}"
                f"_sat{args.sat}"
                f"_hrv{args.hrv}"
                f"_nwp{args.nwp}"
                f"_pv{args.pv}"
                f"_fp16{args.fp16}"
                f"_effectiveBatch{args.batch*args.accumulate}",
    )
    # early_stopping = EarlyStopping(monitor="loss")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16 if args.fp16 else 32,
        devices=[args.num_gpu] if not args.cpu else 1,
        accelerator="auto" if not args.cpu else "cpu",
        auto_select_gpus=False,
        auto_lr_find=False,
        log_every_n_steps=1,
        # limit_val_batches=400 * args.accumulate,
        # limit_train_batches=500 * args.accumulate,
        accumulate_grad_batches=args.accumulate,
        callbacks=[model_checkpoint],
    )
    model = LitModel(
        input_channels=input_channels,
        input_size=args.size,
        use_metnet2=args.use_2,
        center_crop_size=args.center_size,
    )  # , forecast_steps=args.steps*4)
    # trainer.tune(model)
    trainer.fit(model, train_dataloaders=dataloader) #, val_dataloaders=val_dataloader)
