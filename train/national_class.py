from metnet import MetNet, MetNet2
from torchinfo import summary
import torch
import torch.nn.functional as F
from ocf_datapipes.training.metnet_national_class import metnet_national_datapipe
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import datetime
import numpy as np


def prediction2label(pred: np.ndarray):
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def ordinal_regression(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    return torch.nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        use_metnet2: bool = False,
        input_channels=12,
        center_crop_size=64,
        input_size=256,
        forecast_steps=96,
        hidden_dim=2048,
            att_layers=2,
        lr=3e-4,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.learning_rate = lr
        if use_metnet2:
            self.model = MetNet2(
                output_channels=1400,
                input_channels=input_channels,
                center_crop_size=center_crop_size,
                input_size=input_size,
                forecast_steps=forecast_steps,
                use_preprocessor=False,
            )  # every half hour for 48 hours
        else:
            self.model = MetNet(
                output_channels=1400,
                input_channels=input_channels,
                center_crop_size=center_crop_size,
                input_size=input_size,
                num_att_layers=att_layers,
                hidden_dim=hidden_dim,
                forecast_steps=forecast_steps,
                use_preprocessor=False,
            )

    def forward(self, x, forecast_step):
        return self.model(x, forecast_step)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.half()
        y = y.half()
        y = y[:, :, 0] // 1400 # Number of bins, should give bucket then?
        print(y)
        f = np.random.randint(1, skip_num + 1)  # Index 0 is the current generation
        y_hat = self(x, f-1) # torch.tensor(f-1).long().type_as(x))
        predictions = torch.mean(y_hat, dim=(2, 3))
        loss = ordinal_regression(predictions, y[:, f])
        self.log(f"forecast_step_{f-1}_loss", loss, on_step=True)
        total_num = 1
        fs = np.random.choice(list(range(f,97)), 96//skip_num)
        for i, f in enumerate(
            range(f + 1, 97, skip_num)
        ):  # Can only do 12 hours, so extend out to 48 by doing every 4th one
            y_hat = self(x, fs[i]-1) # torch.tensor(f-1).long().type_as(x))
            predictions = torch.mean(y_hat, dim=(2, 3))
            step_loss = ordinal_regression(predictions, y[:, fs[i]])
            loss += step_loss
            self.log(f"forecast_step_{fs[i]}_loss", step_loss, on_step=True)
            total_num += 1
        self.log("loss", loss / total_num)
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
    parser.add_argument("--topo", action="store_true")
    parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--att", type=int, default=2)
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
        use_topo=args.topo
    )
    dataloader = DataLoader(
        dataset=datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers
    )
    # Get the shape of the batch
    batch = next(iter(dataloader))
    input_channels = batch[0].shape[
        2
    ]  # [Batch. Time, Channel, Width, Height] for now assume square
    print(f"Number of input channels: {input_channels}")
    # Validation steps
    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        dirpath=f"/mnt/storage_ssd_4tb/metnet_models/metnet{'_2' if args.use_2 else ''}_class_inchannels{input_channels}"
                f"_step{args.steps}"
                f"_size{args.size}"
                f"_sun{args.sun}"
                f"_sat{args.sat}"
                f"_hrv{args.hrv}"
                f"_nwp{args.nwp}"
                f"_pv{args.pv}"
                f"_topo{args.topo}"
                f"_fp16{args.fp16}"
                f"_effectiveBatch{args.batch*args.accumulate}"
                f"_hidden{args.hidden}"
                f"_att_layers{args.att}",
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
        logger=tb_logger
    )
    model = LitModel(
        input_channels=input_channels,
        input_size=args.size,
        use_metnet2=args.use_2,
        center_crop_size=args.center_size,
        att_layers=args.att,
        hidden_dim=args.hidden
    )  # , forecast_steps=args.steps*4)
    # trainer.tune(model)
    trainer.fit(model, train_dataloaders=dataloader) #, val_dataloaders=val_dataloader)
