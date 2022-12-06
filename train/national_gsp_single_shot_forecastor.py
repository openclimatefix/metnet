from metnet.models import MetNet, MetNet2, MetNetSingleShot
from torchinfo import summary
import matplotlib
matplotlib.use('agg')
import torch
import torch.nn.functional as F
from ocf_datapipes.training.metnet_gsp_national import metnet_national_datapipe
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import datetime
import numpy as np
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader, default_collate
import matplotlib.pyplot as plt
from nowcasting_utils.models.loss import WeightedLosses
from nowcasting_utils.models.metrics import (
    mae_each_forecast_horizon,
    mse_each_forecast_horizon,
)

def collate_fn(batch):
    x, y, start_time = batch
    collated_batch = default_collate((x, y))
    return (collated_batch[0], collated_batch[1], start_time)

class LitModel(pl.LightningModule):
    def __init__(
        self,
            use_2: bool = False,
        input_channels=42,
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
        self.model = MetNetSingleShot(
            output_channels=forecast_steps,
            input_channels=input_channels,
            center_crop_size=center_crop_size,
            input_size=input_size,
            forecast_steps=forecast_steps,
            use_preprocessor=False,
            num_att_layers=att_layers,
            hidden_dim=hidden_dim,
        )
        self.pooler = torch.nn.AdaptiveAvgPool2d(1)
        self.config = self.model.config
        self.save_hyperparameters()
        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_steps)

    def forward(self, x):
        return self.pooler(self.model(x))

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, y = batch
        x = x.half()
        y = y.half()
        y = y[:,1:,0] # Take out the T0 output
        y_hat = self(x)
        loss = self.weighted_losses.get_mse_exp(y_hat, y)
        self.log("loss", loss)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
                f"MSE_EXP/{tag}": mse_exp,
                f"MAE_EXP/{tag}": mae_exp,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        # add metrics for each forecast horizon
        mse_each_forecast_horizon_metric = mse_each_forecast_horizon(output=y_hat, target=y)
        mae_each_forecast_horizon_metric = mae_each_forecast_horizon(output=y_hat, target=y)

        metrics_mse = {
            f"MSE_forecast_horizon_{i}/{tag}": mse_each_forecast_horizon_metric[i]
            for i in range(self.forecast_steps)
        }
        metrics_mae = {
            f"MAE_forecast_horizon_{i}/{tag}": mae_each_forecast_horizon_metric[i]
            for i in range(self.forecast_steps)
        }

        self.log_dict(
            {**metrics_mse, **metrics_mae},
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if batch_idx % 10:  # Log every 10 batches
            self.log_tb_images((x, y, y_hat, batch_idx))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def log_tb_images(self, viz_batch) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

            # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            fig = plt.figure()
            fig.plot(list(range(96)), y_pred.cpu().detach().numpy(), label="Forecast")
            fig.plot(list(range(96)), y_true[:, :, 0].cpu().detach().numpy(), label="Truth")
            fig.title("GT vs Pred National GSP Single Shot")
            fig.legend(loc="best")
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            tb_logger.add_image(f"GT_Vs_Pred/{batch_idx}_{img_idx}", data, batch_idx)
            tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, batch_idx)
            tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, batch_idx)
            plt.clf()
            plt.cla()


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
    parser.add_argument("--gsp", action="store_true")
    parser.add_argument("--sun", action="store_true")
    parser.add_argument("--topo", action="store_true")
    parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--att", type=int, default=2)
    parser.add_argument("--num_pv_systems", type=int, default=1000)
    parser.add_argument("--num_embedding_size", type=int, default=30000)
    parser.add_argument("--pv_out_channels", type=int, default=256)
    parser.add_argument("--pv_embed_channels", type=int, default=256)
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
        use_gsp=args.gsp,
        use_topo=args.topo,
        gsp_in_image=True,
        output_size=args.size
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
    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        monitor="steps",
        mode="max",
        save_last=True,
        save_top_k=10,
        dirpath=f"/mnt/storage_ssd_4tb/metnet_models/metnet_gsp_single_shot_weighted_inchannels{input_channels}"
                f"_step{args.steps}"
                f"_size{args.size}"
                f"_sun{args.sun}"
                f"_sat{args.sat}"
                f"_hrv{args.hrv}"
                f"_nwp{args.nwp}"
                f"_pv{args.pv}"
                f"_topo{args.topo}"
                f"_fp16{args.fp16}"
                f"_effectiveBatch{args.batch*args.accumulate}_att{args.att}_hidden{args.hidden}",
    )
    from pytorch_lightning import loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
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
        center_crop_size=args.center_size,
        att_layers=args.att,
        hidden_dim=args.hidden,

    )  # , forecast_steps=args.steps*4) #.load_from_checkpoint("/mnt/storage_ssd_4tb/metnet_models/metnet_inchannels44_step8_size256_sunTrue_satTrue_hrvTrue_nwpTrue_pvTrue_topoTrue_fp16True_effectiveBatch16/epoch=0-step=1800.ckpt")  # , forecast_steps=args.steps*4)
    # trainer.tune(model)
    trainer.fit(model, train_dataloaders=dataloader) #, val_dataloaders=val_dataloader)
