import torch
try:
    torch.multiprocessing.set_start_method('spawn')
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import fsspec.asyn
from ocf_datapipes.training.metnet_pv_site import metnet_site_datapipe
from metnet.models import MetNetSingleShot
import matplotlib
matplotlib.use('agg')
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import datetime
from torch.utils.data import DataLoader, default_collate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.
    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!
    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0
    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None

def worker_init_fn(worker_id):
    """Configures each dataset worker process.
    1. Get fsspec ready for multi process
    2. To call NowcastingDataset.per_worker_init().
    """
    # fix for fsspec when using multprocess
    set_fsspec_for_multiprocess()
def mse_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MSE for each forecast horizon

    Args:
        output: The model estimate of size (batch_size, forecast_length)
        target: The truth of size (batch_size, forecast_length)

    Returns: A tensor of size (forecast_length)

    """
    return torch.mean((output - target) ** 2, dim=0)


def mae_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MAE for each forecast horizon

    Args:
        output: The model estimate of size (batch_size, forecast_length)
        target: The truth of size (batch_size, forecast_length)

    Returns: A tensor of size (forecast_length)

    """
    return torch.mean(torch.abs(output - target), dim=0)


torch.set_float32_matmul_precision('medium')

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
        forecast_steps=70,
            hidden_dim=2048,
            att_layers=2,
        lr=1e-4,

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

    def forward(self, x):
        return F.relu(self.pooler(self.model(x))[:,:,0,0])

    def training_step(self, batch, batch_idx):
        tag = "train"
        x, y = batch
        y = y[0]
        x = torch.nan_to_num(input=x, posinf=1.0, neginf=0.0)
        y = torch.nan_to_num(input=y, posinf=1.0, neginf=0.0)
        x = x.half()
        y = y.half()
        y = y[:,1:,0] # Take out the T0 output
        y_hat = self(x)
        #loss = self.weighted_losses.get_mse_exp(y_hat, y)
        #self.log("loss", loss)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        loss = nmae_loss
        self.log("loss", loss)
        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
            },
            #on_step=True,
            #on_epoch=True,
            #sync_dist=True  # Required for distributed training
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
            #on_step=True,
            #on_epoch=True,
            #sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if batch_idx % 100 == 0:  # Log every 100 batches
            self.log_tb_images((x, y, y_hat, [batch_idx for _ in range(x.shape[0])]))
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
        for img_idx, (_, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            fig = plt.figure()
            plt.plot(list(range(360)), y_pred.cpu().detach().numpy(), label="Forecast")
            plt.plot(list(range(360)), y_true.cpu().detach().numpy(), label="Truth")
            plt.title("GT vs Pred PV Site Single Shot")
            plt.legend(loc="best")
            tb_logger.add_figure(f"GT_Vs_Pred/{img_idx}", fig, batch_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_2", action="store_true", help="Use MetNet-2")
    parser.add_argument("--config", default="pv_site.yaml")
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
    parser.add_argument("--steps", type=int, default=360, help="Number of forecast steps per pass")
    parser.add_argument("--size", type=int, default=256, help="Input Size in pixels")
    parser.add_argument("--center_size", type=int, default=64, help="Center Crop Size")
    parser.add_argument("--center_meter", type=int, default=64_000, help="Center Crop Size")
    parser.add_argument("--context_meter", type=int, default=512_000, help="Center Crop Size")
    parser.add_argument("--cpu", action="store_true", help="Force run on CPU")
    parser.add_argument("--accumulate", type=int, default=1)
    args = parser.parse_args()
    skip_num = 1 #int(360 / args.steps)
    # Dataprep
    datapipe = metnet_site_datapipe(
        args.config,
        start_time=datetime.datetime(2014, 1, 1),
        end_time=datetime.datetime(2020, 12, 31),
        use_sun=args.sun,
        use_nwp=args.nwp,
        use_sat=args.sat,
        use_hrv=args.hrv,
        use_pv=True,
        use_topo=args.topo,
        pv_in_image=True,
        output_size=args.size,
        center_size_meters=args.center_meter,
        context_size_meters=args.context_meter
    )
    dataloader = DataLoader(
        dataset=datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers
    )
    """
    val_datapipe = metnet_site_datapipe(
        args.config,
        start_time=datetime.datetime(2021, 1, 1),
        end_time=datetime.datetime(2022, 12, 31),
        use_sun=args.sun,
        use_nwp=args.nwp,
        use_sat=args.sat,
        use_hrv=args.hrv,
        use_pv=True,
        use_topo=args.topo,
        pv_in_image=True,
        output_size=args.size
    )
    val_dataloader = DataLoader(
        dataset=val_datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers
    )
    """
    # Get the shape of the batch
    batch = next(iter(dataloader))
    input_channels = batch[0].shape[
        2
    ]  # [Batch. Time, Channel, Width, Height] for now assume square
    print(f"Number of input channels: {input_channels}")
    # Validation steps
    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=100,
        monitor="step",
        mode="max",
        save_last=True,
        save_top_k=10,
        dirpath=f"./pv_metnet_single_shot_nmae_relu_30hour_inchannels{input_channels}"
                f"_step{args.steps}"
                f"_size{args.size}"
                f"_sun{args.sun}"
                f"_sat{args.sat}"
                f"_hrv{args.hrv}"
                f"_nwp{args.nwp}"
                f"_pv{True}"
                f"_topo{args.topo}"
                f"_fp16{args.fp16}"
                f"_effectiveBatch{args.batch*args.accumulate}_att{args.att}_hidden{args.hidden}_centerm{args.center_meter}_contextm{args.context_meter}",
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
        forecast_steps=args.steps

    )  # , forecast_steps=args.steps*4) #.load_from_checkpoint("/mnt/storage_ssd_4tb/metnet_models/metnet_inchannels44_step8_size256_sunTrue_satTrue_hrvTrue_nwpTrue_pvTrue_topoTrue_fp16True_effectiveBatch16/epoch=0-step=1800.ckpt")  # , forecast_steps=args.steps*4)
    # trainer.tune(model)
    #model = model.load_from_checkpoint("/mnt/storage_ssd_4tb/metnet_models/metnet_gsp_single_shot_nmae_relu_inchannels44_step96_size256_sunTrue_satTrue_hrvTrue_nwpTrue_pvFalse_topoTrue_fp16True_effectiveBatch36_att2_hidden512/last-v1.ckpt")
    trainer.fit(model, train_dataloaders=dataloader)

