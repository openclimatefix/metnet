from metnet.models import MetNet, MetNet2, MetNetSingleShot
from torchinfo import summary
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


class LitModel(pl.LightningModule):
    def __init__(
        self,
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
        self.config = self.model.config
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.half()
        y = y.half()
        loss_fn = torch.nn.MSELoss()
        y_hat = self(x)
        loss = loss_fn(torch.mean(y_hat, dim=(2, 3)), y[:, :, 0])
        for i in range(1,97):
            step_loss = loss_fn(torch.mean(y_hat, dim=(2, 3))[:,i-1], y[:, i, 0])
            self.log(f"forecast_step_{i-1}_loss", step_loss, on_step=True)
        self.log("loss", loss)
        return loss

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
        dirpath=f"/mnt/storage_ssd_4tb/metnet_models/metnet_gsp_single_shot_inchannels{input_channels}"
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
