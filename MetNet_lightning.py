from metnet.models.metnet_pylight import MetNetPylight
import torch
import torch.nn.functional as F
from data_prep.prepare_data_MetNet import load_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import time

wandb.login()

model = MetNetPylight(
        hidden_dim=256, #384 original paper
        forecast_steps=1, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=128, #512
        input_size=112, # 112
        n_samples = None, #None = All ~ 23000
        num_workers = 4,
        batch_size = 8,
        learning_rate = 1e-2,
        num_att_layers = 4,
        plot_every = None, #Plot every global_step
        rain_step = 0.2,
        momentum = 0.9,
        att_heads=16,
        keep_biggest = 1,
        leadtime_spacing = 12, #1: 5 minutes, 3: 15 minutes
        )
#PATH_cp = "/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/8leads with agg.ckpt"
#PATH_cp = "/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/full_run_continue.ckpt"
#model = MetNetPylight.load_from_checkpoint(PATH_cp)
model.testing = False
model.learning_rate = 1e-2
'''print(model)
print(model.forecast_steps)
print(model.input_channels)
print(model.output_channels)
print(model.n_samples)
print(model.file_name)
print(model.keep_biggest)
input()
model.keep_biggest = 0.9
model.n_samples = None'''
#model.printer = True
#model.plot_every = None
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
#print(model)

#wandb.restore("/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/wandb/run-20220331_162434-21o2u2sj"
#wandb.init(run_id = "21o2u2sj",resume="must")
wandb_logger = WandbLogger(project="lit-wandb", log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="validation/loss_epoch", mode="min", save_top_k=5)


trainer = pl.Trainer(num_sanity_val_steps=2, track_grad_norm = 2, max_epochs=1000, gpus=-1,log_every_n_steps=50, logger = wandb_logger,strategy="ddp", callbacks=[checkpoint_callback])
start_time = time.time()

trainer.fit(model)
print("--- %s seconds ---" % (time.time() - start_time))
wandb.finish()
