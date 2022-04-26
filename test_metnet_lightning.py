from metnet.models.metnet_pylight import MetNetPylight
import torch
import torch.nn.functional as F
from data_prep.prepare_data_MetNet import load_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning import seed_everything
import time
import numpy as np
import matplotlib.pyplot as plt

if False:
    
    thresh = 1
    control = np.load(f"f1_control_thresh_{thresh}.npy")
    fig, ax = plt.subplots(1,1)
    ax.plot(control, label="persistence")
    print(control)
    f1s  = np.load(f"f1_threshed_0.5_thresh_{thresh}.npy")
    ax.plot(f1s, label=f"No aggregation model")
    print(f1s)
    fig.suptitle(f"F1-score for rainfall threshed at {round(thresh*0.2,3)} mm/h")
    
    ax.set_xlabel("Lead time")
    ax.set_ylabel("F1")
    ax.legend()
    plt.show()
if False:
    print("hej")
    N = 3606
    prob_threshes = [0, 0.1,0.2,0.3, 0.4, 0.5]
    control = np.load(f"f1_control_N_{N}.npy")
    fig, ax = plt.subplots(1,1)
    ax.plot(control, label="persistence")
    for p in prob_threshes:
        f1s  = np.load(f"f1_threshed_{p}_N_{N}.npy")
        ax.plot(f1s, label=f"P(rate>=0.2)>{p}")
       
    fig.suptitle("Different probabillity thresholds for precipitation")
    ax.set_xlabel("Lead time")
    ax.set_ylabel("F1")
    ax.legend()
    plt.show()
      

wandb.login()

'''model = MetNetPylight(
        hidden_dim=256, #384 original paper
        forecast_steps=60, #240 original paper
        input_channels=15, #46 original paper, hour/day/month = 3, lat/long/elevation = 3, GOES+MRMS = 40
        output_channels=128, #512
        input_size=112, # 112
        n_samples = None, #None = All ~ 23000
        num_workers = 4,
        batch_size = None, #None = 8
        learning_rate = 1e-2,
        num_att_layers = 8,
        plot_every = None, #Plot every global_step
        rain_step = 0.2,
        momentum = 0.9,
        att_heads=16,
        keep_biggest = 0.3,
        )'''
#PATH_cp = "epoch=653-step=90251.ckpt"
#PATH_cp = "epoch=429-step=59339.ckpt"
#PATH_cp = "epoch=276-step=14680.ckpt"
#PATH_cp = "epoch=61-step=3285.ckpt"
#PATH_cp = "epoch=464-step=24644.ckpt"

#PATH_cp = "epoch=471-step=25015.ckpt"
#PATH_cp = "epoch=33-step=3569.ckpt"
#PATH_cp = "epoch=430-step=22842.ckpt"
# 
#PATH_cp = "8leadtimessecond8h.ckpt"
#PATH_cp = "epoch=242-step=16766.ckpt"
PATH_cp = "fullrun_1.ckpt"
#PATH_cp = "best_60_leadtime.ckpt"
#PATH_cp = "best_single_leadtime.ckpt"
#PATH_cp = "no agg network.ckpt"


model = MetNetPylight.load_from_checkpoint(PATH_cp)
#model.forecast_steps=8
#model.leadtime_spacing = 3
#model.keep_biggest = 0.15
model.thresh = 1
model.batch_size = 8
model.n_samples = None
model.testing = True
print(model.forecast_steps)
#model.plot_every = None
#MetNetPylight expects already preprocessed data. Can be change by uncommenting the preprocessing step.
#print(model)
model.TPs = np.zeros(model.forecast_steps)
model.FNs = np.zeros(model.forecast_steps)
model.FPs = np.zeros(model.forecast_steps)
model.TPs_control = np.zeros(model.forecast_steps)
model.FNs_control = np.zeros(model.forecast_steps)
model.FPs_control = np.zeros(model.forecast_steps)

model.f1s = [[] for _ in range(model.forecast_steps)]
model.f1s_control = [[] for _ in range(model.forecast_steps)]
model.f1_count = [0 for _ in range(model.forecast_steps)]
model.avg_y_img = [0 for _ in range(model.forecast_steps)]
model.avg_y_hat_img = [0 for _ in range(model.forecast_steps)]
model.skipped = 0
model.not_skipped = 0

    
wandb_logger = WandbLogger(project="lit-wandb")



trainer = pl.Trainer(track_grad_norm = 2, max_epochs=1000, gpus=-1,log_every_n_steps=10, logger = wandb_logger,strategy="ddp")



start_time = time.time()
trainer.test(model)
print("--- %s seconds ---" % (time.time() - start_time))
wandb.finish()
