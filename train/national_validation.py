from metnet import MetNet, MetNet2
import torch
from collections import defaultdict
from ocf_datapipes.training.metnet_national import metnet_national_datapipe
from torch.utils.data import DataLoader, default_collate
import argparse
import datetime
import numpy as np
import glob

def collate_fn(batch):
    x, y, start_time = batch
    collated_batch = default_collate((x,y))
    return (collated_batch[0], collated_batch[1], start_time)


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
        start_time=datetime.datetime(2021, 1, 1),
        end_time=datetime.datetime(2021, 12, 31),
        use_sun=args.sun,
        use_nwp=args.nwp,
        use_sat=args.sat,
        use_hrv=args.hrv,
        use_pv=args.pv,
        use_topo=args.topo,
        mode="val"
    )
    dataloader = DataLoader(
        dataset=datapipe, batch_size=args.batch, pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn
    )
    # Get the shape of the batch
    batch = next(iter(dataloader))
    print(batch[2])
    input_channels = batch[0].shape[
        2
    ]  # [Batch. Time, Channel, Width, Height] for now assume square
    print(f"Number of input channels: {input_channels}")
    # Validation steps
    model = MetNet.from_pretrained("openclimatefix/metnet-uk-metoffice-eumetsat-hrv-sun-pv-topo-256")
    model.eval()

    """
    
    Comparison One (divide by 13852 to get %):
    
    val
    me=-65.77421516033499
    mae=189.47224986708292
    rmse=383.94737252385636
    
    train
    me=-50.971601258979554
    mae=232.5513672145747
    rmse=456.4136986336792
    
    """

    # Now iterate through all times
    first_batch = None
    first_x = None
    first_y = None
    loss_fn = torch.nn.MSELoss() # MSE
    mae_loss = torch.nn.L1Loss() # MAE

    # Loss by timestep into future
    # Change to save every forecast, and every ground truth + time period for it
    # Calculate the errors after, so save out all of them
    # Save as init time -> [[forecast],[truth],[mae],[rmse],[mse]]
    import copy
    per_step_losses = [{} for _ in range(96)]
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, start_time = batch
            for f in range(96):
                y_hat = model(x, f)
                mse = loss_fn(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f+1, 0])
                rmse = torch.sqrt(mse)
                mae = mae_loss(torch.mean(y_hat, dim=(1, 2, 3)), y[:, f+1, 0])
                per_step_losses[f][start_time] = [y_hat.detach().numpy(),
                                                  y.detach().numpy(),
                                                  mse.detach().numpy(),
                                                  rmse.detach().numpy(),
                                                  mae.detach().numpy()]
            if i > 11663: # Gone over the 9 months, so break
                break
    np.save("metnet_all_uses", per_step_losses, allow_pickle=True)
    # Save out to disk
    import json
    with open(f"metnet{'_2' if args.use_2 else ''}_inchannels{input_channels}_step{args.steps}"
        f"_size{args.size}"
        f"_sun{args.sun}"
        f"_sat{args.sat}"
        f"_hrv{args.hrv}"
        f"_nwp{args.nwp}"
        f"_pv{args.pv}"
        f"_topo{args.topo}"
        f"_fp16{args.fp16}"
        f"_effectiveBatch{args.batch * args.accumulate}", 'w') as fout:
        json.dump(per_step_losses, fout)


