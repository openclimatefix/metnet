# MetNet and MetNet-2
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

PyTorch Implementation of Google Research's MetNet for short term weather forecasting (https://arxiv.org/abs/2003.12140), inspired from https://github.com/tcapelle/metnet_pytorch/tree/master/metnet_pytorch

MetNet-2 (https://arxiv.org/pdf/2111.07470.pdf) is a further extension of MetNet that takes in a larger context image to predict up to 12 hours ahead, and is also implemented in PyTorch here.

## Installation

Clone the repository, then run
```shell
pip install -r requirements.txt
pip install -e .
````

Alternatively, you can also install a usually older version through ```pip install metnet```

Please ensure that you're using Python version 3.9 or above.

## Data

While the exact training data used for both MetNet and MetNet-2 haven't been released, the papers do go into some detail as to the inputs, which were GOES-16 and MRMS precipitation data, as well as the time period covered. We will be making those splits available, as well as a larger dataset that covers a longer time period, with [HuggingFace Datasets](https://huggingface.co/datasets/openclimatefix/goes-mrms)! Note: The dataset is not available yet, we are still processing data!

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/goes-mrms")
```

This uses the publicly avaiilable GOES-16 data and the MRMS archive to create a similar set of data to train and test on, with various other splits available as well.

## Pretrained Weights
Pretrained model weights for MetNet and MetNet-2 have not been publicly released, and there is some difficulty in reproducing their training. We release weights for both MetNet and MetNet-2 trained on cloud mask and satellite imagery data with the same parameters as detailed in the papers on HuggingFace Hub for [MetNet](https://huggingface.co/openclimatefix/metnet) and [MetNet-2](https://huggingface.co/openclimatefix/metnet-2). These weights can be downloaded and used using:

```python
from metnet import MetNet, MetNet2
model = MetNet().from_pretrained("openclimatefix/metnet")
model = MetNet2().from_pretrained("openclimatefix/metnet-2")
```

## Example Usage

MetNet can be used with:

```python
from metnet import MetNet
import torch
import torch.nn.functional as F

model = MetNet(
        hidden_dim=32,
        forecast_steps=24,
        input_channels=16,
        output_channels=12,
        sat_channels=12,
        input_size=32,
        )
# MetNet expects original HxW to be 4x the input size
x = torch.randn((2, 12, 16, 128, 128))
out = []
for lead_time in range(24):
        out.append(model(x, lead_time))
out = torch.stack(out, dim=1)
# MetNet creates predictions for the center 1/4th
y = torch.randn((2, 24, 12, 8, 8))
F.mse_loss(out, y).backward()
```

And MetNet-2 with:

```python
from metnet import MetNet2
import torch
import torch.nn.functional as F

model = MetNet2(
        forecast_steps=8,
        input_size=64,
        num_input_timesteps=6,
        upsampler_channels=128,
        lstm_channels=32,
        encoder_channels=64,
        center_crop_size=16,
        )
# MetNet expects original HxW to be 4x the input size
x = torch.randn((2, 6, 12, 256, 256))
out = []
for lead_time in range(8):
        out.append(model(x, lead_time))
out = torch.stack(out, dim=1)
y = torch.rand((2,8,12,64,64))
F.mse_loss(out, y).backward()
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Bieker"/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/metnet/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jack-kelly.com"><img src="https://avatars.githubusercontent.com/u/460756?v=4?s=100" width="100px;" alt="Jack Kelly"/><br /><sub><b>Jack Kelly</b></sub></a><br /><a href="https://github.com/openclimatefix/metnet/commits?author=JackKelly" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ValterFallenius"><img src="https://avatars.githubusercontent.com/u/21970939?v=4?s=100" width="100px;" alt="Valter Fallenius"/><br /><sub><b>Valter Fallenius</b></sub></a><br /><a href="#userTesting-ValterFallenius" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/terigenbuaa"><img src="https://avatars.githubusercontent.com/u/91317406?v=4?s=100" width="100px;" alt="terigenbuaa"/><br /><sub><b>terigenbuaa</b></sub></a><br /><a href="#question-terigenbuaa" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NMC-DAVE"><img src="https://avatars.githubusercontent.com/u/26354668?v=4?s=100" width="100px;" alt="Kan.Dai"/><br /><sub><b>Kan.Dai</b></sub></a><br /><a href="#question-NMC-DAVE" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SaileshBechar"><img src="https://avatars.githubusercontent.com/u/38445041?v=4?s=100" width="100px;" alt="Sailesh Bechar"/><br /><sub><b>Sailesh Bechar</b></sub></a><br /><a href="#question-SaileshBechar" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rahul-maurya11b"><img src="https://avatars.githubusercontent.com/u/98907006?v=4?s=100" width="100px;" alt="Rahul Maurya"/><br /><sub><b>Rahul Maurya</b></sub></a><br /><a href="https://github.com/openclimatefix/metnet/commits?author=rahul-maurya11b" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
