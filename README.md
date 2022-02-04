# MetNet and MetNet-2

PyTorch Implementation of Google Research's MetNet for short term weather forecasting (https://arxiv.org/abs/2003.12140), inspired from https://github.com/tcapelle/metnet_pytorch/tree/master/metnet_pytorch

MetNet-2 (https://arxiv.org/pdf/2111.07470.pdf) is a further extension of MetNet that takes in a larger context image to predict up to 12 hours ahead, and is also implemented in PyTorch here.

## Installation

Clone the repository, then run
```shell
pip install -r requirements.txt
pip install -e .
````

Alternatively, you can also install a usually older version through ```pip install metnet```


## Pretrained Weights
Pretrained model weights for MetNet and MetNet-2 have not been publicly released, and there is some difficulty in reproducing their training. We release weights for both MetNet and MetNet-2 trained on cloud mask and satellite imagery data with the same parameters as detailed in the papers on HuggingFace Hub for [MetNet](https://huggingface.co/openclimatefix/metnet) and [MetNet-2](https://huggingface.co/openclimatefix/metnet-2). These weights can be downloaded and used using:

```python
from metnet import MetNet, MetNet2
model = MetNet().from_pretrained("openclimatefix/metnet")
model = MetNet2().from_pretrained("openclimatefix/metnet-2")
```
