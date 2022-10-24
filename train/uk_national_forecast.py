from metnet import MetNet, MetNet2
from torchinfo import summary

model = MetNet()
summary(model)

print("MetNet-2")
model = MetNet2()
summary(model)
