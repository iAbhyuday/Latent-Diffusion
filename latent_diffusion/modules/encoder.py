import torch as pt
from torch import nn
from latent_diffusion.models.resblock import ResBlock
# out = ()

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            
                nn.Conv2d(in_channels, 128, kernel_size=5, stride=2), 
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # 54
                nn.Conv2d(128, 256, kernel_size=5, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #  25
                ResBlock(64, 128, mode="conv"),
                ResBlock(128, 128),
                ResBlock(128, 128),
                ResBlock(128, 128),
                ResBlock(128, 128),

                ResBlock(128, 256, mode="conv"),
                ResBlock(256, 128),
                ResBlock(128, 256),

                ResBlock(256, 256, mode="conv"),
                ResBlock(256, 256),
                ResBlock(256, 256),
                nn.Conv2d(256, 2, 1)
                )

    def forward(self, x):
        return self.layers(x)
