import torch as pt
from torch import nn
from .resblock import ResBlock

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            ResBlock(in_channels, 32),
            ResBlock(32, 32),
            nn.ConvTranspose2d(32, 64, 2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64, 64),
            ResBlock(64, 128),
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128, 64),
            ResBlock(64, 32),
            nn.ConvTranspose2d(32, out_channels, 2, stride=2),
        )
    
    def forward(self, x): 
        return self.layers(x)