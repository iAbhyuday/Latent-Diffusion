import torch as pt
from torch import nn
from .resblock import ResBlock
from .attention import AttnBlock


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels=None, scale_factor: int = 2):
        super(Upsample, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale_factor

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.UpsamplingNearest2d(scale_factor=self.scale),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ch_factor: int,
        ch_mult=[1, 2, 4, 8],
        num_resblock: int = 3,
        attn_resolution=[],
        resolution=4,
        mid_block: bool = False,
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