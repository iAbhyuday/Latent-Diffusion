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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resblock = num_resblock
        self.num_ch_mult = len(ch_mult)
        self.attn_resolution = attn_resolution
        block_in = ch_factor * self.ch_mult[-1]
        current_resolution = resolution // 2 ** (self.num_ch_mult - 1)
        self.in_conv = nn.Conv2d(in_channels, block_in, kernel_size=3, padding=1)

        self.layers = nn.ModuleDict()
        for i in reversed(range(self.num_ch_mult)):
            block_out = ch_factor * ch_mult[i]
            upblock = nn.ModuleDict()
            block = nn.ModuleDict()

            for j in range(self.num_resblock):
                block.add_module(f"resblock_{i}_{j}", ResBlock(block_in, block_out))
                block_in = block_out
            upblock.add_module(f"resblock_{i}", block)
            if current_resolution in self.attn_resolution:
                upblock.add_module(f"attn_{i}", AttnBlock(block_in))
            if i != 0:
                upblock.add_module(f"upsample_{i}", Upsample(block_in))
                current_resolution = current_resolution * 2
            self.layers.add_module(f"UpBlock_{i}", upblock)

        if not mid_block:
            self.layers.add_module("OutNorm", nn.BatchNorm2d(block_in))
            self.layers.add_module(
                "OutConv", nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x: pt.Tensor):
        h = self.in_conv(x)
        for i in reversed(range(self.num_ch_mult)):
            for j in range(self.num_resblock):
                h = self.layers[f"UpBlock_{i}"][f"resblock_{i}"][f"resblock_{i}_{j}"](h)
            if f"attnblock_{i}" in self.layers[f"UpBlock_{i}"].keys():
                _, h = self.layers[f"UpBlock_{i}"][f"attnblock_{i}"](h)
            if f"upsample_{i}" in self.layers[f"UpBlock_{i}"].keys():
                h = self.layers[f"UpBlock_{i}"][f"upsample_{i}"](h)
        h = self.layers["OutNorm"](h)
        h = self.layers["OutConv"](h)

        return h
