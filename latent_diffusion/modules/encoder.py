import torch as pt
from torch import nn
from .resblock import ResBlock
from .attention import AttnBlock


class Encoder(nn.Module):
    """
    Encoder Module
    Parameters:
        resolution (`int`): Input resolution. Default is 224
        in_channels (`int`): Number of input channels
        out_channels (`int`): Number of output channels
        num_resblocks (`int`): Number of ResNet Blocks
        ch_factor (`int`): initial channnel factor. Default is 64
        ch_mult (`List[int]`): List of channel multipliers
        attn_resolution `List[int]`:
            list of resolutions from which to start applying attention
        mid_block (`bool`): Allow middle attention block (VQGAN)

    """
    def __init__(
            self,
            resolution=224,
            in_channels=3,
            out_channels=256,
            num_resblock=3,
            attn_resolution=[28, 56, 112],
            ch_factor=64,
            ch_mult=[1, 2, 4, 8],
            mid_block=False
    ):
        super(Encoder, self).__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblock = num_resblock
        self.ch_factor = ch_factor
        self.ch_mult = ch_mult
        self.attn_resolution = attn_resolution
        current_resolution = self.resolution
        self.in_conv = nn.Conv2d(
            in_channels, self.ch_factor, kernel_size=3, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.layers = nn.ModuleDict()
        for i, _ in enumerate(self.ch_mult):
            down_block = nn.ModuleDict()
            block = nn.ModuleDict()
            block_in = ch_factor*in_ch_mult[i]
            block_out = ch_factor*self.ch_mult[i]
            for j in range(self.num_resblock):
                block.add_module(
                    f"resblock_{i}_{j}", ResBlock(block_in, block_out))
                block_in = block_out
            down_block.add_module(f"resblock_{i}", block)
            if current_resolution in self.attn_resolution:
                down_block.add_module(f"attnblock_{i}", AttnBlock(block_out))

            if i != len(self.ch_mult)-1:
                down_block.add_module(
                    "downblock", ResBlock(block_out, mode="Down"))
                current_resolution = current_resolution//2
            self.layers.add_module(f"DownBlock_{i}", down_block)

        if not mid_block:
            self.layers.add_module("OutNorm", nn.BatchNorm2d(block_out))
            self.layers.add_module(
                "OutConv",
                nn.Conv2d(block_out, out_channels, kernel_size=3, padding=1))

    def forward(self, x: pt.Tensor):
        """
        The [`Encoder`] forward method
        Args:
            x (`Tensor`): Input Tesnsor of shape [B, C, H, W]
        Returns:
            [`Tensor`]:
                Encoder output
        """
        h = self.in_conv(x)
        for i, _ in enumerate(self.ch_mult):
            for j in range(self.num_resblock):
                h = self.layers[f"DownBlock_{i}"][f"resblock_{i}"][f"resblock_{i}_{j}"](h)

            if f"attnblock_{i}" in self.layers[f"DownBlock_{i}"].keys():
                _, h = self.layers[f"DownBlock_{i}"][f"attnblock_{i}"](h)

            if "downblock" in self.layers[f"DownBlock_{i}"].keys():
                h = self.layers[f"DownBlock_{i}"]["downblock"](h)
        h = self.layers["OutNorm"](h)
        h = self.layers["OutConv"](h)
        return h
