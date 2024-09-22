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
    def __init__(self, ch, out_channels, num_resblocks,
                 attn_resolutions, in_channels,
                 resolution, ch_mult=(1,2,4,8),
                ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_resblocks
        self.resolution = resolution
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = nn.Conv2d(in_channels,
                                 block_in,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResBlock(in_channels=block_in,
                                      out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.BatchNorm2d(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, z):

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        _, h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    _, h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nn.functional.relu(h)
        h = self.conv_out(h)

        return h
