import torch as pt
from torch import nn
from torch.nn.functional import softmax


class AttnBlock(nn.Module):
    """
    Attention Block

    Parameters:
        in_channels (`int`):Number fo input channels
    """
    def __init__(self, in_channels):
        super(AttnBlock, self).__init__()
        self.in_channels = in_channels
        self.qkv = nn.Conv2d(in_channels, 3*in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1
        )

    def forward(self, x: pt.Tensor):
        """
        The [`AttnBlock`] forward method.
        """
        b, c, h, w = x.shape
        # pre-norm
        h_ = self.norm(x)
        q, k, v = self.qkv(h_).reshape(b, h*w, -1).chunk(3, -1)
        scale = c**-0.5
        attn_score = softmax((q@k.permute((0, 2, 1)))*scale, dim=-1)
        attn = (attn_score@v).reshape(b, c, h, w)
        attn = self.proj_out(attn)
        return attn_score, x + attn
