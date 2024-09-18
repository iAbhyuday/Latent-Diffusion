import torch as pt
from torch import nn
from latent_diffusion.utils.init import init_weights


class ResBlock(nn.Module):
    """ ResBlock v1 Module """
    def __init__(self, in_channels, out_channels, stride=1, mode="Identity"):
        """
        
        """
        super(ResBlock, self).__init__()

        self.mode = mode

        if self.mode == "Identity":
            self.tr_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            stride = 2
            self.tr_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False
            )
        self.bn_tr = nn.BatchNorm2d(out_channels)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3,
                padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.layers.apply(init_weights)
        if hasattr(self, "tr_conv"):
            init_weights(self.tr_conv)

    def forward(self, input: pt.Tensor):
        """  """
        res = self.layers(input)
        input = self.bn_tr(self.tr_conv(input))
        return nn.functional.relu(input + res)
