import torch as pt
from torch import nn
from latent_diffusion.utils import init_weights


class ResBlock(nn.Module):
    """
    ResBlock v1 Module

    Parameters:
        in_channels (`int`): Number of input channels
        out_channels (`int`, *optional*, defaults to in_channels):
            Number of output channels
        mode (`str`, *optional*, defaults to "Identity"):
            type of ResBlock. Choose between "Down" and "Identity"
        conv_shortcut (`bool`, *optional*, defaults to True):
            Use 3x3 conv to transform the residual link
    """
    def __init__(
            self,
            in_channels,
            out_channels=None,
            mode="Identity",
            conv_shortcut=True
    ):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.conv_shortcut = conv_shortcut
        stride = 1

        if self.mode == "Identity":
            if conv_shortcut:
                self.tr_conv = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=3, padding=1, bias=False)
            else:
                self.tr_conv = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=1, bias=False)
        else:
            stride = 2
            if self.conv_shortcut:
                self.tr_conv = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1, stride=stride, bias=False
                )
            else:
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

    def forward(self, x: pt.Tensor):
        """
        The [`ResBlock`] forward method.

        Args:
            x (`torch.Tensor`): Input Tensor of shape [N, C, H, W]
        """
        h = self.layers(x)
        x = self.bn_tr(self.tr_conv(x))
        return nn.functional.relu(x + h)
