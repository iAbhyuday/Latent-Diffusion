import torch
from torch import nn

class KLBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            beta (float): Scaling factor for KL divergence (for beta-VAE).
            reduce_mean (bool): If True, return mean KL loss over batch; else return per-sample KLs.
        """
        super(KLBottleNeck, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2*out_channels, kernel_size=1)

    def forward(self, x):
        mean, logvar = self.conv(x).chunk(2, 1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # Compute KL divergence per sample
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return z, kl, mean.mean(), std.mean()
