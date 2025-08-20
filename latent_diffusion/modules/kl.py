import torch
from torch import nn

class KLBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, beta=1.0, reduce_mean=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            beta (float): Scaling factor for KL divergence (for beta-VAE).
            reduce_mean (bool): If True, return mean KL loss over batch; else return per-sample KLs.
        """
        super(KLBottleNeck, self).__init__()
        self.mean = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.logvar = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.beta = beta
        self.reduce_mean = reduce_mean

    def forward(self, x):
        mean = self.mean(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # Compute KL divergence per sample
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])

        if self.reduce_mean:
            kl_loss = self.beta * kl.mean()
        else:
            kl_loss = self.beta * kl

        return z, kl_loss, mean.mean(), std.mean()
