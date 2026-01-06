import torch
from torch import nn

class KLBottleNeck(nn.Module):
    """
    KL Divergence Bottleneck for Variational Autoencoders.
    
    Projects input to mean and log-variance, then samples using the 
    reparameterization trick. Returns the sampled latent, KL divergence,
    and summary statistics.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (latent dimension).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(KLBottleNeck, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1)

    def forward(self, x):
        mean, logvar = self.conv(x).chunk(2, 1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = self.sample(mean, std)
        # Compute KL divergence per sample
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return z, kl, mean.mean(), std.mean()
    
    def sample(self, mean, std):
        x = mean + std * torch.randn(mean.shape, device=mean.device)
        return x
