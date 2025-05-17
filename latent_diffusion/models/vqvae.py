import torch as pt
from torch import nn
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import build_quantizer
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.modules import KLBottleNeck

class VQVAE(nn.Module):
    def __init__(self, config: dict):
        super(VQVAE, self).__init__()
        self.config = config
        
        self.pre_quant = nn.Conv2d(
                config["encoder"]["out_channels"],
                config["quantizer"]["params"]["embed_dim"],
                kernel_size=(1, 1)
                )
        self.encoder = Encoder(**config["encoder"])
        if config["quantizer"]["type"] == "kl":
            self.vq = KLBottleNeck(
                in_channels=config["encoder"]["out_channels"],
                out_channels=config["quantizer"]["params"]["embed_dim"]
            )
        else:
            self.vq = build_quantizer(config["quantizer"])
            self.post_quant = nn.Conv2d(
                config["quantizer"]["params"]["embed_dim"],
                config["decoder"]["in_channels"],
                kernel_size=(1, 1)
                 )

        self.decoder = Decoder(**config["decoder"])

    def forward(self, input_image):
        z = self.encoder(input_image)
        z = self.pre_quant(z)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        code = self.post_quant(code)
        x_ = self.decoder(code)
        recon_loss = nn.functional.mse_loss(x_, input_image)
        return x_, code, commitment_loss, codebook_loss, recon_loss, encoding
