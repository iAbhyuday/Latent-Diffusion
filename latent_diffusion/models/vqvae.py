import torch as pt
from torch import nn
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import Quantizer
from latent_diffusion.modules import PerceptualLoss


class VQVAE(nn.Module):
    def __init__(self, config: dict):
        super(VQVAE, self).__init__()
        self.config = config
        # if config["encoder"]["out_channels"] != config["decoder"]["in_channels"]:
        #     self.pre_quant = nn.Conv2d(
        #         config["encoder"]["out_channels"],
        #         config["decoder"]["in_channels"],
        #         kernel_size=1
        #          )
        self.encoder = Encoder(**config["encoder"])
        self.vq = Quantizer(**config["quantizer"])
        self.decoder = Decoder(**config["decoder"])

    def forward(self, input_image):
        z = self.encoder(input_image)
        if hasattr(self, "pre_quant"):
            z = self.pre_quant(z)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        x_ = self.decoder(code)
        recon_loss = nn.functional.mse_loss(x_, input_image)
        return x_, code, commitment_loss, codebook_loss, recon_loss, encoding
