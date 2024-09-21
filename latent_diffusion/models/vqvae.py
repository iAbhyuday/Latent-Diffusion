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
        self.codebook_size = config["codebook_size"]
        self.embed_dim = config["embed_dim"]
        self.use_ema = config["quantizer"]["use_ema"]
        self.encoder = Encoder(**config["encoder"],)
        self.vq = Quantizer(**config["quantizer"])
        self.decoder = Decoder(**config["decoder"])
        self.percept_loss = PerceptualLoss(**config["percept_loss"])

    
    def forward(self, input_image):
        z = self.encoder(input_image)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        x_ = self.decoder(code)
        recon_loss = nn.functional.mse_loss(x_, input_image)
        percept_loss = self.percept_loss(x_, input_image)

        return x_, code, commitment_loss, codebook_loss, recon_loss, percept_loss, encoding
    