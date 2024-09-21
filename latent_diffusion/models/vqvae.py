import torch as pt
from torch import nn
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import Quantizer
from latent_diffusion.modules import PerceptualLoss


class VQVAE(nn.Module):
    def __init__(self, config: dict):
        super(VQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.use_ema = use_ema
        self.encoder = Encoder(3)
        self.pre_quant = nn.LazyConv2d(embed_dim, kernel_size=3, padding=1)
        self.vq = Quantizer(self.codebook_size, self.embed_dim, use_ema=self.use_ema)
        self.decoder = Decoder(2, 3)
        self.percept_loss = PerceptualLoss(layers=[1, 6, 11, 20, 29], normalized=False)

    
    def forward(self, input_image):
        z = self.encoder(input_image)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        x_ = self.decoder(code)
        recon_loss = nn.functional.mse_loss(x_, input_image)
        percept_loss = self.percept_loss(x_, input_image)

        return x_, code, commitment_loss, codebook_loss, recon_loss, percept_loss, encoding