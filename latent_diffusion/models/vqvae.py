import torch as pt
from torch import nn
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import Quantizer
from latent_diffusion.modules import PerceptualLoss


class VQVAE(nn.Module):
    def __init__(self, embed_dim=512, codebook_size = 256, use_ema = True):
        super(VQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.use_ema = use_ema
        self.encoder = Encoder(resolution=32, num_resblock=2, ch_mult=[1, 4, 8],attn_resolution=[4, 8],ch_factor=32, out_channels=4)
        self.pre_quant = nn.Conv2d(4, embed_dim, kernel_size=3, padding=1)
        self.vq = Quantizer(self.codebook_size, self.embed_dim, use_ema=self.use_ema)
        self.decoder = Decoder(in_channels=embed_dim, resolution=4, out_channels=3, ch_factor=1, ch_mult=[1, 4, 8], num_resblock=2, attn_resolution=[8, 16])
        self.percept_loss = PerceptualLoss(layers=[1, 6, 11, 20, 29], normalized=False)

    
    def forward(self, input_image):
        z = self.encoder(input_image)
        z = self.pre_quant(z)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        x_ = self.decoder(code)
        recon_loss = nn.functional.mse_loss(x_, input_image)
        percept_loss = self.percept_loss(x_, input_image)

        return x_, code, commitment_loss, codebook_loss, recon_loss, percept_loss, encoding