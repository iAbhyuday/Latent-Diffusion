import torch as pt
from torch import nn
import pytorch_lightning as pl
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import build_quantizer
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.modules import KLBottleNeck
from latent_diffusion.utils.metrics import measure_perplexity
from torchmetrics.regression import MinkowskiDistance


class VQVAE(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.encoder = Encoder(**config["encoder"])
        if config["quantizer"]["type"] == "kl":
            self.vq = KLBottleNeck(
                in_channels=config["encoder"]["out_channels"],
                out_channels=config["quantizer"]["params"]["embed_dim"]
            )
            self.beta = float(config["quantizer"]["params"]["beta"])
        else:
            self.vq = build_quantizer(config["quantizer"])
            self.codebook_size = config["quantizer"]["params"]["codebook_size"]
        self.pre_quant = nn.Conv2d(
                config["encoder"]["out_channels"],
                config["quantizer"]["params"]["embed_dim"],
                kernel_size=(1, 1)
          )
        self.post_quant = nn.Conv2d(
                config["quantizer"]["params"]["embed_dim"],
                config["decoder"]["in_channels"],
                kernel_size=(1, 1)
        )
        self.decoder = Decoder(**config["decoder"])
        self.percept_loss = PerceptualLoss(**config["perceptual_loss"])
        self.quantizer_type = config["quantizer"]["type"]

        if config["trainer"]["load_ckpt"]:
            checkpoint = pt.load(config["trainer"]["load_ckpt"], weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])


    def forward(self, input_image):
        z = self.encoder(input_image)
        z = self.pre_quant(z)
        encoding = 0
        if self.config["quantizer"]["type"] == "kl":
            code, kl_loss, mean, std = self.vq(z)
            code = self.post_quant(code)
            self.log("kl mean", mean, on_step=True, prog_bar=True, logger=True)
            self.log("kl std", std, on_step=True, prog_bar=True, logger=True)

            commitment_loss = 0
            codebook_loss = 0
        else:
            z = self.pre_quant(z)
            code, commitment_loss, codebook_loss, encoding = self.vq(z)
            code = self.post_quant(code)
            kl_loss = 0
        x_ = self.decoder(code)
        return x_, code, commitment_loss, codebook_loss, kl_loss, encoding

    def training_step(self, batch, batch_idx):
        input_image, _ = batch
        x_, _, commitment_loss, codebook_loss, kl_loss, encoding = self(input_image)
        # recon_loss = nn.functional.mse_loss(x_, input_image, reduction="sum") / input_image.numel()
        recon_loss = pt.abs(input_image.contiguous() - x_.contiguous()).mean()
        ploss = self.percept_loss(x_, input_image)
        total_loss = recon_loss + ploss
        if self.config["quantizer"]["type"] == "kl":
            kl_loss = pt.sum(kl_loss) / kl_loss.shape[0]
            total_loss = total_loss + self.beta * kl_loss
            self.log("kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            total_loss += commitment_loss
            if codebook_loss:
                total_loss += codebook_loss
                self.log("codebook_loss", codebook_loss, on_epoch=True, prog_bar=True, logger=True)

            # ppl, _ = measure_perplexity(encoding, self.codebook_size)
            # self.log("perplexity", ppl, on_epoch=True, logger=True)
            self.log("commitment_loss", commitment_loss, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss", recon_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_image, _ = batch
        x_, code, commitment_loss, codebook_loss, kl_loss, _ = self(input_image)
        recon_loss = pt.abs(input_image.contiguous() - x_.contiguous()).mean()
        total_loss = recon_loss
        if self.config["quantizer"]["type"] == "kl":
            kl_loss = pt.sum(kl_loss) / kl_loss.shape[0]
            total_loss = total_loss + self.beta * kl_loss
            self.log("val_kl_loss", kl_loss,  on_epoch=True, prog_bar=True, logger=True)

        else:
            total_loss += commitment_loss
            if codebook_loss:
                total_loss += codebook_loss
                self.log("val_codebook_loss", codebook_loss, on_epoch=True, prog_bar=True, logger=True)

            self.log("val_commitment_loss", commitment_loss,  on_epoch=True, prog_bar=True, logger=True)

        self.log("val_loss", total_loss,  on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss,  on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = pt.optim.Adam(self.parameters(), lr=float(self.config["trainer"].get("lr", 1e-4)))
        return optimizer
