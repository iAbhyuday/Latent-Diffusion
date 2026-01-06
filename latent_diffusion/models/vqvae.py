import torch
from torch import nn
import pytorch_lightning as pl
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import build_quantizer
from latent_diffusion.modules import PerceptualLoss


class VQVAE(pl.LightningModule):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) implementation.
    
    This model encodes images to a discrete latent space using vector quantization,
    enabling high-quality image reconstruction and generation.
    
    Args:
        config: Configuration dictionary containing encoder, decoder, quantizer,
                and training parameters.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.encoder = Encoder(**config["encoder"])
        self.vq = build_quantizer(config["quantizer"])
        # Get codebook_size from config or from the quantizer itself
        self.codebook_size = config["quantizer"]["params"].get(
            "codebook_size", getattr(self.vq, "codebook_size", 512)
        )
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
        self.quantizer_type = config["quantizer"]["type"]
        self.perceptual_loss = PerceptualLoss(**config["perceptual_loss"])

    def forward(self, input_image):
        z = self.encoder(input_image)
        z = self.pre_quant(z)
        code, commitment_loss, codebook_loss, encoding = self.vq(z)
        code = self.post_quant(code)
        x_ = self.decoder(code)

        return x_, code, commitment_loss, codebook_loss, encoding

    def training_step(self, batch, batch_idx):
        input_image, _ = batch
        x_, _, commitment_loss, codebook_loss, perplexity = self(input_image)
        recon_loss = torch.abs(input_image.contiguous() - x_.contiguous()).mean()
        ploss = self.perceptual_loss(x_, input_image)
        if not codebook_loss:
            total_loss = recon_loss + ploss + commitment_loss
        else:
            total_loss = recon_loss + ploss + commitment_loss + codebook_loss
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=True, on_epoch=False)
        self.log("ploss", ploss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("commit_loss", commitment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if codebook_loss:
            self.log("codebook_loss", codebook_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ppl", perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recloss", recon_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_image, _ = batch
        x_, _, commitment_loss, codebook_loss, _ = self(input_image)
        recon_loss = torch.abs(input_image.contiguous() - x_.contiguous()).mean()
        ploss = self.perceptual_loss(x_, input_image)

        if codebook_loss:
            total_loss = recon_loss + ploss + commitment_loss + codebook_loss
        else:
            total_loss = recon_loss + ploss + commitment_loss
        self.log("val_ploss", ploss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_commit_loss", commitment_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss,  on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", total_loss,  on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        """Configure gradient clipping for training stability."""
        clip_val = self.config["trainer"].get("gradient_clip_val", 1.0)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm"
        )

    def configure_optimizers(self):
        weight_decay = self.config["trainer"].get("weight_decay", 1e-2)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.config["trainer"]["lr"]), 
            weight_decay=weight_decay
        )
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        if self.config["trainer"].get("warmup", False):
            warmup_steps = float(self.config["trainer"].get("warmup_steps", 0.1))
            self.warmup_steps = (steps_per_epoch * self.trainer.max_epochs ) / self.trainer.accumulate_grad_batches
            self.warmup_steps = int(warmup_steps * self.warmup_steps) 
            print(f"warmup_steps : {self.warmup_steps}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda step: min((step + 1) / self.warmup_steps, 1.0)
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(self.trainer.max_epochs * steps_per_epoch) // self.trainer.accumulate_grad_batches - self.warmup_steps,
                eta_min=float(self.config["trainer"].get("min_lr", 1e-7)),
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[self.warmup_steps]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                },}
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(self.trainer.max_epochs * steps_per_epoch) // self.trainer.accumulate_grad_batches,
                eta_min=float(self.config["trainer"].get("min_lr", 1e-7)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },}

