import torch
from torch import nn
import lpips
import torch.nn.functional as F
import pytorch_lightning as pl
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import KLBottleNeck
from latent_diffusion.modules import NLayerDiscriminator
from torchmetrics.image.fid import FrechetInceptionDistance
from latent_diffusion.losses.loss import gan_loss_hinge_dis


class FIDMetric:
    """
    Wrapper for FID (Fréchet Inception Distance) metric computation.
    
    Handles proper preprocessing of images for FID calculation.
    """
    def __init__(self, dim: int = 2048):
        self.fid = FrechetInceptionDistance(feature=dim).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
    def measure(self, x, y):
        
        # Map from [-1,1] → [0,1]
        x = (x.clamp(-1, 1) + 1) / 2
        y = (y.clamp(-1, 1) + 1) / 2
        # Resize to 299x299 for Inception (if not already)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert to uint8 [0,255]
        x = (x * 255).to(torch.uint8)
        y = (y * 255).to(torch.uint8)
        # Update FID
        self.fid.update(x, real=True)
        self.fid.update(y, real=False)

    def compute(self):
        fid_score = self.fid.compute()
        self.fid.reset()
        return fid_score


class VAE(pl.LightningModule):
    """
    Variational Autoencoder with KL regularization and GAN-based discriminator.
    
    This model uses a KL bottleneck for continuous latent space regularization
    and includes a PatchGAN discriminator for adversarial training.
    
    Args:
        config: Configuration dictionary containing encoder, decoder, quantizer,
                discriminator, and training parameters.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False
        
        # Training configuration with sensible defaults
        self.disc_start = config["trainer"].get("disc_start", 10000)  # Warm-up before discriminator
        self.grad_acc_steps = config["trainer"].get("grad_acc_steps", 2)
        self.disc_weight = config["trainer"].get("disc_weight", 0.5)
        self.adversarial_weight = config["trainer"].get("adversarial_weight", 0.1)
        self.use_adaptive_weight = config["trainer"].get("use_adaptive_weight", False)
        self.gradient_clip_val = config["trainer"].get("gradient_clip_val", 1.0)
        self.weight_decay = config["trainer"].get("weight_decay", 1e-2)
        self.d_weight_max = config["trainer"].get("d_weight_max", 50.0)
        
        # Encoder
        self.encoder = Encoder(**config["encoder"])
        self.pre_quant = nn.Conv2d(
                config["encoder"]["out_channels"],
                2 * config["quantizer"]["params"]["in_channels"],
                kernel_size=(1, 1)
          )
        
        # KL Bottleneck
        self.kl = KLBottleNeck(
                2 * config["quantizer"]["params"]["in_channels"],
                out_channels=config["quantizer"]["params"]["embed_dim"]
        )
        
        # Decoder
        self.post_quant = nn.Conv2d(
                config["quantizer"]["params"]["embed_dim"],
                config["decoder"]["in_channels"],
                kernel_size=(1, 1)
        )
        self.decoder = Decoder(**config["decoder"])

        # Loss weights
        self.beta = float(config["quantizer"]["params"]["beta"])
        self.p_weight = self.config["perceptual_loss"]["scale"]
        
        # Discriminator
        self.discriminator = NLayerDiscriminator(**config["disc_config"])
        
        # Learnable log-variance for uncertainty weighting
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.)

        # Perceptual loss (will be moved to correct device in on_fit_start)
        self.perceptual_loss_fn = lpips.LPIPS(net='vgg')
        for param in self.perceptual_loss_fn.parameters():
            param.requires_grad = False  # Freeze LPIPS weights
        
        # FID metric (initialized lazily to use correct device)
        self._fid_metric = None

    @property
    def fid_metric(self):
        """Lazy initialization of FID metric to ensure correct device."""
        if self._fid_metric is None:
            self._fid_metric = FIDMetric()
            # Move to same device as model
            if hasattr(self, 'device'):
                self._fid_metric.fid = self._fid_metric.fid.to(self.device)
        return self._fid_metric

    def encode(self, x):
        z = self.encoder(x)
        z = self.pre_quant(z)
        z, kl, mu, sigma = self.kl(z)
        return z, kl, mu, sigma

    def decode(self, z):
        z = self.post_quant(z)
        x_ = self.decoder(z)
        return x_
    
    def forward(self, input_image):
        z, kl, mean, std = self.encode(input_image)
        x_ = self.decode(z)
        kl_loss = (torch.sum(kl) / kl.shape[0])
        return x_, z, kl_loss, mean, std

    def training_step(self, batch, batch_idx):
        input_image, _ = batch
        g_opt, d_opt = self.optimizers()
        g_sched, d_sched = self.lr_schedulers()
        x_, _, kl_loss, mean, std = self(input_image)
        recon_loss = torch.abs(input_image.contiguous() - x_.contiguous()).mean(dim=[1,2,3])
        ploss = self.p_weight * self.perceptual_loss_fn(input_image.contiguous(), x_.contiguous()).squeeze()
        nll_loss = recon_loss + ploss
        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar
        w_nll = nll_loss.mean()
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        total_loss = w_nll + self.beta * kl_loss
        
        # Discriminator adversarial loss (only after warm-up)
        if self.trainer.global_step > self.disc_start:
            dis_fake = self.discriminator(x_.contiguous())
            g_loss = -torch.mean(dis_fake)
            
            # Use adaptive or fixed weighting for adversarial loss
            if self.use_adaptive_weight:
                d_weight = self.calculate_adaptive_weight(w_nll, g_loss, self.get_last_layer())
                self.log("d_weight", d_weight, on_step=True, on_epoch=False, logger=True)
            else:
                d_weight = self.adversarial_weight
            
            total_loss = total_loss + d_weight * g_loss
            self.log("g_loss", g_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)     
        self.manual_backward(total_loss)
        if (batch_idx + 1) % self.grad_acc_steps == 0:
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.gradient_clip_val)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.gradient_clip_val)
            g_opt.step()
            g_sched.step()  # Step LR scheduler (required for manual optimization)
            g_opt.zero_grad()

        # Discriminator Update
        if self.trainer.global_step > self.disc_start:
            dis_real = self.discriminator(input_image.contiguous().detach())
            dis_fake = self.discriminator(x_.contiguous().detach())
            d_loss = gan_loss_hinge_dis(dis_fake, dis_real)
            self.manual_backward(d_loss)
            if (batch_idx + 1) % self.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.gradient_clip_val)
                d_opt.step()
                d_sched.step()  # Step discriminator LR scheduler
                d_opt.zero_grad()
                self.log("d_loss", d_loss.detach(), prog_bar=True, on_epoch=True, on_step=True)

        g_lr = g_opt.param_groups[0]["lr"]
        self.log("lr", g_lr, on_step=True, prog_bar=True, on_epoch=False)
        self.log("logvar", self.logvar.detach(), on_step=True, on_epoch=False, logger=True)
        self.log("mu", mean, on_step=True, prog_bar=True, on_epoch=False, logger=True)
        self.log("std", std, on_step=True, prog_bar=True, on_epoch=False, logger=True)
        self.log("kl_loss", self.beta * kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss", w_nll.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_image, _ = batch
        x_, _, kl_loss, _, _= self(input_image)
        
        recon_loss = torch.abs(input_image.contiguous() - x_.contiguous()).mean(dim=[1,2,3])
        ploss = self.p_weight * self.perceptual_loss_fn(input_image.contiguous(), x_.contiguous()).squeeze()
        nll_loss = recon_loss + ploss
        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        total_loss = nll_loss + self.beta * kl_loss
        self.fid_metric.measure(input_image, x_)
        self.log("val_recon_loss", nll_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    def on_validation_epoch_end(self):
        fid_score = self.fid_metric.compute()
        self.log("fid", fid_score, on_epoch=True, prog_bar=True, logger=True)
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        self.log("nll_grad", torch.norm(nll_grads).detach(), on_step=True, on_epoch=False, logger=True)
        self.log("g_grad", torch.norm(g_grads).detach(), on_step=True, on_epoch=False, logger=True)
        d_weight = torch.clamp(d_weight, 0.0, self.d_weight_max).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for generator and discriminator."""
        g_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+
            list(self.pre_quant.parameters())+
            list(self.kl.parameters())+
            list(self.post_quant.parameters())+
            list(self.decoder.parameters())+
            [self.logvar],  # Include learnable log-variance for uncertainty weighting
            lr=self.config["trainer"].get("lr", 6e-6),
            weight_decay=self.weight_decay)
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.config["trainer"].get("disc_lr", 2e-6), 
            weight_decay=self.weight_decay
        )
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        max_steps = (self.trainer.max_epochs * steps_per_epoch) // self.grad_acc_steps
        
        d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer,
            T_max=max_steps,
            eta_min=self.config["trainer"].get("min_lr", 4.5e-6),
        )
        
        g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer,
            T_max=max_steps,
            eta_min=self.config["trainer"].get("min_lr", 1e-6),
        )
        
        return (
            [g_optimizer, d_optimizer], 
            [
                {"scheduler": g_sched, "interval": "step"}, 
                {"scheduler": d_sched, "interval": "step"}
            ]
        )
