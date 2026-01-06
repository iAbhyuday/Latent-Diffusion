import torch as pt
from torch import nn
import lpips
import torch.nn.functional as F
import pytorch_lightning as pl
from latent_diffusion.modules import Encoder
from latent_diffusion.modules import Decoder
from latent_diffusion.modules import KLBottleNeck
from latent_diffusion.modules import NLayerDiscriminator
from torchmetrics.image.fid import FrechetInceptionDistance
from latent_diffusion.losses.loss import *
perceptual_loss = lpips.LPIPS(net='vgg').to(pt.device("cuda" if pt.cuda.is_available() else "cpu"))
# fid = FrechetInceptionDistance(feature=2048).to(pt.device("cuda" if pt.cuda.is_available() else "cpu"))

class FIDMetric:
    def __init__(self, dim=2048):
        self.fid = FrechetInceptionDistance(feature=dim).to(pt.device("cuda" if pt.cuda.is_available() else "cpu"))
    
    def measure(self, x, y):
        
        # Map from [-1,1] → [0,1]
        x = (x.clamp(-1, 1) + 1) / 2
        y = (y.clamp(-1, 1) + 1) / 2
        # Resize to 299x299 for Inception (if not already)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert to uint8 [0,255]
        x = (x * 255).to(pt.uint8)
        y = (y * 255).to(pt.uint8)
        # Update FID
        self.fid.update(x, real=True)
        self.fid.update(y, real=False)

    def compute(self):
        fid_score = self.fid.compute()
        self.fid.reset()
        return fid_score

FID = FIDMetric()
class VAE(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False
        self.disc_start = 0
        self.grad_acc_steps = config["trainer"].get("grad_acc_steps", 2)
        self.disc_weight = config["trainer"].get("disc_weight", 0.5)
        self.encoder = Encoder(**config["encoder"])
        self.pre_quant = nn.Conv2d(
                config["encoder"]["out_channels"],
                2 * config["quantizer"]["params"]["in_channels"],
                kernel_size=(1, 1)
          )
        self.kl = KLBottleNeck(
                2 * config["quantizer"]["params"]["in_channels"],
                out_channels=config["quantizer"]["params"]["embed_dim"]
        )        
        self.post_quant = nn.Conv2d(
                config["quantizer"]["params"]["embed_dim"],
                config["decoder"]["in_channels"],
                kernel_size=(1, 1)
        )
        self.decoder = Decoder(**config["decoder"])

        self.beta = float(config["quantizer"]["params"]["beta"])
        self.p_weight = self.config["perceptual_loss"]["scale"]
        self.discriminator = NLayerDiscriminator(**config["disc_config"])
        
        self.logvar = nn.Parameter(pt.ones(size=()) * 0.)

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
        kl_loss = (pt.sum(kl) / kl.shape[0])
        return x_, z, kl_loss, mean, std

    def training_step(self, batch, batch_idx):
        input_image, _ = batch
        g_opt, d_opt = self.optimizers()
        g_sched, d_sched = self.lr_schedulers()
        x_, _, kl_loss, mean, std = self(input_image)
        recon_loss = pt.abs(input_image.contiguous() - x_.contiguous()).mean(dim=[1,2,3])
        ploss = self.p_weight * perceptual_loss(input_image.contiguous(), x_.contiguous()).squeeze()
        nll_loss = recon_loss + ploss
        nll_loss = nll_loss / pt.exp(self.logvar) + self.logvar
        w_nll = nll_loss.mean()
        nll_loss = pt.sum(nll_loss) / nll_loss.shape[0] 
        
        total_loss = w_nll + self.beta * kl_loss
        # Vae Update
        if self.trainer.global_step > self.disc_start:
            dis_fake = self.discriminator(x_.contiguous())
            g_loss = -pt.mean(dis_fake)
            #d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, self.get_last_layer())
            total_loss = total_loss + 0.2 * g_loss
            self.log("g_loss", g_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)   
            #self.log("d_w", d_weight, on_step=True, on_epoch=False, logger=True)     
        self.manual_backward(total_loss)
        if (batch_idx + 1) % self.grad_acc_steps == 0:
            g_opt.step()
            #g_sched.step()
            g_opt.zero_grad()

        # Discriminator Update
        if self.trainer.global_step > self.disc_start:
            dis_real = self.discriminator(input_image.contiguous().detach())
            dis_fake = self.discriminator(x_.contiguous().detach())
            d_loss = gan_loss_hinge_dis(dis_fake, dis_real)
            self.manual_backward(d_loss)
            if (batch_idx + 1) % self.grad_acc_steps ==0:
                # pt.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                d_opt.step()
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
        
        recon_loss = pt.abs(input_image.contiguous() - x_.contiguous()).mean(dim=[1,2,3])
        ploss = self.p_weight * perceptual_loss(input_image.contiguous(), x_.contiguous()).squeeze()
        nll_loss = recon_loss + self.p_weight * ploss
        nll_loss = nll_loss / pt.exp(self.logvar) + self.logvar
        nll_loss = pt.sum(nll_loss) / nll_loss.shape[0] 
        total_loss = nll_loss + self.beta * kl_loss
        FID.measure(input_image, x_)
        self.log("val_recon_loss", nll_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    def on_validation_epoch_end(self):
        fid_score = FID.compute()
        self.log("fid", fid_score, on_epoch=True, prog_bar=True, logger=True)
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = pt.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = pt.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = pt.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = pt.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = pt.norm(nll_grads) / (pt.norm(g_grads) + 1e-4)
        self.log("nll_grad", pt.norm(nll_grads).detach(), on_step=True, on_epoch=False, logger=True)
        self.log("g_grad", pt.norm(g_grads).detach(), on_step=True, on_epoch=False, logger=True)
        d_weight = pt.clamp(d_weight, 0.0, 50).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight
    def configure_optimizers(self):
        g_optimizer = pt.optim.Adam(
            list(self.encoder.parameters())+
            list(self.pre_quant.parameters())+
            list(self.kl.parameters())+
            list(self.post_quant.parameters())+
            list(self.decoder.parameters()),
            lr=6e-6,
            weight_decay=1e-2)
        d_optimizer = pt.optim.Adam(self.discriminator.parameters(), lr=2e-6, weight_decay=1e-2)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        max_steps = (self.trainer.max_epochs * steps_per_epoch) // self.grad_acc_steps
        d_sched = pt.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer,
            T_max=max_steps,
            eta_min=self.config["trainer"].get("min_lr", 4.5e-6),
            )
        if False:
            self.warmup_steps = (steps_per_epoch * self.trainer.max_epochs ) // self.grad_acc_steps
            self.warmup_steps = int(self.config["trainer"]["warmup_steps"] * self.warmup_steps) 
            print(f"warmup_steps : {self.warmup_steps}")

            warmup_scheduler = pt.optim.lr_scheduler.LambdaLR(
                g_optimizer, 
                lr_lambda=lambda step: min(max((step + 1) / self.warmup_steps, 1e-1), 1.0)
            )
            total_cosine_steps = max_steps - self.warmup_steps
            main_scheduler =  pt.optim.lr_scheduler.CosineAnnealingLR(
                g_optimizer,
                T_max=total_cosine_steps,
                eta_min=self.config["trainer"].get("min_lr", 1e-7),
            )

            scheduler = pt.optim.lr_scheduler.SequentialLR(
                g_optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[self.warmup_steps]
            )
            return (
                [g_optimizer, d_optimizer], 
                [
                    {"scheduler": scheduler, "interval": "step"}, 
                    {"scheduler": d_sched, "interval": "step"}
                ]
            )
        else:
            scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(
                g_optimizer,
                T_max=max_steps,
                eta_min=self.config["trainer"].get("min_lr", 1e-6),
            )
            return (
                [g_optimizer, d_optimizer], 
                [
                    {"scheduler": scheduler, "interval": "step"}, 
                    {"scheduler": d_sched, "interval": "step"}
                ]
            )
