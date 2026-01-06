"""
Training script for VAE (KL) on COCO-17 or CIFAR-10 dataset.
Uses PyTorch Lightning with WandB logging.

Usage:
    python train.py --config configs/coco17-kl.yaml
    python train.py --config configs/cifar10-kl-6gb.yaml
"""
import os
import argparse
import yaml
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from latent_diffusion.models import VAE
from latent_diffusion.data.dataset import build_datamodule


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VAE on COCO-17 or CIFAR-10 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/coco17-kl.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (use TensorBoard instead)",
    )
    return parser.parse_args()


# EMACallback definition locally for now, could be moved to modules/callbacks.py
from pytorch_lightning import Callback
class EMACallback(Callback):
    def __init__(self, decay=0.9999, update_every=1):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.ema_state = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.update_every != 0:
            return
        
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if "encoder" in name or "decoder" in name:
                    if not param.requires_grad:
                        continue
                    if name not in self.ema_state:
                        self.ema_state[name] = param.detach().clone()
                    else:
                        self.ema_state[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def on_validation_start(self, trainer, pl_module):
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.ema_state:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.ema_state[name])

    def on_validation_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class ImageReconstructionCallback(pl.callbacks.Callback):
    def __init__(self, num_images=8):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx != 0: return
        
        images, _ = batch
        images = images[:self.num_images]
        images = images.to(pl_module.device)
        with torch.no_grad():
            reconstructions = pl_module(images)
            if isinstance(reconstructions, tuple):
                reconstructions = reconstructions[0]

        # Denormalize to [0,1] for visualization
        # Assuming model works in [-1, 1]
        def denorm(x):
            return ((x + 1.0) / 2).clamp(0, 1)

        images = denorm(images.cpu())
        reconstructions = denorm(reconstructions.cpu())

        # Log as W&B images
        wandb_images = []
        for orig, recon in zip(images, reconstructions):
            combined = torch.cat([orig, recon], dim=2) 
            wandb_images.append(wandb.Image(combined))
        
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
             trainer.logger.experiment.log({
                "reconstructions": wandb_images
            })
        # Also support TensorBoard
        elif trainer.logger is not None and isinstance(trainer.logger, TensorBoardLogger):
            grid = torch.cat([images, reconstructions], dim=0)
            trainer.logger.experiment.add_images(
                "Validation/Reconstruction", grid, trainer.global_step
            )


def main():
    args = parse_args()
    print(f"Loading config from: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    trainer_cfg = cfg["trainer"]

    # Dynamic DataModule
    data_module = build_datamodule(cfg)

    # Setup logger
    if args.no_wandb:
        logger = TensorBoardLogger(
            save_dir=trainer_cfg.get("tensorboard_log_dir", "./logs"),
            name=trainer_cfg["name"],
        )
    else:
        logger = WandbLogger(
            project=f"{trainer_cfg['name']}-LPIPS",
            name=trainer_cfg["name"],
            log_model=False
        )
    
    trainer = pl.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        logger=logger,
        accelerator=trainer_cfg.get("device", "auto"),
        log_every_n_steps=10,
        precision="16-mixed",
        callbacks=[
            EMACallback(),
            ImageReconstructionCallback(num_images=16),
            ModelCheckpoint(
                dirpath=trainer_cfg.get("checkpoint_dir", "./models"),
                monitor="val_loss" if "cifar" in trainer_cfg["name"].lower() else "fid", # CIDAR has no FID metric in loop usually
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-kl",
                save_last=True,
            )
        ],
    )
    
    # Load or create model
    ckpt_path = args.resume or (trainer_cfg.get("load_ckpt") if trainer_cfg.get("load_ckpt") else None)
    if ckpt_path:
        model = VAE.load_from_checkpoint(ckpt_path, config=cfg, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        model = VAE(cfg)
    
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()