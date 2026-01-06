"""
Training script for VAE (KL) on COCO-17 dataset.
Uses PyTorch Lightning with WandB logging.

Usage:
    python train.py --config configs/coco17-kl.yaml
"""
import os
import argparse
import yaml
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Lambda, RandomHorizontalFlip
from torchvision.datasets import CocoCaptions
from latent_diffusion.models import VAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VAE on COCO-17 dataset",
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

args = parse_args()

print(f"Loading config from: {args.config}")
with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]

# Transforms (set up based on resolution from config)
resolution = cfg["encoder"]["resolution"]
t_transforms = Compose([
    Resize((resolution, resolution)),
    RandomHorizontalFlip(0.5),
    ToTensor(),
    Lambda(lambda x: x * 2 - 1)
])
v_transforms = Compose([
    Resize((resolution, resolution)),
    ToTensor(),
    Lambda(lambda x: x * 2 - 1)
])

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (channel, height, width).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """

    images, _ = zip(*data)
    images = torch.stack(images, 0) 
    return images, 0


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, val_batch_size, train_root, train_ann, val_root, val_ann):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_root = train_root
        self.train_ann = train_ann
        self.val_root = val_root
        self.val_ann = val_ann

    def setup(self, stage=None):
        self.train_data = CocoCaptions(
                root=self.train_root,
                annFile=self.train_ann,
                transform=t_transforms,
            )
        self.val_data = CocoCaptions(
                root=self.val_root,
                annFile=self.val_ann,
                transform=v_transforms,
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True,drop_last=True)


# Get data paths from config (with fallback defaults)
data_cfg = cfg.get("data", {})
data_module = CocoDataModule(
    train_batch_size=trainer_cfg["train_batch_size"],
    val_batch_size=trainer_cfg["val_batch_size"],
    train_root=data_cfg.get("train_root", "./data/coco17/train2017"),
    train_ann=data_cfg.get("train_ann", "./data/coco17/annotations/captions_train2017.json"),
    val_root=data_cfg.get("val_root", "./data/coco17/val2017"),
    val_ann=data_cfg.get("val_ann", "./data/coco17/annotations/captions_val2017.json"),
)
#%%
logger = TensorBoardLogger(
    save_dir=trainer_cfg["tensorboard_log_dir"],
    name=trainer_cfg["name"]
)

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
        # Swap to EMA weights for validation
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.ema_state:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.ema_state[name])

    def on_validation_end(self, trainer, pl_module):
        # Restore normal weights
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
        images, _ = batch
        images = images[:self.num_images]
        images = images.to(pl_module.device)
        with torch.no_grad():
            reconstructions = pl_module(images)
            if isinstance(reconstructions, tuple):
                reconstructions = reconstructions[0]

        # Denormalize to [0,1] if model outputs [-1,1]
        def denorm(x):
            return ((x + 1.0) / 2).clamp(0, 1)

        images = denorm(images.cpu())
        reconstructions = denorm(reconstructions.cpu())

        # Log as W&B images (original and reconstruction pairs)
        wandb_images = []
        for orig, recon in zip(images, reconstructions):
            # Combine original + reconstruction in one image
            combined = torch.cat([orig, recon], dim=2)  # concatenate width-wise
            wandb_images.append(wandb.Image(combined))
        
        if trainer.logger is not None:
            trainer.logger.experiment.log({
                "reconstructions": wandb_images
            })


def main():
    """Main training function."""
    print(f"Loading config from: {args.config}")
    
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
                monitor="fid",
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