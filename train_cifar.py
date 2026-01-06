"""
Training script for VQVAE on CIFAR-10 dataset.
Uses PyTorch Lightning with TensorBoard logging.

Usage:
    python train_cifar.py --config configs/cifar10.yaml
"""
import argparse
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.datasets import CIFAR10
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from latent_diffusion.models import VQVAE


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VQVAE on CIFAR-10 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/cifar10.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


class CIFARDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR-10 dataset."""
    
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        data_dir: str = "./data",
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        # CIFAR-10 is 32x32
        self.transforms = Compose([
            Resize(32),
            ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_data = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.transforms,
            download=True,
        )
        self.val_data = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.transforms,
            download=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageReconstructionCallback(pl.callbacks.Callback):
    """Callback to log image reconstructions during validation."""
    
    def __init__(self, num_images: int = 8):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx != 0:
            return  # Only log first batch
            
        images, _ = batch
        images = images[:self.num_images].to(pl_module.device)
        
        with torch.no_grad():
            reconstructions = pl_module(images)
            if isinstance(reconstructions, tuple):
                reconstructions = reconstructions[0]
        
        # Denormalize for visualization
        def denorm(x):
            return x.clamp(0, 1)
        
        img_grid = torch.cat([denorm(images.cpu()), denorm(reconstructions.cpu())], dim=0)
        trainer.logger.experiment.add_images(
            "Validation/Reconstruction",
            img_grid,
            global_step=trainer.global_step,
        )


def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    trainer_cfg = cfg["trainer"]
    
    # Setup data module
    data_module = CIFARDataModule(
        train_batch_size=trainer_cfg["train_batch_size"],
        val_batch_size=trainer_cfg["val_batch_size"],
        data_dir=trainer_cfg.get("data_dir", "./data"),
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=trainer_cfg.get("tensorboard_log_dir", "./logs"),
        name=trainer_cfg["name"],
    )
    
    # Setup callbacks
    callbacks = [
        ImageReconstructionCallback(num_images=8),
        ModelCheckpoint(
            dirpath=trainer_cfg.get("checkpoint_dir", "./models"),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-cifar10",
            save_last=True,
        ),
    ]
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 100),
        logger=logger,
        accelerator=trainer_cfg.get("device", "auto"),
        log_every_n_steps=10,
        callbacks=callbacks,
    )
    
    # Load or create model
    ckpt_path = args.resume or (trainer_cfg.get("load_ckpt") if trainer_cfg.get("load_ckpt") else None)
    if ckpt_path:
        model = VQVAE.load_from_checkpoint(ckpt_path, config=cfg, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        model = VQVAE(cfg)
    
    # Train
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
