"""
Training script for VQVAE on COCO-17 dataset.
Uses PyTorch Lightning for clean, maintainable training.
"""
import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, RandomHorizontalFlip, Lambda
from torchvision.datasets import CocoCaptions
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from latent_diffusion.models import VQVAE


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We use a custom collate_fn because merging captions with padding 
    is not supported in the default collate function.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (channel, height, width).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, H, W).
        labels: placeholder (0) since we don't use captions for VQVAE.
    """
    images, _ = zip(*data)
    images = torch.stack(images, 0)
    return images, 0


class CocoDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for COCO-17 dataset."""
    
    def __init__(
        self,
        train_root: str,
        train_ann: str,
        val_root: str,
        val_ann: str,
        train_batch_size: int,
        val_batch_size: int,
        resolution: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_root = train_root
        self.train_ann = train_ann
        self.val_root = val_root
        self.val_ann = val_ann
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.resolution = resolution
        self.num_workers = num_workers
        
        # Transforms
        self.train_transforms = Compose([
            Resize((resolution, resolution)),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ])
        self.val_transforms = Compose([
            Resize((resolution, resolution)),
            ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_data = CocoCaptions(
            root=self.train_root,
            annFile=self.train_ann,
            transform=self.train_transforms,
        )
        self.val_data = CocoCaptions(
            root=self.val_root,
            annFile=self.val_ann,
            transform=self.val_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True,
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
            return  # Only log for first batch
            
        images, _ = batch
        images = images[:self.num_images].to(pl_module.device)
        
        with torch.no_grad():
            reconstructions = pl_module(images)
            if isinstance(reconstructions, tuple):
                reconstructions = reconstructions[0]

        # Denormalize for visualization (assuming [0,1] input)
        def denorm(x):
            return x.clamp(0, 1)

        images = denorm(images.cpu())
        reconstructions = denorm(reconstructions.cpu())

        # Log to TensorBoard
        img_grid = torch.cat([images, reconstructions], dim=0)
        trainer.logger.experiment.add_images(
            "Validation/Reconstruction",
            img_grid,
            global_step=trainer.global_step,
        )


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train VQVAE on COCO-17 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/coco17-vq.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    trainer_cfg = cfg["trainer"]
    data_cfg = cfg.get("data", {})
    resolution = cfg["encoder"]["resolution"]

    # Setup data module with paths from config
    data_module = CocoDataModule(
        train_root=data_cfg.get("train_root", "./data/coco17/train2017"),
        train_ann=data_cfg.get("train_ann", "./data/coco17/annotations/captions_train2017.json"),
        val_root=data_cfg.get("val_root", "./data/coco17/val2017"),
        val_ann=data_cfg.get("val_ann", "./data/coco17/annotations/captions_val2017.json"),
        train_batch_size=trainer_cfg["train_batch_size"],
        val_batch_size=trainer_cfg["val_batch_size"],
        resolution=resolution,
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
            filename="best-{epoch:02d}-vq",
            save_last=True,
        ),
    ]

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 25),
        logger=logger,
        accelerator=trainer_cfg.get("device", "auto"),
        log_every_n_steps=10,
        precision="16-mixed",
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
