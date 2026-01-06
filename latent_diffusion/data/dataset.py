"""
Data loading utilities for Latent Diffusion training.
Contains DataModules for COCO-17 and CIFAR-10 datasets.
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, RandomHorizontalFlip, Lambda
from torchvision.datasets import CocoCaptions, CIFAR10


def coco_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
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
        
        # Transforms match train.py: Resize -> RandomFlip -> ToTensor -> [-1, 1]
        self.train_transforms = Compose([
            Resize((resolution, resolution)),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1)
        ])
        
        self.val_transforms = Compose([
            Resize((resolution, resolution)),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1)
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
            collate_fn=coco_collate_fn,
            prefetch_factor=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=coco_collate_fn,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True,
        )


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
        
        # CIFAR-10 is 32x32. Normalize to [-1, 1]
        self.transforms = Compose([
            Resize(32),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1)
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


def build_datamodule(config: dict) -> pl.LightningDataModule:
    """Factory function to create the appropriate DataModule based on config."""
    trainer_cfg = config["trainer"]
    dataset_name = trainer_cfg["name"].lower()
    
    if "cifar" in dataset_name:
        return CIFARDataModule(
            train_batch_size=trainer_cfg["train_batch_size"],
            val_batch_size=trainer_cfg["val_batch_size"],
            data_dir=trainer_cfg.get("data_dir", "./data"),
            num_workers=config.get("data", {}).get("num_workers", 4),
        )
    elif "coco" in dataset_name:
        data_cfg = config.get("data", {})
        return CocoDataModule(
            train_root=data_cfg.get("train_root", "./data/coco17/train2017"),
            train_ann=data_cfg.get("train_ann", "./data/coco17/annotations/captions_train2017.json"),
            val_root=data_cfg.get("val_root", "./data/coco17/val2017"),
            val_ann=data_cfg.get("val_ann", "./data/coco17/annotations/captions_val2017.json"),
            train_batch_size=trainer_cfg["train_batch_size"],
            val_batch_size=trainer_cfg["val_batch_size"],
            resolution=config["encoder"].get("resolution", 128),
            num_workers=data_cfg.get("num_workers", 4),
        )
    else:
        # Fallback to defaults or raise error
        print(f"Warning: Could not infer dataset from name '{dataset_name}'. Defaulting to COCO if paths exist.")
        data_cfg = config.get("data", {})
        return CocoDataModule(
            train_root=data_cfg.get("train_root", "./data/coco17/train2017"),
            train_ann=data_cfg.get("train_ann", "./data/coco17/annotations/captions_train2017.json"),
            val_root=data_cfg.get("val_root", "./data/coco17/val2017"),
            val_ann=data_cfg.get("val_ann", "./data/coco17/annotations/captions_val2017.json"),
            train_batch_size=trainer_cfg.get("train_batch_size", 32),
            val_batch_size=trainer_cfg.get("val_batch_size", 32),
            resolution=config.get("encoder", {}).get("resolution", 128),
            num_workers=4,
        )
