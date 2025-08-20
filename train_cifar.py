import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.datasets import CIFAR10
from latent_diffusion.models import VQVAE
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.utils.metrics import measure_perplexity
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

with open("configs/cifar10-kl.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]

t_transforms = Compose([Resize(32), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
v_transforms = Compose([Resize(32), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, val_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):
        self.train_data = CIFAR10(root="data", train=True, transform=t_transforms, download=True)
        self.val_data = CIFAR10(root="data", train=False, transform=v_transforms, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=4)


data_module = CIFARDataModule(
    train_batch_size=trainer_cfg["train_batch_size"],
    val_batch_size=trainer_cfg["val_batch_size"]
)

logger = TensorBoardLogger(
    save_dir=trainer_cfg["tensorboard_log_dir"],
    name=trainer_cfg["name"]
)
class ImageReconstructionCallback(pl.callbacks.Callback):
    def __init__(self, num_images=8):
        super().__init__()
        self.num_images = num_images

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        images = images[:self.num_images]
        with torch.no_grad():
            reconstructions = pl_module(images.to(pl_module.device))
        # If model returns tuple (recon, ...), take first
        if isinstance(reconstructions, tuple):
            reconstructions = reconstructions[0]
        # Denormalize for visualization
        def denorm(x):
            return x * 0.5 + 0.5
        img_grid = torch.cat([denorm(images.cpu()), denorm(reconstructions.cpu())], dim=0)
        trainer.logger.experiment.add_images(
            "Validation/Reconstruction",
            img_grid,
            global_step=trainer.global_step
        )

trainer = pl.Trainer(
    max_epochs=20,
    logger=logger,
    accelerator=trainer_cfg["device"],
    log_every_n_steps=10,
    callbacks=[ImageReconstructionCallback(num_images=8)]
)
model = VQVAE(cfg)
trainer.fit(model, datamodule=data_module)
