#%%
import os
import yaml
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, Lambda
from torchvision.datasets import CocoCaptions
from latent_diffusion.models import VAE
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.utils.metrics import measure_perplexity
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import setproctitle
from pytorch_lightning.loggers import WandbLogger
#%%
print("IMPORT COMPLETE")
with open("configs/coco17-kl.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]

#%%%
writer = SummaryWriter(
    log_dir=os.path.join(trainer_cfg["tensorboard_log_dir"], trainer_cfg["name"])
    )
wandb_logger = WandbLogger(
    project=trainer_cfg["name"],
    name="silu-16x16x8-coco",
    log_model=False
)

torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device(trainer_cfg["device"])
resolution = cfg["encoder"]["resolution"]
#%%
resolution = cfg["encoder"]["resolution"]
t_transforms = Compose([Resize((resolution, resolution)), ToTensor(), Lambda(lambda x: x * 2 - 1)])
v_transforms = Compose([Resize((resolution, resolution)), ToTensor(), Lambda(lambda x: x * 2 - 1)])

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
    def __init__(self, train_batch_size, val_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):
        self.train_data = CocoCaptions(
                root="/media/z004e29c/mlinux/coco17/train2017",
                annFile="/media/z004e29c/mlinux/coco17/annotations/captions_train2017.json",
                transform=t_transforms,
            )
        self.val_data = CocoCaptions(
                root="/media/z004e29c/mlinux/coco17/val2017",
                annFile="/media/z004e29c/mlinux/coco17/annotations/captions_val2017.json",
                transform=v_transforms,
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True,drop_last=True)



data_module = CocoDataModule(
    train_batch_size=trainer_cfg["train_batch_size"],
    val_batch_size=trainer_cfg["val_batch_size"]
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



import pytorch_lightning as pl
import torch
import wandb

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



trainer = pl.Trainer(
    max_epochs=trainer_cfg["max_epochs"],
    logger=wandb_logger,
    accelerator=trainer_cfg["device"],
    log_every_n_steps=10,
    precision="16-mixed",
    accumulate_grad_batches=4,
    callbacks=[
        EMACallback(),
        ImageReconstructionCallback(num_images=16),
        ModelCheckpoint(
            dirpath=trainer_cfg["checkpoint_dir"],
            monitor="val_recon_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-kl",
            save_last=True,
        )
    ],

)
model = VAE.load_from_checkpoint(cfg["trainer"]["load_ckpt"], strict=False) if cfg["trainer"]["load_ckpt"] else VAE(cfg)
trainer.fit(model, datamodule=data_module)