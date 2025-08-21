#%%
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.datasets import CocoCaptions
from latent_diffusion.models import VQVAE
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.utils.metrics import measure_perplexity
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import setproctitle
setproctitle.setproctitle("DoNotKill: abhyuday")
#%%
print("IMPORT COMPLETE")
with open("configs/coco17.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]

#%%%
writer = SummaryWriter(
    log_dir=os.path.join(trainer_cfg["tensorboard_log_dir"], trainer_cfg["name"])
    )
torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device(trainer_cfg["device"])
resolution = cfg["encoder"]["resolution"]
#%%
resolution = cfg["encoder"]["resolution"]
t_transforms = Compose([Resize((resolution, resolution)), ToTensor()])
v_transforms = Compose([Resize((resolution, resolution)), ToTensor()])

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
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)



data_module = CocoDataModule(
    train_batch_size=trainer_cfg["train_batch_size"],
    val_batch_size=trainer_cfg["val_batch_size"]
)
#%%
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


# sched = ReduceLROnPlateau(optim, mode="min", min_lr=1e-7, threshold=1e-4, factor=0.5, patience=5)

trainer = pl.Trainer(
    max_epochs=30,
    logger=logger,
    accelerator=trainer_cfg["device"],
    log_every_n_steps=10,
    accumulate_grad_batches=4,
    callbacks=[
        ImageReconstructionCallback(num_images=8),
        ModelCheckpoint(
            monitor="val_recon_loss_epoch",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_recon_loss_epoch:.2f}",
            save_last=True,
        )
    ],

)

model = VQVAE(cfg)
trainer.fit(model, datamodule=data_module)