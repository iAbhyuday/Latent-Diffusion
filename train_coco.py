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
with open("configs/coco17-vq.yaml", encoding="utf-8") as f:
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

train_data = DataLoader(train_data, batch_size=trainer_cfg["train_batch_size"], shuffle=True, num_workers=16, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)
val_data = DataLoader(val_data, batch_size=trainer_cfg["val_batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

x, y = next(iter(train_data))
#%%
model = VQVAE(cfg).to(trainer_cfg["device"])
percept_loss = PerceptualLoss(**cfg["perceptual_loss"]).to(trainer_cfg["device"])
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#%%
if cfg["trainer"]["load_ckpt"]:
    checkpoint = torch.load(cfg["trainer"]["load_ckpt"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

#%%
lr = float(trainer_cfg["lr"])
n_epochs = 20
num_batches = len(train_data)
sched = ReduceLROnPlateau(optim, mode="min", min_lr=1e-7, threshold=1e-4, factor=0.5, patience=5)
# %%


best=10
for e in range(0, n_epochs):
    r_recon = 0.0
    r_com = 0.0
    r_cdl = 0.0
    r_pl = 0.0
    r_ppl = 0.0

    with tqdm(train_data, unit="batch", desc=f"Epoch {e+1}", position=0, leave=True) as data:
        model.train()
        for batch_idx, (x,_) in enumerate(data):
            writer.add_scalar("lr", optim.param_groups[0]["lr"], e*num_batches + batch_idx)
            x = x.to(device)

            x_, cd, cl, cdl, rl, enc = model(x)
            pl = percept_loss(x_, x)
            loss = cl + rl + pl
            
            r_recon += rl.item()
            r_com += cl.item()
            r_pl += pl.item()
            ppl, clp = measure_perplexity(enc, cfg["quantizer"]["codebook_size"])
            r_ppl += ppl.item()
            if not cfg["quantizer"]["use_ema"]:
                loss += cdl
                r_cdl += cdl.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            
            data.set_postfix(
                {
                    "recon_loss": r_recon / (batch_idx+1),
                    "commit_loss": r_com / (batch_idx+1),
                    "codebook_loss": r_cdl / (batch_idx+1),
                    "percept_loss": r_pl / (batch_idx+1),
                    "perplexity": r_ppl/ (batch_idx+1)
                }
            )
        
        writer.add_scalar("recon_loss", r_recon / num_batches, e)
        writer.add_scalar("commit_loss", r_com / num_batches, e)
        writer.add_scalar("codebook_loss", r_cdl / num_batches, e)
        writer.add_scalar("perceptual_loss", r_pl / num_batches, e)
        writer.add_scalar("perplexity", r_ppl / num_batches, e)
    
    rec_loss = (r_recon + r_pl)/num_batches
    sched.step(r_recon/num_batches)
    if rec_loss < best:
        best = rec_loss
        checkpoint = {
            "epoch": e,
            "lr": optim.param_groups[0]["lr"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            }

        torch.save(checkpoint, f"{trainer_cfg['name']}_best.pth")


    if e % 1 == 0:
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


# %%
checkpoint = {
    "epoch": n_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
}

torch.save(checkpoint, f"{trainer_cfg['name']}_e{n_epochs}.pth")
writer.close()
