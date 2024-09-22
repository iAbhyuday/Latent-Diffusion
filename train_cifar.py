#%%
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    Normalize,
    Resize,
    ToTensor,
    Compose,
)
from torchvision.datasets import CIFAR10
from latent_diffusion.models import VQVAE


#%%
with open("configs/cifar10.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]
# %%
torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device(trainer_cfg["device"])
writer = SummaryWriter(
    log_dir=os.path.join(trainer_cfg["tensorboard_log_dir"],trainer_cfg["name"])
    )
# %%
t_transforms = Compose(
    [
        Resize(32),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

v_transforms = Compose(
    [Resize(32), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
train_data = CIFAR10(
    root="data",
    train=True,
    transform=t_transforms,
    download=True
    )
data_var = np.var(train_data.data)
print(data_var)
val_data = CIFAR10(
    root="data",
    train=False,
    transform=v_transforms,
    download=True
    )
# %%
train_data = DataLoader(
    dataset=train_data,
    batch_size=trainer_cfg["trainw_batch_size"], shuffle=True, num_workers=4, drop_last=True
)
val_data = DataLoader(
    dataset=val_data, batch_size=trainer_cfg["val_batch_size"], shuffle=True, num_workers=4
    )
#%%
model = VQVAE(cfg)
# %%
