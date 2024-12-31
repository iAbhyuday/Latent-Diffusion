# %%
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
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.utils.metrics import measure_perplexity

# %%
with open("configs/cifar10.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
trainer_cfg = cfg["trainer"]
# %%
torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device(trainer_cfg["device"])
writer = SummaryWriter(
    log_dir=os.path.join(trainer_cfg["tensorboard_log_dir"], trainer_cfg["name"])
)
# %%
t_transforms = Compose(
    [Resize(32), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

v_transforms = Compose(
    [Resize(32), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
train_data = CIFAR10(root="data", train=True, transform=t_transforms, download=True)
data_var = np.var(train_data.data)
print(data_var)
val_data = CIFAR10(root="data", train=False, transform=v_transforms, download=True)
# %%
train_data = DataLoader(
    dataset=train_data,
    batch_size=trainer_cfg["trainw_batch_size"],
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
val_data = DataLoader(
    dataset=val_data,
    batch_size=trainer_cfg["val_batch_size"],
    shuffle=True,
    num_workers=4,
)
# %%
model = VQVAE(cfg)
# %%
lr = float(trainer_cfg["lr"])
n_epochs = 20
num_batches = len(train_data)

percept_loss = PerceptualLoss(**cfg["perceptual_loss"]).to(trainer_cfg["device"])
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


if cfg["trainer"]["load_ckpt"]:
    checkpoint = torch.load(cfg["trainer"]["load_ckpt"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])


best = 10
for e in range(0, n_epochs):
    r_recon = 0.0
    r_com = 0.0
    r_cdl = 0.0
    r_pl = 0.0
    r_ppl = 0.0

    with tqdm(
        train_data, unit="batch", desc=f"Epoch {e+1}", position=0, leave=True
    ) as data:
        model.train()
        for batch_idx, (x, _) in enumerate(data):
            writer.add_scalar(
                "lr", optim.param_groups[0]["lr"], e * num_batches + batch_idx
            )
            x = x.to(device)

            x_, cd, cl, cdl, rl, enc = model(x)
            pl = percept_loss(x_, x)
            loss = cl + rl + pl

            r_recon += rl.item()
            r_com += cl.item()
            r_pl += pl.item()
            ppl, clp = measure_perplexity(
                enc, cfg["quantizer"]["params"]["codebook_size"]
            )
            r_ppl += ppl.item()
            if cfg["quantizer"]["type"] == "vanilla":
                loss += cdl
                r_cdl += cdl.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            data.set_postfix(
                {
                    "recon_loss": r_recon / (batch_idx + 1),
                    "commit_loss": r_com / (batch_idx + 1),
                    "codebook_loss": r_cdl / (batch_idx + 1),
                    "percept_loss": r_pl / (batch_idx + 1),
                    "perplexity": r_ppl / (batch_idx + 1),
                }
            )

        writer.add_scalar("recon_loss", r_recon / num_batches, e)
        writer.add_scalar("commit_loss", r_com / num_batches, e)
        writer.add_scalar("codebook_loss", r_cdl / num_batches, e)
        writer.add_scalar("perceptual_loss", r_pl / num_batches, e)
        writer.add_scalar("perplexity", r_ppl / num_batches, e)

    rec_loss = (r_recon + r_pl) / num_batches
    # sched.step(r_recon/num_batches)
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
            model.eval()
            x, _ = next(iter(val_data))
            x = x.to(device)
            x_, cd, cl, cdl, rl, _ = model(x)
            writer.add_images("input", x, e)
            writer.add_images("output", x_, e)


# %%
checkpoint = {
    "epoch": n_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
}

torch.save(checkpoint, f"{trainer_cfg['name']}_e{n_epochs}.pth")
writer.close()
