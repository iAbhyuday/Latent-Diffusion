#%%
import os
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from latent_diffusion.models import VQVAE
from latent_diffusion.modules import PerceptualLoss
from latent_diffusion.utils.metrics import measure_perplexity
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

transforms = Compose(
    [
        Resize((resolution, resolution)),
        ToTensor(),
    ]
)


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


train_data = CocoCaptions(
    root="/media/z004e29c/mlinux/coco17/train2017",
    annFile="/media/z004e29c/mlinux/coco17/annotations/captions_train2017.json",
    transform=transforms,
    )
val_data = CocoCaptions(
    root="/media/z004e29c/mlinux/coco17/val2017",
    annFile="/media/z004e29c/mlinux/coco17/annotations/captions_val2017.json",
    transform=transforms,
    )

train_data = DataLoader(train_data, batch_size=trainer_cfg["train_batch_size"], shuffle=True, num_workers=16, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)
val_data = DataLoader(val_data, batch_size=trainer_cfg["val_batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

x, y = next(iter(train_data))
#%%
model = VQVAE(cfg).to(trainer_cfg["device"])
percept_loss = PerceptualLoss(cfg["perceptual_loss"])

#%%

#%%
lr = float(trainer_cfg["lr"])
n_epochs = trainer_cfg["epochs"]
num_batches = len(train_data)
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#optim.load_state_dict(checkpoint['optimizer_state_dict'])
#sched = CosineAnnealingWarmRestarts(optim, T_0=num_batches*10, eta_min =7e-7)
# %%



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
            pl = percept_loss(x_, x).item()
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
            #sched.step()
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
    if e % 1 == 0:
        with torch.no_grad():
            model.eval()
            x, _ = next(iter(val_data))
            x = x.to(device)
            x_, cd, cl, cdl, rl, pl, _ = model(x)
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

