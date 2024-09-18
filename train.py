#%%
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from latent_diffusion.models.vqvae import VQVAE
#%%
torch.backends.cudnn.benchmark = True
scaler = GradScaler()
writer = SummaryWriter(
    log_dir="/mnt/sda/ab/projects/FY24/cmc/logs/VQAE-COCO-224"
    )
torch.set_printoptions(precision=3, sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%

transforms = Compose(
    [
        Resize((112, 112)),
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

train_data = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=16, collate_fn=collate_fn, prefetch_factor=2, pin_memory=True)
val_data = DataLoader(val_data, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

x, y = next(iter(train_data))
print(x.shape)
print(x.min(), x.max())
print(y)
#%%
use_ema = True
codebook_size = 128
lr = 3e-4
n_epochs = 50

x = x.to(device)
model = VQVAE(embed_dim=14*14, use_ema=use_ema).to(device)
code = model.encoder(x)
print(code.shape)
code, commitment_loss, codebook_loss, encoding = model.vq(code)
print(code.shape)
x_ = model.decoder(code)
print(x_.shape)
optim = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
# %%


num_batches = len(train_data)
for e in range(0, n_epochs):
    r_recon = 0.0
    r_com = 0.0
    r_cdl = 0.0
    r_pl = 0.0

    with tqdm(train_data, unit="batch", desc=f"Epoch {e+1}", position=0, leave=True) as data:
        model.train()
        for batch_idx, (x,_) in enumerate(data):

            x = x.to(device)

            x_, cd, cl, cdl, rl, pl, enc = model(x)
            loss = cl + rl + pl
            
            r_recon += rl.item()
            r_com += cl.item()
            r_pl += pl.item()

            if not use_ema:
                loss += cdl
                r_cdl += cdl.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
            

            data.set_postfix(
                {
                    "recon_loss": r_recon / num_batches,
                    "commit_loss": r_com / num_batches,
                    "codebook_loss": r_cdl / num_batches,
                    "percept_loss": r_pl / num_batches
                }
            )
        writer.add_scalar("recon_loss", r_recon / num_batches, e)
        writer.add_scalar("commit_loss", r_com / num_batches, e)
        writer.add_scalar("codebook_loss", r_cdl / num_batches, e)
        writer.add_scalar("perceptual_loss", r_pl / num_batches, e)

        data.set_postfix(
            {
                "recon_loss": r_recon / num_batches,
                "commit_loss": r_com / num_batches,
                "codebook_loss": r_cdl / num_batches,
                "percept_loss": r_pl / num_batches
            }
        )

        if e % 2 == 0:
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

torch.save(checkpoint, "model_checkpoint.pth")
writer.close()