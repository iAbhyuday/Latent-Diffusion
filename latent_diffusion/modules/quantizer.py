import torch
from torch import nn, einsum
from torch.nn.functional import one_hot
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector quantizer module.
    Parameters:
        codebook_size (int): number of codebook vectors
        embed_dim (int): dimensions codebook vectors
        commit_cost (float): commit cost (beta) for commitment loss term.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        commit_cost: float = 0.25,
    ):
        super(VectorQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.commit_cost = commit_cost

    def forward(self, z: torch.Tensor):
        """
        The [`Quantizer`] forward method.

        Args:
            z (`torch.Tensor`):
                Latent embedding from encoder of shape [N, C, H, W]
        Returns:
            `tuple`:
                quantized_code, commitment_loss, codebook_loss,
                codebook_indices
        """
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        # (N, d)
        flat_z = z.view(-1, self.embed_dim)
        # (N, n)
        dist = (
            (flat_z**2).sum(1)
            + (self.codebook.weight**2).sum(1)
            - 2 * (flat_z @ self.codebook.weight.T)
        )
        # (N, )
        encoding = torch.argmin(dist, dim=1).unsqueeze(1)
        # (N, n)
        idx = torch.zeros(encoding.shape[0], self.codebook_size, device=z.device)
        idx.scatter_(1, encoding, 1)
        #  (N, n) * (n, d) -> (N, d)
        code = idx @ self.codebook.weight
        # (b, c, h, w)
        code = code.view(z.shape)
        codebook_loss = nn.MSELoss()(code, z.detach())
        commitment_loss = self.commit_cost * nn.MSELoss()(z, code.detach())
        # straight-through estimator (dc/dz)
        code = z + (code - z).detach()

        e_mean = torch.mean(idx, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        code = code.permute(0, 3, 1, 2).contiguous()
        return code, commitment_loss, codebook_loss,  perplexity


class EMAQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        commit_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super(EMAQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.commit_cost = commit_cost

        self.register_buffer(
            "codebook", torch.randn(codebook_size, embed_dim)
        )
        self.register_buffer("n_i", torch.zeros(codebook_size))
        self.register_buffer("e_i", self.codebook.clone())
        self.decay = decay
        self.eps = eps

        nn.init.uniform_(
            self.codebook.data, 
            -1.0 / self.codebook_size, 
            1.0 / self.codebook_size
        )

    def forward(self, z):
        b, c, h, w = z.shape
        flat_z = z.view(-1, self.embed_dim)   # (N, d)

        # Compute distances
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            + self.codebook.pow(2).sum(1)
            - 2 * flat_z @ self.codebook.T
        )

        # Encoding indices
        encoding = torch.argmin(dist, dim=1)
        idx = F.one_hot(encoding, num_classes=self.codebook_size).type(flat_z.dtype)

        # Quantized output
        code = idx @ self.codebook
        code = code.view(z.shape)

        if self.training:
            with torch.no_grad():
                # EMA cluster sizes
                encodings_sum = idx.sum(0)  # (n,)
                self.n_i.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

                # EMA embedding sum
                embed_sum = idx.T @ flat_z  # (n, d)
                self.e_i.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.n_i.sum()
                smoothed_cluster_size = (
                    (self.n_i + self.eps) 
                    / (n + self.codebook_size * self.eps) * n
                )
                embed_normalized = self.e_i / smoothed_cluster_size.unsqueeze(1)
                self.codebook.data.copy_(embed_normalized)

        commitment_loss = self.commit_cost * F.mse_loss(z, code.detach())

        code = z + (code - z).detach()


        avg_probs = idx.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return code, commitment_loss, None, perplexity


class GumbleQuantizer(nn.Module):

    def __init__(self, codebook_size=512, embed_dim=32, tau=1, kld_scale=5e-4):
        super(GumbleQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.tau = tau
        self.kld_scale = kld_scale
        self.proj = nn.Conv2d(embed_dim, codebook_size, 1)
        self.codebook = nn.Embedding(codebook_size, embed_dim)

    def forward(self, z):
        logits = self.proj(z)
        hard = False if self.training else True
        soft_logits = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=1)
        code = einsum("b n h w, n d -> b d h w", soft_logits, self.codebook.weight)
        qy = nn.functional.softmax(logits, dim=1)
        loss = self.kld_scale * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        
        return code, loss, None, soft_logits.argmax(1)



def build_quantizer(config):
    if config["type"] == "ema":
        return EMAQuantizer(**config["params"])
    elif config["type"] == "gumbel":
        return GumbleQuantizer(**config["params"])
    else:
        return VectorQuantizer(**config["params"])
