import torch
from torch import nn, einsum
from torch.nn.functional import one_hot


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
        self.codebook = nn.Parameter(torch.zeros(codebook_size, embed_dim),
                                     requires_grad=True)
        self.codebook.data.uniform_(
            -1.0 / self.embed_dim, 1.0 / self.embed_dim
            )
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
        # (N, 1, d)
        flat_z = z.view(-1, 1, self.embed_dim)
        # (N, n)
        dist = (
            (flat_z**2).sum(2)
            + (self.codebook.data**2).sum(1)
            - 2 * (flat_z.squeeze(1) @ self.codebook.data.T)
        )
        # dist = torch.norm(flat_z - self.codebook, dim=2)
        # (N, )
        encoding = torch.argmin(dist, dim=1)
        # (N, n)
        idx = one_hot(
            encoding, num_classes=self.codebook_size).float()
        #  (N, n) * (n, d) -> (N, d)
        code = idx @ self.codebook.data
        # (b, c, h, w)
        code = code.view(z.shape)
        codebook_loss = nn.MSELoss()(code, z.detach())
        commitment_loss = self.commit_cost * nn.MSELoss()(z, code.detach())
        # straight-through estimator (dc/dz)
        code = z + (code - z).detach()
 
        return code, commitment_loss, codebook_loss, idx
    
    
class EMAQuantizer(nn.Module):
    
    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        commit_cost: float = 0.25,
    ):
        super(EMAQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.commit_cost = commit_cost
        self.register_buffer(
            "codebook",
            torch.FloatTensor(torch.randn((codebook_size, embed_dim)))
        )
        self.register_buffer("n_i", torch.zeros((codebook_size,)))
        self.register_buffer(
            "e_i", self.get_buffer("codebook").data.clone()
            )
        self.register_buffer("decay", torch.tensor(0.99))
        self.register_buffer("eps", torch.tensor(1e-5))

    def forward(self, z):
        b, c, h, w = z.shape
        # (N, 1, d)
        flat_z = z.view(-1, 1, self.embed_dim)
        # (N, n)
        dist = (
            (flat_z**2).sum(2)
            + (self.codebook.data**2).sum(1)
            - 2 * (flat_z.squeeze(1) @ self.codebook.data.T)
        )
        # (N, )
        encoding = torch.argmin(dist, dim=1)
        # (N, n)
        idx = one_hot(
            encoding, num_classes=self.codebook_size).float()
        #  (N, n) * (n, d) -> (N, d)
        code = idx @ self.codebook.data
        # (b, c, h, w)
        code = code.view(z.shape)
        
        if self.training:
            with torch.no_grad():
                # (n, N) * (N, d) -> (n, d)
                code_update = idx.T @ flat_z.squeeze(1)
                # (n, )
                n_i = self.decay * self.get_buffer("n_i") + \
                    (1 - self.decay) * idx.sum(0)
                #  stable n (Laplace smoothing)
                self.n_i = (n_i + self.eps) / \
                    (b + self.codebook_size * self.eps) * b
                # (n, d)
                self.e_i = self.decay * self.e_i + \
                    (1 - self.decay) * code_update
                # update codebook
                self.codebook.data = self.e_i / self.n_i.unsqueeze(1)
        commitment_loss = self.commit_cost * nn.MSELoss()(z, code.detach())
        # straight-through estimator (dc/dz)
        code = z + (code - z).detach()

        return code, commitment_loss, None, idx
        

class GumbleQuantizer(nn.Module):
    
    def __init__(
        self,
        codebook_size=512,
        embed_dim=32,
        tau=1,
        kld_scale=5e-4
    ):
        super(GumbleQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.tau = tau
        self.kld_scale = kld_scale
        self.proj = nn.Conv2d(embed_dim, codebook_size, 1)
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        
    def forward(self, z):
        z = self.proj(z)
        hard = False if self.training else True
        weight_logits = nn.functional.gumbel_softmax(
            z, tau=self.tau, hard=hard, dim=1
        )
        logits = einsum('b n h w, n d -> b d h w', weight_logits, self.codebook.weight)
        qy = nn.functional.softmax(logits, dim=1)
        loss = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        return logits, loss, None, weight_logits.argmax(1)


def build_quantizer(config):
    if config["type"]=="ema":
        return EMAQuantizer(**config)
    elif config["type"]=="gumbel":
        return GumbleQuantizer(**config)
    else:
        return VectorQuantizer(**config)