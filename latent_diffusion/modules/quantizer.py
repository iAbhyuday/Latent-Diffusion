import torch 
from torch import nn


class Quantizer(nn.Module):

    def __init__(
        self,
        codebook_len: int = 512,
        embed_dim: int = 256,
        commit_cost: float = 0.25,
        use_ema: bool = True,
    ):
        """
        Vector quantizer module.
        Parameters:
            codebook_len (int): number of codebook vectors
            embed_dim (int): dimensions codebook vectors
            commit_cost (float): commit cost (beta) for commitment loss term.
            use_ema (bool): use ema for codebook update
        """
        super(Quantizer, self).__init__()
        self.embed_dim = embed_dim
        self.codebook_len = codebook_len
        self.use_ema = use_ema
        if self.use_ema:
            self.register_buffer(
                "codebook",
                torch.FloatTensor(torch.randn((codebook_len, embed_dim)))
            )
            self.register_buffer("n_i", torch.zeros((codebook_len,)))
            self.register_buffer(
                "e_i", self.get_buffer("codebook").data.clone()
                )
            self.register_buffer("decay", torch.tensor(0.99))
            self.register_buffer("eps", torch.tensor(1e-5))
        else:
            self.codebook = nn.Parameter(
                torch.rand((self.codebook_len, embed_dim)), requires_grad=True
            )
        self.commit_cost = commit_cost


    def forward(self, z: torch.Tensor):
        b, c, h, w = z.shape
        # (N, 1, d)
        flat_z = z.view(-1, 1, self.embed_dim)
        # (N, n)
        dist = (
            (flat_z**2).sum(2)
            + (self.codebook**2).sum(1)
            - 2 * (flat_z.squeeze(1) @ self.codebook.T)
        )
        # dist = torch.norm(flat_z - self.codebook, dim=2)

        # (N, )
        encoding = torch.argmin(dist, dim=1)

        # (N, n)
        idx = nn.functional.one_hot(
            encoding,
            num_classes=self.codebook_len
            ).float()
        #  (N, n) * (n, d) -> (N, d)
        code = idx @ self.codebook.data
        # (b, c, h, w)
        code = code.view(z.shape)
        if self.use_ema:
            if self.training:
                with torch.no_grad():
                    # (n, N) * (N, d) -> (n, d)
                    code_update = idx.T @ flat_z.squeeze(1)
                    # (n, )
                    n_i = self.decay * self.n_i + (1 - self.decay) * idx.sum(0)
                    #  stable n (Laplace smoothing)
                    self.n_i = (n_i + self.eps) / \
                        (b + self.codebook_len * self.eps) * b
                    # (n, d)
                    self.e_i = self.decay * self.e_i + \
                        (1 - self.decay) * code_update
                    # update codebook
                    self.codebook.data = self.e_i / self.n_i.unsqueeze(1)
            codebook_loss = None
        else:
            codebook_loss = nn.MSELoss()(code, z.detach())
        commitment_loss = self.commit_cost * nn.MSELoss()(z, code.detach())
        # straight-through estimator (dc/dz)
        code = z + (code - z).detach()

        return code, commitment_loss, codebook_loss, encoding
