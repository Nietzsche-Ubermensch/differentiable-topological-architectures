import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class SemanticComponentDisentanglement(nn.Module):
    """SCD: competitive feature routing across semantic components."""

    def __init__(self, input_dim: int, n_components: int = 8, bit_len: int = 64):
        super().__init__()
        self.routers = nn.ModuleList([nn.Linear(input_dim, bit_len) for _ in range(n_components)])
        self.gate = nn.Linear(input_dim, n_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.gate(x), dim=-1)          # (B, n_components)
        components = torch.stack([r(x) for r in self.routers], dim=1)  # (B, n_comp, bit_len)
        out = (gates.unsqueeze(-1) * components).sum(dim=1)
        return torch.tanh(out)


class DCHUC(nn.Module):
    """
    Deep Cross-modal Hashing with Unified Codes.
    SCD + competitive feature routing.
    Iteratively learns unified hash codes for DB samples while
    simultaneously optimizing unseen query hashing functions.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, bit_len: int = 64, n_components: int = 8):
        super().__init__()
        self.img_scd = SemanticComponentDisentanglement(img_dim, n_components, bit_len)
        self.txt_scd = SemanticComponentDisentanglement(txt_dim, n_components, bit_len)
        self.db_codes = None  # Unified hash codes for database — set during training
        self.loss_fn = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_hash = self.img_scd(img_feat)
        txt_hash = self.txt_scd(txt_feat)
        return img_hash, txt_hash

    def update_db_codes(self, img_hash: torch.Tensor, txt_hash: torch.Tensor):
        """Iterative unified code update."""
        self.db_codes = torch.sign(0.5 * (img_hash + txt_hash)).detach()
