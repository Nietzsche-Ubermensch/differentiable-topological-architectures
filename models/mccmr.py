import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class MutualCompromiseMechanism(nn.Module):
    """
    Coordinates feature fusion by mutual compromise:
    each modality adapts toward the other before hash projection.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.img_to_txt_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.txt_to_img_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.img_compromise = nn.Linear(dim * 2, dim)
        self.txt_compromise = nn.Linear(dim * 2, dim)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_q = img_feat.unsqueeze(1)
        txt_q = txt_feat.unsqueeze(1)
        img_from_txt, _ = self.img_to_txt_attn(img_q, txt_q, txt_q)
        txt_from_img, _ = self.txt_to_img_attn(txt_q, img_q, img_q)
        img_compromise = torch.relu(self.img_compromise(torch.cat([img_feat, img_from_txt.squeeze(1)], dim=-1)))
        txt_compromise = torch.relu(self.txt_compromise(torch.cat([txt_feat, txt_from_img.squeeze(1)], dim=-1)))
        return img_compromise, txt_compromise


class MCCMR(nn.Module):
    """
    Mutual Compromise Cross-Modal Retrieval.
    Mutual compromise mechanism for feature fusion coordination.
    Coordinates feature fusion to enhance semantic consistency
    across modalities prior to metric distance evaluation.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, hidden: int = 512, bit_len: int = 64):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.mcm = MutualCompromiseMechanism(hidden)
        self.img_hash = nn.Linear(hidden, bit_len)
        self.txt_hash = nn.Linear(hidden, bit_len)
        self.loss_fn = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_h = torch.relu(self.img_proj(img_feat))
        txt_h = torch.relu(self.txt_proj(txt_feat))
        img_comp, txt_comp = self.mcm(img_h, txt_h)
        img_hash = torch.tanh(self.img_hash(img_comp))
        txt_hash = torch.tanh(self.txt_hash(txt_comp))
        return img_hash, txt_hash
