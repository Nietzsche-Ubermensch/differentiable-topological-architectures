import torch
import torch.nn as nn
import torch.nn.functional as F
from hash.differentiable_hash_distance import DifferentiableHashDistance


class JSDivergenceLoss(nn.Module):
    """Jensen-Shannon divergence as differentiable alignment loss."""

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = F.softmax(p, dim=-1) + 1e-8
        q = F.softmax(q, dim=-1) + 1e-8
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') +
                      F.kl_div(m.log(), q, reduction='batchmean'))


class AttentionReweighting(nn.Module):
    """Attention-driven sample re-weighting for label-wise alignment."""

    def __init__(self, dim: int, n_labels: int):
        super().__init__()
        self.label_embed = nn.Embedding(n_labels, dim)
        self.score = nn.Linear(dim * 2, 1)

    def forward(self, feat: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        label_e = self.label_embed(label_ids)      # (B, n_labels, dim)
        feat_exp = feat.unsqueeze(1).expand_as(label_e)
        combined = torch.cat([feat_exp, label_e], dim=-1)
        weights = torch.sigmoid(self.score(combined)).squeeze(-1)  # (B, n_labels)
        return (weights.unsqueeze(-1) * label_e).sum(dim=1)


class LDSAH(nn.Module):
    """
    Label-Driven Semantic Alignment Hashing.
    JS divergence + attention-driven sample re-weighting.
    Label-wise semantic alignment penalizes dissimilarity across heterogeneous latent spaces.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, n_labels: int = 80, bit_len: int = 64):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, 512), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, 512), nn.ReLU())
        self.img_reweight = AttentionReweighting(512, n_labels)
        self.txt_reweight = AttentionReweighting(512, n_labels)
        self.img_hash = nn.Linear(512, bit_len)
        self.txt_hash = nn.Linear(512, bit_len)
        self.js_loss = JSDivergenceLoss()
        self.hash_loss = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, label_ids: torch.Tensor):
        img_f = self.img_proj(img_feat)
        txt_f = self.txt_proj(txt_feat)
        img_f = img_f + self.img_reweight(img_f, label_ids)
        txt_f = txt_f + self.txt_reweight(txt_f, label_ids)
        img_hash = torch.tanh(self.img_hash(img_f))
        txt_hash = torch.tanh(self.txt_hash(txt_f))
        return img_hash, txt_hash

    def alignment_loss(self, img_hash: torch.Tensor, txt_hash: torch.Tensor) -> torch.Tensor:
        return self.js_loss(img_hash, txt_hash)
