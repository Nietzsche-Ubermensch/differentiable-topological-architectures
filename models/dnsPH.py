import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class C2BFNet(nn.Module):
    """
    Context-aware Cross-layer Bilinear Fusion Network.
    LSTM for spatial feature modeling + cross-layer bilinear pooling.
    """

    def __init__(self, img_dim: int, txt_dim: int, hidden: int = 512):
        super().__init__()
        self.img_lstm = nn.LSTM(img_dim, hidden, batch_first=True, num_layers=2)
        self.txt_lstm = nn.LSTM(txt_dim, hidden, batch_first=True, num_layers=2)
        self.bilinear = nn.Bilinear(hidden, hidden, hidden)

    def forward(self, img_seq: torch.Tensor, txt_seq: torch.Tensor):
        img_out, _ = self.img_lstm(img_seq)
        txt_out, _ = self.txt_lstm(txt_seq)
        img_ctx = img_out[:, -1]
        txt_ctx = txt_out[:, -1]
        fused = self.bilinear(img_ctx, txt_ctx)
        return fused, img_ctx, txt_ctx


class DNsPH(nn.Module):
    """
    Dense Neighborhood-structure Preserving Hashing.
    C2BF-Net + LSTM spatial modeling.
    Fuses image, text, semantic labels via adaptive weighting
    to reconstruct joint semantic similarity matrix with fine-grained neighborhoods.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, label_dim: int = 80, bit_len: int = 64):
        super().__init__()
        self.c2bf = C2BFNet(img_dim, txt_dim)
        self.label_proj = nn.Linear(label_dim, 512)
        self.adaptive_weight = nn.Linear(512 * 3, 3)  # Weights for img, txt, label
        self.hash_proj = nn.Linear(512, bit_len)
        self.loss_fn = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, labels: torch.Tensor):
        fused, img_ctx, txt_ctx = self.c2bf(
            img_feat.unsqueeze(1), txt_feat.unsqueeze(1)
        )
        label_f = torch.relu(self.label_proj(labels.float()))
        combined = torch.cat([img_ctx, txt_ctx, label_f], dim=-1)
        weights = torch.softmax(self.adaptive_weight(combined), dim=-1)
        weighted = weights[:, 0:1] * img_ctx + weights[:, 1:2] * txt_ctx + weights[:, 2:3] * label_f
        hash_codes = torch.tanh(self.hash_proj(weighted))
        return hash_codes
