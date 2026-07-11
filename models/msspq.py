import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class UnifiedSemanticSpaceProjection(nn.Module):
    """
    Projects all modalities into a unified semantic space
    before mapping to binary codes.
    """

    def __init__(self, *input_dims, unified_dim: int = 512, bit_len: int = 64):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, unified_dim), nn.LayerNorm(unified_dim), nn.GELU())
            for d in input_dims
        ])
        self.cross_modal_align = nn.MultiheadAttention(unified_dim, num_heads=8, batch_first=True)
        self.hash_head = nn.Linear(unified_dim, bit_len)

    def forward(self, *modality_feats):
        projected = [proj(feat) for proj, feat in zip(self.projectors, modality_feats)]
        stacked = torch.stack(projected, dim=1)       # (B, n_modalities, unified_dim)
        aligned, _ = self.cross_modal_align(stacked, stacked, stacked)
        fused = aligned.mean(dim=1)
        return torch.tanh(self.hash_head(fused))


class MSSPQ(nn.Module):
    """
    Multi-modal Semantic Space Projection + Quantization.
    Unified semantic space projection.
    Maps diverse modalities into compact binary codes
    with explicit cross-modal semantic consistency constraints.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, bit_len: int = 64):
        super().__init__()
        self.projection = UnifiedSemanticSpaceProjection(img_dim, txt_dim, unified_dim=512, bit_len=bit_len)
        self.consistency_loss = nn.CosineEmbeddingLoss()
        self.hash_loss = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        hash_codes = self.projection(img_feat, txt_feat)
        return hash_codes

    def cross_modal_consistency(self, img_hash: torch.Tensor, txt_hash: torch.Tensor) -> torch.Tensor:
        target = torch.ones(img_hash.size(0), device=img_hash.device)
        return self.consistency_loss(img_hash, txt_hash, target)
