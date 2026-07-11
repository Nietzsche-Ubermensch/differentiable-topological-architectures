import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class ProgressiveQuantizationAnnealing(nn.Module):
    """Branch-wise annealing: sharpens tanh approximation over epochs."""

    def __init__(self, bit_len: int, n_branches: int = 3):
        super().__init__()
        self.branches = nn.ModuleList([nn.Linear(512, bit_len) for _ in range(n_branches)])
        self.alpha = 1.0  # Annealing temperature — increase each epoch

    def anneal(self, epoch: int, max_alpha: float = 10.0, n_epochs: int = 100):
        self.alpha = 1.0 + (max_alpha - 1.0) * (epoch / n_epochs)

    def forward(self, branch_feats: list) -> list:
        return [torch.tanh(self.alpha * b(f)) for b, f in zip(self.branches, branch_feats)]


class StochasticGate(nn.Module):
    """Patient context-adaptive stochastic gating to prevent semantic fragmentation."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        g = torch.sigmoid(self.gate_proj(x))
        if training:
            noise = torch.bernoulli(g)
            return x * noise
        return x * g


class TriPAH(nn.Module):
    """
    Tri-Prompt Affinity Hashing.
    Tri-view: images + textual reports + learnable prompt views.
    Branch-wise progressive quantization annealing +
    imbalance-aware classification to prevent discretization collapse.
    Designed for noisy medical cross-modal retrieval.
    """

    def __init__(self, img_dim: int = 768, txt_dim: int = 768, bit_len: int = 64, n_classes: int = 14):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, 512), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, 512), nn.ReLU())
        self.prompt_embed = nn.Parameter(torch.randn(1, 512))  # Learnable prompt view
        self.stochastic_gate = StochasticGate(512)
        self.pqa = ProgressiveQuantizationAnnealing(bit_len, n_branches=3)
        self.classifier = nn.Linear(bit_len, n_classes)
        self.loss_fn = DifferentiableHashDistance(bit_len=bit_len)
        # Imbalance-aware: per-class frequency weights updated during training
        self.register_buffer('class_weights', torch.ones(n_classes))

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_f = self.img_proj(img_feat)
        txt_f = self.txt_proj(txt_feat)
        prompt_f = self.prompt_embed.expand(img_f.size(0), -1)
        img_f = self.stochastic_gate(img_f, self.training)
        branch_feats = [img_f, txt_f, prompt_f]
        hashes = self.pqa(branch_feats)  # List of 3 hash tensors
        fused = torch.stack(hashes, dim=1).mean(dim=1)
        cls_logits = self.classifier(fused)
        return fused, hashes, cls_logits

    def set_epoch(self, epoch: int):
        self.pqa.anneal(epoch)
