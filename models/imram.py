import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class MemoryAccumulationModule(nn.Module):
    """Accumulates cross-modal alignment context across recurrent steps."""

    def __init__(self, dim: int, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(dim, num_heads=8, batch_first=True) for _ in range(n_steps)])
        self.memory_gate = nn.GRUCell(dim, dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        memory = torch.zeros_like(query)
        for step in range(self.n_steps):
            attended, _ = self.attn_layers[step](query.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1))
            attended = attended.squeeze(1)
            memory = self.memory_gate(attended, memory)
        return memory


class IMRAM(nn.Module):
    """
    Iterative Matching with Recurrent Attention Memory.
    Recurrent attention + memory accumulation.
    Progressively aligns modality fragments to capture shifting semantics.
    """

    def __init__(self, img_dim: int = 2048, txt_dim: int = 768, hidden: int = 512, n_steps: int = 3):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.img_memory = MemoryAccumulationModule(hidden, n_steps)
        self.txt_memory = MemoryAccumulationModule(hidden, n_steps)
        self.loss_fn = DifferentiableHashDistance(bit_len=hidden)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_h = torch.relu(self.img_proj(img_feat))
        txt_h = torch.relu(self.txt_proj(txt_feat))
        img_aligned = self.img_memory(img_h, txt_h)
        txt_aligned = self.txt_memory(txt_h, img_h)
        return img_aligned, txt_aligned
