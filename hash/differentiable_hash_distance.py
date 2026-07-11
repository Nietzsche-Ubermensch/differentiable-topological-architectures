import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableHashDistance(nn.Module):
    """
    Continuous surrogate for Hamming distance.
    Operates on tanh-constrained pre-binary activations.
    Provides pairwise + ranking-based differentiable loss.
    """

    def __init__(self, bit_len: int = 64, margin: float = 0.5):
        super().__init__()
        self.bit_len = bit_len
        self.margin = margin

    def hamming_surrogate(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Continuous Hamming approximation via inner product on tanh codes."""
        return 0.5 * (self.bit_len - (u @ v.T))

    def pairwise_log_likelihood_loss(
        self, u: torch.Tensor, v: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Log-likelihood pairwise loss.
        s[i,j] = 1 if similar, 0 if dissimilar.
        Theta[i,j] = 0.5 * u[i] . v[j]
        L = -sum(s*Theta - log(1 + exp(Theta)))
        """
        theta = 0.5 * (u @ v.T)
        loss = -torch.mean(s * theta - torch.log(1.0 + torch.exp(theta)))
        return loss

    def contrastive_loss(
        self, u: torch.Tensor, v: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive: minimize surrogate dist for similar, push apart for dissimilar."""
        dist = self.hamming_surrogate(u, v).diag()
        pos = labels * dist.pow(2)
        neg = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return torch.mean(pos + neg)

    def ndcg_ranking_loss(
        self, scores: torch.Tensor, relevance: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable NDCG surrogate for ranking-based hashing."""
        sorted_rel, _ = relevance.sort(descending=True)
        ideal_dcg = (sorted_rel / torch.log2(torch.arange(2, sorted_rel.size(0) + 2, dtype=torch.float, device=relevance.device))).sum()
        ranked = scores.argsort(descending=True)
        dcg = (relevance[ranked] / torch.log2(torch.arange(2, ranked.size(0) + 2, dtype=torch.float, device=relevance.device))).sum()
        return 1.0 - dcg / (ideal_dcg + 1e-8)

    def quantization_loss(self, continuous_codes: torch.Tensor) -> torch.Tensor:
        """Penalize deviation from {-1, +1} binary manifold."""
        return torch.mean((continuous_codes.abs() - 1.0).pow(2))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        similarity_matrix: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.1,
    ) -> torch.Tensor:
        ll_loss = self.pairwise_log_likelihood_loss(u, v, similarity_matrix)
        q_loss = self.quantization_loss(u) + self.quantization_loss(v)
        return ll_loss + beta * q_loss
