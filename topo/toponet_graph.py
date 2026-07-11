"""Graph-based topology reasoning for driving scenes — TopoNet interface."""
import torch
import torch.nn as nn


class TopologyReasoningGNN(nn.Module):
    """
    Graph neural network for road topology reasoning.
    Reference: OpenDriveLab/TopoNet (SCIENCE CHINA 2026)
    Jointly models lane centerlines and traffic element relationships.
    """

    def __init__(self, node_dim: int = 256, edge_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, node_dim)
        self.edge_enc = nn.Linear(edge_dim, node_dim)
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, node_dim),
                nn.LayerNorm(node_dim),
                nn.GELU()
            ) for _ in range(n_layers)
        ])
        self.centerline_head = nn.Linear(node_dim, 2)   # (x, y) waypoints
        self.topology_head = nn.Linear(node_dim * 2, 1) # Edge existence logit

    def forward(
        self,
        node_feats: torch.Tensor,       # (N, node_dim)
        edge_index: torch.Tensor,       # (2, E)
        edge_feats: torch.Tensor,       # (E, edge_dim)
    ):
        h = torch.relu(self.node_enc(node_feats))
        e = torch.relu(self.edge_enc(edge_feats))
        src, dst = edge_index
        for layer in self.gnn_layers:
            msg = torch.cat([h[src], e], dim=-1)
            agg = torch.zeros_like(h).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg[:, :h.size(-1)]), msg[:, :h.size(-1)])
            h = layer(torch.cat([h, agg], dim=-1))
        centerlines = self.centerline_head(h)
        edge_logits = self.topology_head(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        return centerlines, edge_logits
