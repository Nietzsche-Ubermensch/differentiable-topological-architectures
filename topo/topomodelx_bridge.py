"""Bridge to TopoModelX: simplicial and cell-complex neural network interface."""
import torch
import torch.nn as nn


class SimplicialMessagePassing(nn.Module):
    """
    Simplified simplicial message passing layer.
    Reference: pyt-team/TopoModelX
    Operates on node, edge, and triangle features simultaneously.
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.update = nn.GRUCell(out_dim, out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        B1: torch.Tensor,   # Boundary matrix: edges -> nodes
    ) -> torch.Tensor:
        """B1: sparse (n_nodes, n_edges) boundary matrix."""
        node_msg = self.node_proj(node_feats)
        edge_msg = self.edge_proj(edge_feats)
        # Aggregate edge messages to nodes via boundary matrix
        aggregated = B1 @ edge_msg                          # (n_nodes, out_dim)
        updated = self.update(aggregated, node_msg)
        return updated


class CellComplexNetwork(nn.Module):
    """
    Cell complex neural network: operates on 0-, 1-, 2-cells
    (nodes, edges, faces) with higher-order message passing.
    """

    def __init__(self, dims: list, out_dim: int, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            SimplicialMessagePassing(dims[0], dims[1], out_dim)
            for _ in range(n_layers)
        ])
        self.readout = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats, edge_feats, B1):
        x = node_feats
        for layer in self.layers:
            x = torch.relu(layer(x, edge_feats, B1))
        return self.readout(x.mean(dim=0, keepdim=True))
