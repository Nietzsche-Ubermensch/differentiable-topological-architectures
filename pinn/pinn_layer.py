import torch
import torch.nn as nn
from typing import Callable, List, Optional


class PINNLayer(nn.Module):
    """
    Physics-Informed Neural Network Layer.
    Mesh-free universal function approximator constrained by PDE residuals.
    Uses nested automatic differentiation (torch.autograd.grad with create_graph=True)
    to compute exact spatial/temporal derivatives for PDE enforcement.
    """

    def __init__(self, in_dim: int = 2, hidden_dims: List[int] = [64, 64, 64], out_dim: int = 1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B,1) spatial, t: (B,1) temporal. Both require_grad=True."""
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def compute_derivatives(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """Compute du/dt, du/dx, d2u/dx2 via nested autograd."""
        grad_outputs = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t, u_x, u_xx

    def burgers_residual(self, x: torch.Tensor, t: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
        """Residual of 1D Burgers: u_t + u*u_x - nu*u_xx = 0"""
        u = self.forward(x, t)
        u_t, u_x, u_xx = self.compute_derivatives(u, x, t)
        return u_t + u * u_x - nu * u_xx

    def heat_residual(self, x: torch.Tensor, t: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        """Residual of heat equation: u_t - alpha*u_xx = 0"""
        u = self.forward(x, t)
        u_t, _, u_xx = self.compute_derivatives(u, x, t)
        return u_t - alpha * u_xx


class PINNLoss(nn.Module):
    """
    Multi-objective PINN loss: L = l1*MSE_u + l2*MSE_R + l3*MSE_BC + l4*MSE_IC
    Variance-aware formulation suppresses high localized residuals.
    """

    def __init__(self, l1: float = 1.0, l2: float = 1.0, l3: float = 10.0, l4: float = 10.0):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4 = l1, l2, l3, l4

    def forward(
        self,
        u_pred: torch.Tensor,
        u_obs: torch.Tensor,
        residual: torch.Tensor,
        bc_pred: torch.Tensor,
        bc_true: torch.Tensor,
        ic_pred: torch.Tensor,
        ic_true: torch.Tensor,
    ) -> torch.Tensor:
        mse_u = torch.mean((u_pred - u_obs).pow(2))
        mse_r = torch.mean(residual.pow(2))
        mse_bc = torch.mean((bc_pred - bc_true).pow(2))
        mse_ic = torch.mean((ic_pred - ic_true).pow(2))
        return self.l1 * mse_u + self.l2 * mse_r + self.l3 * mse_bc + self.l4 * mse_ic

    def variance_aware_residual(self, residual: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Heteroscedastic loss: penalizes both residual and predicted variance."""
        return torch.mean(residual.pow(2) * torch.exp(-log_var) + log_var)
