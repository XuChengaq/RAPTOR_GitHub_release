"""RAPTOR for resource-aware physical-field reconstruction."""
from __future__ import annotations

import torch
from torch import nn

from transolver_tiny.model import PhysicsAttention


class LocalResourceAdapter(nn.Module):
    """Residual adapter with KNN or parameter-matched node-wise context."""

    def __init__(
        self,
        dim: int,
        alpha_init: float = 0.0,
        neighbor_mode: str = "knn_mean",
    ):
        super().__init__()
        if neighbor_mode not in {"knn_mean", "self"}:
            raise ValueError(
                f"neighbor_mode must be 'knn_mean' or 'self', got {neighbor_mode!r}"
            )
        self.neighbor_mode = neighbor_mode
        self.norm = nn.LayerNorm(dim)
        self.message = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_scale = None

    def forward(self, x: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
        if self.neighbor_mode == "knn_mean":
            batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
            neighbors = x[batch, knn].mean(dim=2)
        else:
            # Parameter-matched capacity control: preserve the 2d MLP input
            # while removing all cross-node information.
            neighbors = x
        message = self.message(torch.cat([self.norm(x), neighbors], dim=-1))
        scale = torch.tanh(self.alpha)
        self.last_scale = scale.detach()
        return scale * message


class RAPTORBlock(nn.Module):
    """Exact Transolver block plus an independent local resource adapter."""

    def __init__(self, dim: int, heads: int, slices: int,
                 adapter_enabled: bool = True, adapter_alpha_init: float = 0.0,
                 adapter_neighbor_mode: str = "knn_mean"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.physics_attention = PhysicsAttention(dim, heads, slices)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.local_adapter = (
            LocalResourceAdapter(dim, adapter_alpha_init, adapter_neighbor_mode)
            if adapter_enabled else None
        )

    def forward(self, x: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
        source = x
        x = x + self.physics_attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        if self.local_adapter is None:
            return x
        return x + self.local_adapter(source, knn)


class RAPTOR(nn.Module):
    def __init__(
        self,
        cin: int = 7,
        cout: int = 4,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        slices: int = 64,
        knn_k: int = 12,
        adapter_enabled: bool = True,
        adapter_alpha_init: float = 0.0,
        adapter_neighbor_mode: str = "knn_mean",
    ):
        super().__init__()
        self.slices, self.knn_k = slices, knn_k
        self.adapter_neighbor_mode = adapter_neighbor_mode
        self.input_projection = nn.Linear(cin, dim)
        self.blocks = nn.ModuleList(
            [RAPTORBlock(
                dim,
                heads,
                slices,
                adapter_enabled,
                adapter_alpha_init,
                adapter_neighbor_mode,
            )
             for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, cout)
        )

    def forward(self, x: torch.Tensor, knn: torch.Tensor, groups=None) -> torch.Tensor:
        z = self.input_projection(x)
        for block in self.blocks:
            z = block(z, knn)
        return self.output_projection(z)

    def adapter_scales(self) -> list[float]:
        return [
            float(torch.tanh(block.local_adapter.alpha).detach().cpu())
            for block in self.blocks if block.local_adapter is not None
        ]

    def connection_audit(self, nodes: int) -> dict[str, int | bool | str]:
        knn_connections = (
            nodes * self.knn_k if self.adapter_neighbor_mode == "knn_mean" else 0
        )
        return {
            "nodes": nodes,
            "knn_connections_per_layer": knn_connections,
            "point_to_slice_scores_per_layer": nodes * self.slices,
            "slice_attention_scores_per_layer": self.slices * self.slices,
            "dense_node_connections_per_layer": nodes * nodes,
            "uses_dense_node_attention": False,
            "adapter_neighbor_mode": self.adapter_neighbor_mode,
        }
