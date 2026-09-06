"""Resource-matched Transolver-Tiny for the existing AirfRANS protocol.

The Physics-Attention implementation follows the official Transolver design:
soft point-to-slice assignment, attention among physics tokens, and deslicing.
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicsAttention(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4, slices: int = 64):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if slices < 1:
            raise ValueError("slices must be positive")
        self.heads = heads
        self.slices = slices
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.project_x = nn.Linear(dim, dim)
        self.project_fx = nn.Linear(dim, dim)
        self.project_slice = nn.Linear(self.head_dim, slices)
        nn.init.orthogonal_(self.project_slice.weight)
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.last_slice_shape = None
        self.last_attention_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x_mid = self.project_x(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        fx_mid = self.project_fx(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        temperature = self.temperature.clamp(0.1, 5.0)
        weights = torch.softmax(self.project_slice(x_mid) / temperature, dim=-1)
        normalizer = weights.sum(dim=2).clamp_min(1e-5)
        tokens = torch.einsum("bhnd,bhns->bhsd", fx_mid, weights)
        tokens = tokens / normalizer.unsqueeze(-1)
        q, k, v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        token_out = attention @ v
        point_out = torch.einsum("bhsd,bhns->bhnd", token_out, weights)
        point_out = point_out.permute(0, 2, 1, 3).reshape(b, n, d)
        self.last_slice_shape = tuple(weights.shape)
        self.last_attention_shape = tuple(attention.shape)
        return self.to_out(point_out)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, slices: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.physics_attention = PhysicsAttention(dim, heads, slices)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.physics_attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TransolverTiny(nn.Module):
    def __init__(
        self,
        cin: int = 7,
        cout: int = 4,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        slices: int = 64,
    ):
        super().__init__()
        self.input_projection = nn.Linear(cin, dim)
        self.blocks = nn.ModuleList(
            [TransolverBlock(dim, heads, slices) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, cout)
        )

    def forward(self, x: torch.Tensor, knn=None, groups=None) -> torch.Tensor:
        z = self.input_projection(x)
        for block in self.blocks:
            z = block(z)
        return self.output_projection(z)

