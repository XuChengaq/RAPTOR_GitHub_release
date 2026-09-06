"""Single-GPU Tiny adaptation of the official Transolver++ eidetic slicer."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EideticPhysicsAttention(nn.Module):
    def __init__(self, dim=64, heads=4, slices=64):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim, self.slices = heads, dim // heads, slices
        self.bias = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.in_project = nn.Linear(dim, dim)
        self.project_slice = nn.Linear(self.head_dim, slices)
        nn.init.orthogonal_(self.project_slice.weight)
        self.project_temperature = nn.Sequential(
            nn.Linear(self.head_dim, slices), nn.GELU(), nn.Linear(slices, 1), nn.GELU()
        )
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch, nodes, _ = x.shape
        input_dtype = x.dtype

        # Gumbel sampling and low-temperature softmax are numerically unsafe in
        # FP16: rounded random values can produce non-finite logarithms, causing
        # GradScaler to skip every optimizer step.  Keep the eidetic slicing
        # path in FP32 while allowing the surrounding network to use AMP.
        with torch.amp.autocast(x.device.type, enabled=False):
            point = self.in_project(x.float()).view(
                batch, nodes, self.heads, self.head_dim
            ).permute(0, 2, 1, 3)
            logits = self.project_slice(point)
            temperature = torch.clamp(
                self.project_temperature(point) + self.bias, min=0.01
            )
            if self.training:
                eps = torch.finfo(logits.dtype).eps
                uniform = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
                noise = -torch.log(-torch.log(uniform))
                weights = F.softmax((logits + noise) / temperature, dim=-1)
            else:
                weights = F.softmax(logits / temperature, dim=-1)
            norm = weights.sum(dim=2).clamp_min(1e-5)
            tokens = torch.einsum("bhnc,bhns->bhsc", point, weights) / norm[..., None]
            attended = F.scaled_dot_product_attention(
                self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
            )
            restored = torch.einsum("bhsc,bhns->bhnc", attended, weights)
            output = self.to_out(
                restored.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
            )
        return output.to(input_dtype)

    def connection_audit(self, nodes):
        return {"point_to_slice_scores_per_layer": nodes * self.slices,
                "slice_attention_scores_per_layer": self.slices * self.slices,
                "uses_dense_node_attention": False}


class Block(nn.Module):
    def __init__(self, dim, heads, slices):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EideticPhysicsAttention(dim, heads, slices)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TransolverPlusTiny(nn.Module):
    def __init__(self, dim=64, depth=4, heads=4, slices=64, input_dim=7, output_dim=4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, slices) for _ in range(depth)])
        self.output_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))
        self.slices = slices

    def forward(self, x, knn=None, groups=None):
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x)

    def connection_audit(self, nodes):
        return self.blocks[0].attn.connection_audit(nodes)
