"""Tiny LinearNO with the paper's asymmetric canonical linear attention.

Hu et al. generalize Physics-Attention by learning independent query and key
maps, normalize them along the latent-state and node axes respectively, and
remove slice self-attention.  This module preserves those operations while
matching the width/depth used by the local AirfRANS protocol.
"""
from __future__ import annotations

import torch
from torch import nn


class LinearAttentionNO(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4, states: int = 64):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if states < 1:
            raise ValueError("states must be positive")
        self.heads = heads
        self.head_dim = dim // heads
        self.states = states
        # Independent projections implement LinearNO's generalization step.
        self.to_query = nn.Linear(self.head_dim, states)
        self.to_key = nn.Linear(self.head_dim, states)
        self.to_value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.last_query_map_shape = None
        self.last_key_map_shape = None
        self.last_query_sums = None
        self.last_key_sums = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, dim = x.shape
        features = x.reshape(batch, nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # phi(Q): each node distributes unit mass over M latent states.
        query_map = torch.softmax(self.to_query(features), dim=-1)
        # psi(K): each latent state distributes unit mass over N nodes.
        key_map = torch.softmax(self.to_key(features), dim=-2)
        values = self.to_value(features)
        latent_values = torch.einsum("bhnm,bhnd->bhmd", key_map, values)
        output = torch.einsum("bhnm,bhmd->bhnd", query_map, latent_values)
        output = output.permute(0, 2, 1, 3).reshape(batch, nodes, dim)
        self.last_query_map_shape = tuple(query_map.shape)
        self.last_key_map_shape = tuple(key_map.shape)
        self.last_query_sums = query_map.detach().sum(dim=-1)
        self.last_key_sums = key_map.detach().sum(dim=-2)
        return self.to_out(output)

    def connection_audit(self, nodes: int) -> dict:
        return {
            "linear_attention_scores_per_layer": nodes * self.states,
            "uses_slice_self_attention": False,
            "uses_dense_node_attention": False,
        }


class LinearNOBlock(nn.Module):
    def __init__(self, dim: int, heads: int, states: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = LinearAttentionNO(dim, heads, states)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class LinearNOTiny(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 4,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        states: int = 64,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList(
            [LinearNOBlock(dim, heads, states) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, output_dim)
        )
        self.states = states

    def forward(self, x: torch.Tensor, knn=None, groups=None) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x)

    def connection_audit(self, nodes: int) -> dict:
        return self.blocks[0].attention.connection_audit(nodes)

