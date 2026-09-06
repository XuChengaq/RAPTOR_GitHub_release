"""Matched Tiny wrapper around the official 2026 MoNo implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "MoNo"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from utils.models.mono import MONO  # noqa: E402


class MoNoTiny(nn.Module):
    def __init__(self, modes=64, dim=64, layers=2):
        super().__init__()
        self.modes = modes
        self.model = MONO(n_mode=modes, n_dim=dim, n_layer=layers,
                          coords_dim=2, condition_dim=7, sol_dim=4,
                          act="GELU", time_proj=False, attn_drop_rate=0.0,
                          out_droprate=0.0, normed_first_stage=True)

    def forward(self, x, knn=None, groups=None):
        # The official mixed-feature scatter requires identical dtypes and is
        # not autocast-safe. Keep this baseline CUDA-only but execute its core
        # in FP32, matching the upstream numerical path.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            return self.model(x[..., :2], x)

    def connection_audit(self, nodes):
        modes = [max(1, self.modes // (2 ** i)) for i in range(4)]
        routing = nodes * modes[0] + sum(a * b for a, b in zip(modes, modes[1:]))
        latent = sum(m * m for m in modes)
        return {"hierarchical_routing_scores": routing,
                "latent_attention_scores": latent,
                "uses_dense_node_attention": False}
