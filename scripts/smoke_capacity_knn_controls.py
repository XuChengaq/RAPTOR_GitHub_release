"""Dependency-light smoke checks for capacity-control experiment variants."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from raptor.model import RAPTOR


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 37, 7)
    knn = torch.randint(0, 37, (2, 37, 6))
    nodewise = RAPTOR(
        dim=32,
        depth=1,
        heads=4,
        slices=8,
        knn_k=6,
        adapter_alpha_init=0.3,
        adapter_neighbor_mode="self",
    ).eval()
    local = RAPTOR(
        dim=32,
        depth=1,
        heads=4,
        slices=8,
        knn_k=6,
        adapter_alpha_init=0.3,
        adapter_neighbor_mode="knn_mean",
    ).eval()
    assert sum(p.numel() for p in nodewise.parameters()) == sum(
        p.numel() for p in local.parameters()
    )
    with torch.no_grad():
        expected = nodewise(x, knn)
        actual = nodewise(x, torch.flip(knn, dims=(-1,)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert nodewise.connection_audit(37)["knn_connections_per_layer"] == 0

    try:
        RAPTOR(adapter_neighbor_mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid neighbor mode was not rejected")

    checkpoint = (
        PROJECT
        / "outputs"
        / "formal"
        / "reynolds-raptor-s42"
        / "best.pt"
    )
    if checkpoint.exists():
        preserved = RAPTOR(dim=64, depth=4, heads=4, slices=64, knn_k=12)
        preserved.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
    print("CAPACITY_KNN_CONTROL_SMOKE_OK")


if __name__ == "__main__":
    main()
