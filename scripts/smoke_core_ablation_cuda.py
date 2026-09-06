"""CUDA AMP smoke test for the two RAPTOR ablation variants."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from raptor.model import RAPTOR  # noqa: E402


def run_variant(adapter_enabled: bool, alpha: float) -> None:
    model = RAPTOR(
        dim=64, depth=4, heads=4, slices=64, knn_k=12,
        adapter_enabled=adapter_enabled, adapter_alpha_init=alpha,
    ).cuda().train()
    x = torch.randn(1, 256, 7, device="cuda")
    knn = torch.randint(0, 256, (1, 256, 12), device="cuda")
    groups = torch.arange(256, device="cuda").view(1, -1) % 64
    with torch.amp.autocast("cuda", enabled=True):
        output = model(x, knn, groups)
        loss = output.square().mean()
    loss.backward()
    assert output.shape == (1, 256, 4)
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"PASS adapter_enabled={adapter_enabled} alpha={alpha} loss={loss.item():.6g}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run_variant(False, 0.0)
    run_variant(True, 0.1)
    print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
