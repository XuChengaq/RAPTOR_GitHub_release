"""Small CUDA AMP smoke test for LinearNO-Tiny."""
from __future__ import annotations

import torch
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from linearno_tiny import LinearNOTiny


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    model = LinearNOTiny().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x = torch.randn(1, 128, 7, device="cuda")
    target = torch.randn(1, 128, 4, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss = (model(x) - target).square().mean()
    scaler.scale(loss).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("LinearNO-Tiny CUDA AMP gradients are not finite")
    scaler.step(optimizer)
    scaler.update()
    print(f"LINEARNO_CUDA_SMOKE_OK loss={float(loss):.6g}", flush=True)


if __name__ == "__main__":
    main()
