"""Diagnose whether AMP causes non-finite Transolver++-Tiny gradients."""
from pathlib import Path
import sys

import numpy as np
import torch

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))

from baseline_common import _tensors
from transolver_plus_tiny import TransolverPlusTiny


cache_files = list((project / "cache").glob("*reynolds*.npz"))
if not cache_files:
    raise FileNotFoundError("Reynolds cache not found")

data = np.load(cache_files[0])
device = torch.device("cuda")
model = TransolverPlusTiny(dim=64, depth=4, heads=4, slices=64).to(device)
x, y, knn, groups = _tensors(
    data["train_x"][0], data["train_y"][0], data["train_knn"][0],
    data["train_groups"][0], data, device,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda", enabled=True)
optimizer.zero_grad(set_to_none=True)
before = model.output_projection[-1].weight.detach().clone()
with torch.amp.autocast("cuda", enabled=True):
    output = model(x, knn, groups)
    loss = torch.mean((output - y) ** 2)
scaler.scale(loss).backward()
nonfinite = [
    name for name, parameter in model.named_parameters()
    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
]
scaler.step(optimizer)
scaler.update()
delta = (model.output_projection[-1].weight.detach() - before).abs().max().item()
print({
    "loss": loss.item(),
    "output_finite": torch.isfinite(output).all().item(),
    "nonfinite_gradients": nonfinite,
    "output_weight_delta": delta,
    "scaler": scaler.get_scale(),
})
