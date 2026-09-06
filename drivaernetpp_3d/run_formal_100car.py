from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from raptor.model import RAPTOR
from transolver_tiny.model import TransolverTiny
from run_small_3d import batch, mse, prepare_cache


class PointNet3D(nn.Module):
    """PointNet baseline: shared point encoder, global max context, point decoder."""

    def __init__(self, cin: int = 3, cout: int = 1, dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cin, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, cout),
        )

    def forward(self, x: torch.Tensor, knn=None, groups=None) -> torch.Tensor:
        local = self.encoder(x)
        global_context = local.max(dim=1, keepdim=True).values.expand_as(local)
        return self.decoder(torch.cat([local, global_context], dim=-1))


def make_model(name: str, k: int) -> nn.Module:
    if name == "pointnet":
        return PointNet3D(cin=3, cout=1, dim=64)
    if name == "transolver_tiny":
        return TransolverTiny(cin=3, cout=1, dim=64, depth=4, heads=4, slices=64)
    if name == "raptor":
        return RAPTOR(cin=3, cout=1, dim=64, depth=4, heads=4, slices=64,
                      knn_k=k, adapter_enabled=True, adapter_alpha_init=0.0)
    if name == "no_adapter":
        return RAPTOR(cin=3, cout=1, dim=64, depth=4, heads=4, slices=64,
                      knn_k=k, adapter_enabled=False, adapter_alpha_init=0.0)
    raise ValueError(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_normalized_mse", "validation_normalized_mse",
                         "best_validation_normalized_mse"])
        writer.writerows(rows)


def train_one(name: str, data, output: Path, args, device: torch.device) -> None:
    result_file = output / "result.json"
    if result_file.exists():
        print(f"[skip] {name}: result.json exists", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    model = make_model(name, args.knn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    train_end, val_end = 80, 90
    history: list[tuple] = []
    best, stale, start_epoch = float("inf"), 0, 1
    best_file, last_file = output / "best.pt", output / "last.pt"

    if last_file.exists():
        state = torch.load(last_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best, stale = float(state["best"]), int(state["stale"])
        history = [tuple(row) for row in state["history"]]
        start_epoch = int(state["epoch"]) + 1
        print(f"[resume] {name} epoch={start_epoch} best={best:.6g}", flush=True)

    x0, y0, k0 = batch(data, 0, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        smoke_pred = model(x0, k0)
        smoke_loss = torch.mean((smoke_pred - y0) ** 2)
    scaler.scale(smoke_loss).backward()
    optimizer.zero_grad(set_to_none=True)
    (output / "cuda_smoke.json").write_text(json.dumps({
        "passed": True, "model": name, "device": torch.cuda.get_device_name(0),
        "loss": float(smoke_loss.detach()), "output_shape": list(smoke_pred.shape),
    }, indent=2), encoding="utf-8")

    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for i in np.random.permutation(train_end):
            x, y, knn = batch(data, int(i), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = torch.mean((model(x, knn) - y) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        val_loss = mse(model, data, range(train_end, val_end), device, True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), best_file)
        else:
            stale += 1
        history.append((epoch, train_loss, val_loss, best))
        completed_epoch = epoch
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "stale": stale, "history": history}, last_file)
        save_history(output / "history.csv", history)
        print(f"[formal100] model={name} epoch={epoch} train={train_loss:.6g} "
              f"val={val_loss:.6g} best={best:.6g} stale={stale}", flush=True)
        if stale >= args.patience:
            print(f"[early-stop] {name} epoch={epoch}", flush=True)
            break

    model.load_state_dict(torch.load(best_file, map_location=device, weights_only=True))
    test_loss = mse(model, data, range(val_end, 100), device, True)
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(val_end, 100):
            x, _, knn = batch(data, i, device)
            with torch.amp.autocast("cuda"):
                predictions.append(model(x, knn).float().cpu().numpy()[0])
    np.savez_compressed(output / "test_predictions.npz", x=data["x"][val_end:100],
                        y_true=data["y"][val_end:100], y_pred=np.stack(predictions),
                        y_mean=data["y_mean"], y_std=data["y_std"])
    result = {
        "status": "complete", "model": name,
        "scope": "100-vehicle single-channel surface-field reconstruction",
        "seed": args.seed, "samples": 100, "train_samples": 80,
        "validation_samples": 10, "test_samples": 10,
        "points_per_vehicle": args.points, "knn_k": args.knn,
        "epochs_completed": completed_epoch, "best_validation_normalized_mse": best,
        "test_normalized_mse": test_loss, "train_seconds_this_invocation": time.time() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(0),
        "protocol": {"optimizer": "AdamW", "lr": args.lr,
                     "weight_decay": args.weight_decay, "amp": True,
                     "gradient_clip": 1.0, "patience": args.patience},
        "note": ("no_adapter uses the RAPTOR wrapper with its local adapter disabled; "
                 "it audits the exact Transolver-style backbone independently.") if name == "no_adapter" else "",
    }
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "formal_100car")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--models", nargs="+", default=["pointnet", "transolver_tiny", "raptor", "no_adapter"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    cache = prepare_cache(args.archive, HERE / "cache", 100, args.points, args.knn, args.seed)
    data = np.load(cache)
    assert data["x"].shape == (100, args.points, 3)
    assert data["y"].shape == (100, args.points, 1)
    assert data["knn"].shape == (100, args.points, args.knn)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)
    for name in args.models:
        train_one(name, data, args.output / name, args, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
