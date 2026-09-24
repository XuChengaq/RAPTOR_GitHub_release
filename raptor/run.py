"""Matched-protocol CUDA runner for RAPTOR."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from data import build_cache  # noqa: E402
from raptor.model import RAPTOR  # noqa: E402
from transolver_tiny.run import evaluate, seed_all, tensors, validation_loss  # noqa: E402


def adapter_variant(cfg):
    if not cfg.get("adapter_enabled", True):
        return "no_local_adapter", "Transolver-Tiny physics-attention backbone without the RAPTOR KNN local adapter"
    neighbor_mode = cfg.get("adapter_neighbor_mode", "knn_mean")
    if neighbor_mode == "self":
        return (
            "nodewise_residual_control",
            "Parameter-matched zero-gated node-wise residual adapter without cross-node aggregation",
        )
    alpha = float(cfg.get("adapter_alpha_init", 0.0))
    if alpha == 0.0:
        return "zero_initialized_adapter", "Transolver-Tiny physics-attention backbone with zero-initialized RAPTOR KNN local adapters"
    return "nonzero_initialized_adapter", f"Transolver-Tiny physics-attention backbone with RAPTOR KNN local adapters initialized at alpha={alpha:g}"


def run_one(cfg, task, seed, data, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    seed_all(seed)
    device = torch.device("cuda")
    model = RAPTOR(
        dim=cfg["hidden_dim"], depth=cfg["depth"], heads=cfg["heads"],
        slices=cfg["slice_count"], knn_k=cfg["knn_k"],
        adapter_enabled=cfg.get("adapter_enabled", True),
        adapter_alpha_init=cfg.get("adapter_alpha_init", 0.0),
        adapter_neighbor_mode=cfg.get("adapter_neighbor_mode", "knn_mean"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    best, stale, history = float("inf"), 0, []
    checkpoint = output / "best.pt"
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for index in np.random.permutation(len(data["train_x"])):
            xt, yt, kt, gt = tensors(data["train_x"][index], data["train_y"][index], data["train_knn"][index], data["train_groups"][index], data, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                loss = torch.mean((model(xt, kt, gt) - yt) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item()
        train_loss = total / len(data["train_x"])
        val_loss = validation_loss(model, data, device, cfg["amp"])
        scales = model.adapter_scales()
        history.append((epoch, train_loss, val_loss, *scales))
        adapter_mean = float(np.mean(np.abs(scales))) if scales else 0.0
        print(f"[raptor {task} s{seed}] epoch={epoch} train={train_loss:.6g} val={val_loss:.6g} adapter_mean={adapter_mean:.5f}", flush=True)
        if val_loss < best - 1e-6:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    training_peak = torch.cuda.max_memory_allocated(device) / 2**20
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics = evaluate(model, data, device, cfg["amp"])
    metrics["peak_memory_mib"] = max(training_peak, metrics["peak_memory_mib"])
    variant, method = adapter_variant(cfg)
    metrics.update(
        model="raptor", task=task, seed=seed, epochs=epoch,
        best_validation_mse=best, train_seconds=time.time() - started,
        parameters=sum(p.numel() for p in model.parameters()), device="cuda",
        hidden_dim=cfg["hidden_dim"], depth=cfg["depth"], heads=cfg["heads"],
        slice_count=cfg["slice_count"], knn_k=cfg["knn_k"],
        adapter_scales=model.adapter_scales(),
        adapter_enabled=cfg.get("adapter_enabled", True),
        adapter_alpha_init=cfg.get("adapter_alpha_init", 0.0),
        adapter_neighbor_mode=cfg.get("adapter_neighbor_mode", "knn_mean"),
        ablation_variant=variant,
        connection_audit=model.connection_audit(cfg["points_per_simulation"]),
        protocol="matched_airfrans_formal_v1",
        method=method,
    )
    (output / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse_normalized", "validation_mse_normalized", *[f"adapter_scale_{i}" for i in range(cfg["depth"])]] )
        writer.writerows(history)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/smoke.yaml")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / args.config).read_text(encoding="utf-8"))
    tasks = [args.task] if args.task else cfg["tasks"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    results = []
    for task in tasks:
        cache = build_cache(PROJECT / cfg["data_root"], task, cfg["train_simulations"], cfg["test_simulations"], cfg["points_per_simulation"], cfg["surface_fraction"], cfg["token_count"], cfg["knn_k"], PROJECT / cfg["cache_dir"], seed=cfg["data_seed"], val_fraction=cfg["validation_fraction"])
        data = np.load(cache)
        for seed in seeds:
            output = PROJECT / cfg["output_dir"] / f"{task}-raptor-s{seed}"
            output.mkdir(parents=True, exist_ok=True)
            result = output / "result.json"
            if result.exists():
                results.append(json.loads(result.read_text(encoding="utf-8")))
                print("SKIP", output.name, flush=True)
            else:
                results.append(run_one(cfg, task, seed, data, output))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
