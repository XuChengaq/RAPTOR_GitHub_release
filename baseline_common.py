"""Shared CUDA-only training and evaluation for matched AirfRANS baselines."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.sqrt(np.sum(np.square(pred - target), dtype=np.float64))
    den = np.sqrt(np.sum(np.square(target), dtype=np.float64))
    return float(num / (den + 1e-12))


def _tensors(x, y, k, g, data, device):
    xn = ((x - data["xmean"]) / data["xstd"]).astype("float32")
    yn = ((y - data["ymean"]) / data["ystd"]).astype("float32")
    return (
        torch.from_numpy(xn)[None].to(device),
        torch.from_numpy(yn)[None].to(device),
        torch.from_numpy(k)[None].to(device),
        torch.from_numpy(g)[None].to(device),
    )


def _validation_loss(model, data, device, amp):
    model.eval()
    total = 0.0
    with torch.no_grad():
        rows = zip(data["val_x"], data["val_y"], data["val_knn"], data["val_groups"])
        for x, y, k, g in rows:
            xt, yt, kt, gt = _tensors(x, y, k, g, data, device)
            with torch.amp.autocast("cuda", enabled=amp):
                total += float(torch.mean((model(xt, kt, gt) - yt) ** 2))
    return total / len(data["val_x"])


def _evaluate(model, data, device, amp):
    model.eval()
    preds, elapsed = [], []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for x, k, g in zip(data["test_x"], data["test_knn"], data["test_groups"]):
            dummy = np.zeros((len(x), 4), dtype="float32")
            xt, _, kt, gt = _tensors(x, dummy, k, g, data, device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(xt, kt, gt)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            preds.append((out.float().cpu().numpy()[0] * data["ystd"] + data["ymean"]).astype("float32"))
    pred, target = np.stack(preds), data["test_y"]
    names = ("ux", "uy", "p", "nut")
    metrics = {f"rel_l2_{name}": _rel_l2(pred[..., i], target[..., i]) for i, name in enumerate(names)}
    metrics.update({f"mse_{name}": float(np.mean((pred[..., i] - target[..., i]) ** 2)) for i, name in enumerate(names)})
    surface = data["test_surface"] > 0.5
    metrics["macro_rel_l2"] = float(np.mean([metrics[f"rel_l2_{name}"] for name in names]))
    metrics["surface_pressure_mse"] = float(np.mean((pred[..., 2][surface] - target[..., 2][surface]) ** 2))
    metrics["inference_ms_per_sim"] = 1000.0 * float(np.mean(elapsed))
    metrics["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return metrics


def train_and_evaluate(cfg, task, seed, data, output: Path, model_name, model_factory, metadata):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    seed_all(seed)
    device = torch.device("cuda")
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    best, stale, history = float("inf"), 0, []
    checkpoint = output / "best.pt"
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        order = np.random.permutation(len(data["train_x"]))
        for index in order:
            xt, yt, kt, gt = _tensors(data["train_x"][index], data["train_y"][index], data["train_knn"][index], data["train_groups"][index], data, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                loss = torch.mean((model(xt, kt, gt) - yt) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item()
        train_loss = total / len(order)
        val_loss = _validation_loss(model, data, device, cfg["amp"])
        history.append((epoch, train_loss, val_loss))
        print(f"[{model_name} {task} s{seed}] epoch={epoch} train={train_loss:.6g} val={val_loss:.6g}", flush=True)
        if val_loss < best - 1e-6:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    training_peak = torch.cuda.max_memory_allocated(device) / 2**20
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics = _evaluate(model, data, device, cfg["amp"])
    metrics["peak_memory_mib"] = max(training_peak, metrics["peak_memory_mib"])
    metrics.update(model=model_name, task=task, seed=seed, epochs=epoch,
                   best_validation_mse=best, train_seconds=time.time() - started,
                   parameters=sum(p.numel() for p in model.parameters()), device="cuda",
                   protocol="matched_airfrans_reynolds_v2", **metadata)
    (output / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse_normalized", "validation_mse_normalized"])
        writer.writerows(history)
    return metrics
