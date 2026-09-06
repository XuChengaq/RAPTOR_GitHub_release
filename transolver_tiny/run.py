"""Independent runner for the Transolver-Tiny AirfRANS baseline."""
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
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from data import build_cache  # noqa: E402
from transolver_tiny.model import TransolverTiny  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rel_l2(pred, target) -> float:
    # Avoid a late BLAS/OpenMP initialization conflict between NumPy and
    # PyTorch on Windows; elementwise reductions are sufficient for L2 here.
    numerator = np.sqrt(np.sum(np.square(pred - target), dtype=np.float64))
    denominator = np.sqrt(np.sum(np.square(target), dtype=np.float64))
    return float(numerator / (denominator + 1e-12))


def tensors(x, y, k, g, data, device):
    xn = ((x - data["xmean"]) / data["xstd"]).astype("float32")
    yn = ((y - data["ymean"]) / data["ystd"]).astype("float32")
    return (
        torch.from_numpy(xn)[None].to(device),
        torch.from_numpy(yn)[None].to(device),
        torch.from_numpy(k)[None].to(device),
        torch.from_numpy(g)[None].to(device),
    )


def validation_loss(model, data, device, amp):
    model.eval()
    total = 0.0
    with torch.no_grad():
        rows = zip(data["val_x"], data["val_y"], data["val_knn"], data["val_groups"])
        for x, y, k, g in rows:
            xt, yt, kt, gt = tensors(x, y, k, g, data, device)
            with torch.amp.autocast("cuda", enabled=amp):
                total += float(torch.mean((model(xt, kt, gt) - yt) ** 2))
    return total / len(data["val_x"])


def evaluate(model, data, device, amp):
    model.eval()
    preds, elapsed = [], []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        rows = zip(data["test_x"], data["test_knn"], data["test_groups"])
        for x, k, g in rows:
            dummy_y = np.zeros((len(x), 4), dtype="float32")
            xt, _, kt, gt = tensors(x, dummy_y, k, g, data, device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(xt, kt, gt)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            preds.append((pred.cpu().numpy()[0] * data["ystd"] + data["ymean"]).astype("float32"))
    pred, target = np.stack(preds), data["test_y"]
    surface = data["test_surface"] > 0.5
    names = ["ux", "uy", "p", "nut"]
    metrics = {f"rel_l2_{name}": rel_l2(pred[..., i], target[..., i]) for i, name in enumerate(names)}
    metrics.update({f"mse_{name}": float(np.mean((pred[..., i] - target[..., i]) ** 2)) for i, name in enumerate(names)})
    metrics["macro_rel_l2"] = float(np.mean([metrics[f"rel_l2_{name}"] for name in names]))
    metrics["surface_pressure_mse"] = float(np.mean((pred[..., 2][surface] - target[..., 2][surface]) ** 2))
    metrics["inference_ms_per_sim"] = 1000 * float(np.mean(elapsed))
    metrics["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return metrics


def run_one(cfg, task, seed, data, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden for this baseline")
    seed_all(seed)
    device = torch.device("cuda")
    model = TransolverTiny(dim=cfg["hidden_dim"], depth=cfg["depth"], heads=cfg["heads"], slices=cfg["slice_count"]).to(device)
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
        train_loss = total / len(order)
        val_loss = validation_loss(model, data, device, cfg["amp"])
        history.append((epoch, train_loss, val_loss))
        print(f"[transolver_tiny {task} s{seed}] epoch={epoch} train={train_loss:.6g} val={val_loss:.6g}", flush=True)
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
    metrics.update(model="transolver_tiny", task=task, seed=seed, epochs=epoch, best_validation_mse=best, train_seconds=time.time() - started, parameters=sum(p.numel() for p in model.parameters()), device="cuda", hidden_dim=cfg["hidden_dim"], depth=cfg["depth"], heads=cfg["heads"], slice_count=cfg["slice_count"], protocol="matched_airfrans_formal_v1")
    (output / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse_normalized", "validation_mse_normalized"])
        writer.writerows(history)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--completed-epochs", type=int)
    parser.add_argument("--best-validation", type=float)
    parser.add_argument("--train-seconds", type=float)
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / args.config).read_text(encoding="utf-8"))
    tasks = [args.task] if args.task else cfg["tasks"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    results = []
    for task in tasks:
        cache = build_cache(PROJECT / cfg["data_root"], task, cfg["train_simulations"], cfg["test_simulations"], cfg["points_per_simulation"], cfg["surface_fraction"], cfg["token_count"], cfg["knn_k"], PROJECT / cfg["cache_dir"], seed=cfg["data_seed"], val_fraction=cfg["validation_fraction"])
        data = np.load(cache)
        for seed in seeds:
            output = PROJECT / cfg["output_dir"] / f"{task}-transolver_tiny-s{seed}"
            output.mkdir(parents=True, exist_ok=True)
            result = output / "result.json"
            if result.exists():
                results.append(json.loads(result.read_text(encoding="utf-8")))
                print("SKIP", output.name, flush=True)
            elif args.evaluate_only:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required; CPU fallback is forbidden")
                checkpoint = output / "best.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                model = TransolverTiny(dim=cfg["hidden_dim"], depth=cfg["depth"], heads=cfg["heads"], slices=cfg["slice_count"]).cuda()
                model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
                metrics = evaluate(model, data, torch.device("cuda"), cfg["amp"])
                metrics.update(model="transolver_tiny", task=task, seed=seed,
                               epochs=args.completed_epochs,
                               best_validation_mse=args.best_validation,
                               train_seconds=args.train_seconds,
                               parameters=sum(p.numel() for p in model.parameters()),
                               device="cuda", hidden_dim=cfg["hidden_dim"],
                               depth=cfg["depth"], heads=cfg["heads"],
                               slice_count=cfg["slice_count"],
                               protocol="matched_airfrans_formal_v1",
                               recovered_evaluation=True)
                result.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                results.append(metrics)
            else:
                results.append(run_one(cfg, task, seed, data, output))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
