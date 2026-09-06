from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "drivaernetpp_3d"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(HERE.parent))

from audit_dataset import SliceReader, nested_data_offset
from raptor.model import RAPTOR
from transolver_tiny.model import TransolverTiny
from run_small_3d import batch, knn_indices, mse

FAMILIES = ("J20", "P180", "boeing737", "erj")


def open_inner(archive: Path):
    outer = zipfile.ZipFile(archive)
    nested = next(x for x in outer.infolist() if x.filename.lower().endswith(".zip"))
    if nested.compress_type != zipfile.ZIP_STORED:
        outer.close()
        raise RuntimeError("nested ZIP must be stored for direct access")
    start = nested_data_offset(archive, nested)
    outer.close()
    raw = SliceReader(archive, start, nested.file_size)
    buffered = io.BufferedReader(raw)
    return raw, buffered, zipfile.ZipFile(buffered)


def stratified_names(inner: zipfile.ZipFile, seed: int) -> tuple[list[str], dict]:
    by_family = {family: [] for family in FAMILIES}
    for name in inner.namelist():
        if not name.lower().endswith(".npz"):
            continue
        parts = name.split("/")
        if len(parts) >= 3 and parts[0].lower() == "plane" and parts[1] in by_family:
            by_family[parts[1]].append(name)
    rng = random.Random(seed)
    for family in FAMILIES:
        by_family[family].sort()
        rng.shuffle(by_family[family])
        if len(by_family[family]) < 25:
            raise RuntimeError(f"not enough samples for {family}: {len(by_family[family])}")

    # Exact 80/10/10 while retaining near-identical family balance.
    train, validation, test = [], [], []
    allocation = {
        "J20": (20, 3, 2), "P180": (20, 3, 2),
        "boeing737": (20, 2, 3), "erj": (20, 2, 3),
    }
    for family in FAMILIES:
        names = by_family[family][:25]
        nt, nv, _ = allocation[family]
        train.extend(names[:nt])
        validation.extend(names[nt:nt + nv])
        test.extend(names[nt + nv:])
    return train + validation + test, {
        "families": {family: 25 for family in FAMILIES},
        "allocation": {k: {"train": v[0], "validation": v[1], "test": v[2]}
                       for k, v in allocation.items()},
        "train": train, "validation": validation, "test": test,
    }


def prepare_cache(archive: Path, output: Path, points: int, k: int, seed: int) -> Path:
    cache = output / f"aircraft100_n{points}_k{k}_seed{seed}.npz"
    manifest_file = output / "split_manifest.json"
    if cache.exists() and manifest_file.exists():
        return cache
    output.mkdir(parents=True, exist_ok=True)
    raw, buffered, inner = open_inner(archive)
    xs, ys, knns = [], [], []
    try:
        names, manifest = stratified_names(inner, seed)
        for index, name in enumerate(names):
            with inner.open(name) as handle:
                payload = np.load(io.BytesIO(handle.read()), allow_pickle=False)
                nodes = np.asarray(payload["nodes_list"], dtype=np.float32)
                target = np.asarray(payload["features_list"], dtype=np.float32).reshape(-1, 1)
            if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) != len(target):
                raise ValueError(f"invalid arrays in {name}: nodes={nodes.shape}, target={target.shape}")
            if len(nodes) < points:
                raise ValueError(f"sample has only {len(nodes)} nodes: {name}")
            rng = np.random.default_rng(seed + index)
            chosen = np.sort(rng.choice(len(nodes), size=points, replace=False))
            x, y = nodes[chosen], target[chosen]
            xs.append(x)
            ys.append(y)
            knns.append(knn_indices(x, k))
            print(f"[cache-aircraft] {index + 1}/100 {name}", flush=True)
    finally:
        inner.close(); buffered.close(); raw.close()

    train_y = np.stack(ys[:80])
    y_mean = train_y.mean(axis=(0, 1), keepdims=True)
    y_std = train_y.std(axis=(0, 1), keepdims=True) + 1e-6
    np.savez_compressed(
        cache, x=np.stack(xs).astype(np.float32),
        y=((np.stack(ys) - y_mean) / y_std).astype(np.float32),
        knn=np.stack(knns).astype(np.int64), y_mean=y_mean, y_std=y_std,
    )
    manifest.update({
        "seed": seed, "samples": 100, "points_per_aircraft": points,
        "target": "single-channel aircraft surface node field (features_list)",
        "field_provenance": "semantic physical quantity not asserted by this experiment",
    })
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache


def make_model(name: str, k: int):
    if name == "transolver_tiny":
        return TransolverTiny(cin=3, cout=1, dim=64, depth=4, heads=4, slices=64)
    if name == "raptor":
        return RAPTOR(cin=3, cout=1, dim=64, depth=4, heads=4, slices=64,
                      knn_k=k, adapter_enabled=True, adapter_alpha_init=0.0)
    raise ValueError(name)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_history(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_normalized_mse", "validation_normalized_mse",
                         "best_validation_normalized_mse"])
        writer.writerows(rows)


def train_one(name: str, data, output: Path, args, device):
    result_file = output / "result.json"
    if result_file.exists():
        print(f"[skip] {name}: result.json exists", flush=True); return
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    model = make_model(name, args.knn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    history, best, stale, start_epoch = [], float("inf"), 0, 1
    best_file, last_file = output / "best.pt", output / "last.pt"
    if last_file.exists():
        state = torch.load(last_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best, stale, history = float(state["best"]), int(state["stale"]), state["history"]
        start_epoch = int(state["epoch"]) + 1
        print(f"[resume] {name} epoch={start_epoch} best={best:.6g}", flush=True)

    x0, y0, k0 = batch(data, 0, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        smoke_pred = model(x0, k0)
        smoke_loss = torch.mean((smoke_pred - y0) ** 2)
    scaler.scale(smoke_loss).backward(); optimizer.zero_grad(set_to_none=True)
    (output / "cuda_smoke.json").write_text(json.dumps({
        "passed": True, "model": name, "device": torch.cuda.get_device_name(0),
        "loss": float(smoke_loss.detach()), "output_shape": list(smoke_pred.shape),
    }, indent=2), encoding="utf-8")

    started = time.time(); torch.cuda.reset_peak_memory_stats(device); completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); losses = []
        for i in np.random.permutation(80):
            x, y, knn = batch(data, int(i), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = torch.mean((model(x, knn) - y) ** 2)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses)); val_loss = mse(model, data, range(80, 90), device, True)
        if val_loss < best:
            best, stale = val_loss, 0; torch.save(model.state_dict(), best_file)
        else:
            stale += 1
        history.append((epoch, train_loss, val_loss, best)); completed_epoch = epoch
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "stale": stale,
                    "history": history}, last_file)
        save_history(output / "history.csv", history)
        print(f"[aircraft100] model={name} epoch={epoch} train={train_loss:.6g} "
              f"val={val_loss:.6g} best={best:.6g} stale={stale}", flush=True)
        if stale >= args.patience:
            print(f"[early-stop] {name} epoch={epoch}", flush=True); break

    model.load_state_dict(torch.load(best_file, map_location=device, weights_only=True))
    test_loss = mse(model, data, range(90, 100), device, True)
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(90, 100):
            x, _, knn = batch(data, i, device)
            with torch.amp.autocast("cuda"):
                predictions.append(model(x, knn).float().cpu().numpy()[0])
    np.savez_compressed(output / "test_predictions.npz", x=data["x"][90:100],
                        y_true=data["y"][90:100], y_pred=np.stack(predictions),
                        y_mean=data["y_mean"], y_std=data["y_std"])
    result = {
        "status": "complete", "model": name,
        "scope": "100-aircraft stratified single-channel surface-field transfer trial",
        "families": list(FAMILIES), "seed": args.seed, "samples": 100,
        "train_samples": 80, "validation_samples": 10, "test_samples": 10,
        "points_per_aircraft": args.points, "knn_k": args.knn,
        "epochs_completed": completed_epoch, "best_validation_normalized_mse": best,
        "test_normalized_mse": test_loss,
        "train_seconds_this_invocation": time.time() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(0),
    }
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "formal_100aircraft")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--models", nargs="+", default=["transolver_tiny", "raptor"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    cache = prepare_cache(args.archive, HERE / "cache", args.points, args.knn, args.seed)
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
