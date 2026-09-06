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
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from audit_dataset import SliceReader, nested_data_offset
from raptor.model import RAPTOR


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


def choose_samples(inner: zipfile.ZipFile, count: int, seed: int) -> list[str]:
    families: dict[str, list[str]] = {}
    for name in inner.namelist():
        if name.startswith("DrivAerNet/") and name.lower().endswith(".npz"):
            family = name.split("/")[1]
            families.setdefault(family, []).append(name)
    rng = random.Random(seed)
    for names in families.values():
        names.sort()
        rng.shuffle(names)
    selected: list[str] = []
    family_names = sorted(families)
    cursor = 0
    while len(selected) < count:
        family = family_names[cursor % len(family_names)]
        index = cursor // len(family_names)
        if index < len(families[family]):
            selected.append(families[family][index])
        cursor += 1
    return selected


def knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    pts = torch.from_numpy(points).float()
    distance = torch.cdist(pts, pts)
    return torch.topk(distance, k=k + 1, largest=False).indices[:, 1:].numpy().astype(np.int64)


def prepare_cache(archive: Path, output: Path, samples: int, points: int, k: int, seed: int) -> Path:
    cache = output / f"small3d_s{samples}_n{points}_k{k}_seed{seed}.npz"
    split_file = output / "split_manifest.json"
    if cache.exists() and split_file.exists():
        return cache
    output.mkdir(parents=True, exist_ok=True)
    raw, buffered, inner = open_inner(archive)
    try:
        names = choose_samples(inner, samples, seed)
        xs, ys, knns = [], [], []
        for index, name in enumerate(names):
            with inner.open(name) as handle:
                payload = np.load(io.BytesIO(handle.read()), allow_pickle=False)
                nodes = np.asarray(payload["nodes_list"], dtype=np.float32)
                target = np.asarray(payload["features_list"], dtype=np.float32).reshape(-1, 1)
            if len(nodes) != len(target) or len(nodes) < points:
                raise ValueError(f"invalid node target arrays: {name}")
            rng = np.random.default_rng(seed + index)
            chosen = np.sort(rng.choice(len(nodes), size=points, replace=False))
            x, y = nodes[chosen], target[chosen]
            xs.append(x)
            ys.append(y)
            knns.append(knn_indices(x, k))
            print(f"[cache] {index + 1}/{samples} {name}", flush=True)
    finally:
        inner.close()
        buffered.close()
        raw.close()

    train_end = int(samples * 0.8)
    val_end = train_end + int(samples * 0.1)
    train_y = np.stack(ys[:train_end])
    y_mean = train_y.mean(axis=(0, 1), keepdims=True)
    y_std = train_y.std(axis=(0, 1), keepdims=True) + 1e-6
    x = np.stack(xs).astype(np.float32)
    y = ((np.stack(ys) - y_mean) / y_std).astype(np.float32)
    knn = np.stack(knns).astype(np.int64)
    np.savez_compressed(cache, x=x, y=y, knn=knn, y_mean=y_mean, y_std=y_std)
    manifest = {
        "seed": seed,
        "samples": samples,
        "points_per_vehicle": points,
        "target": "single-channel vehicle surface node field (features_list)",
        "train": names[:train_end],
        "validation": names[train_end:val_end],
        "test": names[val_end:],
    }
    split_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache


def batch(data, index: int, device: torch.device):
    x = torch.from_numpy(data["x"][index:index + 1]).to(device)
    y = torch.from_numpy(data["y"][index:index + 1]).to(device)
    knn = torch.from_numpy(data["knn"][index:index + 1]).long().to(device)
    return x, y, knn


@torch.no_grad()
def mse(model, data, indices, device, amp):
    model.eval()
    values = []
    for i in indices:
        x, y, knn = batch(data, i, device)
        with torch.amp.autocast("cuda", enabled=amp):
            values.append(torch.mean((model(x, knn) - y) ** 2).item())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "small_trial")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cache = prepare_cache(args.archive, HERE / "cache", args.samples, args.points, args.knn, args.seed)
    data = np.load(cache)
    assert data["x"].shape == (args.samples, args.points, 3)
    assert data["y"].shape == (args.samples, args.points, 1)
    assert data["knn"].shape == (args.samples, args.points, args.knn)
    device = torch.device("cuda")
    model = RAPTOR(cin=3, cout=1, dim=48, depth=3, heads=4, slices=32, knn_k=args.knn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    train_end, val_end = int(args.samples * 0.8), int(args.samples * 0.9)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    checkpoint = args.output / "best.pt"
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    # Mandatory one-sample CUDA AMP forward/backward smoke before training.
    x0, y0, k0 = batch(data, 0, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        smoke_loss = torch.mean((model(x0, k0) - y0) ** 2)
    scaler.scale(smoke_loss).backward()
    optimizer.zero_grad(set_to_none=True)
    (args.output / "cuda_smoke.json").write_text(json.dumps({
        "passed": True, "device": torch.cuda.get_device_name(0),
        "loss": float(smoke_loss.detach()), "shape": list(model(x0, k0).shape),
    }, indent=2), encoding="utf-8")
    print(f"[smoke] passed loss={float(smoke_loss.detach()):.6g}", flush=True)

    for epoch in range(1, args.epochs + 1):
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
            losses.append(loss.detach().item())
        train_loss = float(np.mean(losses))
        val_loss = mse(model, data, range(train_end, val_end), device, True)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), checkpoint)
        history.append((epoch, train_loss, val_loss, best))
        print(f"[raptor3d] epoch={epoch} train={train_loss:.6g} val={val_loss:.6g} best={best:.6g}", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss = mse(model, data, range(val_end, args.samples), device, True)
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(val_end, args.samples):
            x, _, knn = batch(data, i, device)
            with torch.amp.autocast("cuda"):
                test_predictions.append(model(x, knn).float().cpu().numpy()[0])
    np.savez_compressed(
        args.output / "test_predictions.npz",
        x=data["x"][val_end:args.samples],
        y_true=data["y"][val_end:args.samples],
        y_pred=np.stack(test_predictions),
        y_mean=data["y_mean"],
        y_std=data["y_std"],
    )
    result = {
        "status": "complete", "scope": "small-scale single-channel vehicle surface-field trial",
        "samples": args.samples, "train_samples": train_end,
        "validation_samples": val_end - train_end, "test_samples": args.samples - val_end,
        "points_per_vehicle": args.points, "seed": args.seed, "epochs": args.epochs,
        "best_validation_normalized_mse": best, "test_normalized_mse": test_loss,
        "train_seconds": time.time() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()), "device": torch.cuda.get_device_name(0),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_normalized_mse", "validation_normalized_mse", "best_validation_normalized_mse"])
        writer.writerows(history)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
