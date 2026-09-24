"""Export per-simulation test predictions from the saved best checkpoints.

Run only after the 20-job training pipeline has completed. This is a read-only
checkpoint evaluation; it never trains a model or changes an existing result.
Each NPZ stores predictions in the original AirfRANS target units and uses the
fixed cache ordering as its simulation identifier. The sidecar records hashes
so the arrays can be tied to their checkpoint and test cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run_airfrans_multiseed_pipeline import (  # noqa: E402
    MODELS,
    OUT,
    STATUS,
    TASKS,
    config_for,
    load_config,
    make_job,
    validate_result,
)

SEEDS = (41, 42, 43)
VARIABLES = ("ux", "uy", "p", "nut")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sqrt(np.sum(np.square(pred - target), dtype=np.float64))
    denominator = np.sqrt(np.sum(np.square(target), dtype=np.float64))
    return float(numerator / (denominator + 1e-12))


def model_for(name: str, cfg: dict):
    if name == "raptor":
        from raptor.model import RAPTOR

        return RAPTOR(
            dim=cfg["hidden_dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            slices=cfg["slice_count"],
            knn_k=cfg["knn_k"],
            adapter_enabled=cfg.get("adapter_enabled", True),
            adapter_alpha_init=cfg.get("adapter_alpha_init", 0.0),
            adapter_neighbor_mode=cfg.get("adapter_neighbor_mode", "knn_mean"),
        )
    if name == "transolver_tiny":
        from transolver_tiny.model import TransolverTiny

        return TransolverTiny(
            dim=cfg["hidden_dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            slices=cfg["slice_count"],
        )
    from run_modern_baselines import build_model

    return build_model(name, cfg)


def check_aggregate(pred: np.ndarray, target: np.ndarray, result: dict) -> dict:
    recomputed = {
        f"rel_l2_{name}": rel_l2(pred[..., channel], target[..., channel])
        for channel, name in enumerate(VARIABLES)
    }
    recomputed["macro_rel_l2"] = float(np.mean(list(recomputed.values())))
    # The original evaluator converts predictions to float32 before scoring.
    # A tolerance above normal fp32 summation drift still catches protocol
    # changes or a checkpoint/cache mismatch.
    tolerance = 2e-5
    for name, actual in recomputed.items():
        expected = float(result[name])
        if abs(actual - expected) > tolerance:
            raise ValueError(
                f"{name} mismatch: checkpoint/cache gives {actual:.9g}; "
                f"stored result gives {expected:.9g}"
            )
    return recomputed


def export_one(job: dict, device, torch) -> None:
    cfg = load_config(job["config"], job["task"], job["model"])
    directory = PROJECT / job["result_dir"]
    checkpoint = directory / "best.pt"
    result_path = directory / "result.json"
    array_path = directory / "test_predictions.npz"
    manifest_path = directory / "test_predictions.json"
    if array_path.exists() or manifest_path.exists():
        if array_path.exists() and manifest_path.exists():
            print(f"SKIP_EXISTING {job['task']} {job['model']} s{job['seed']}", flush=True)
            return
        raise RuntimeError(f"Incomplete prediction export in {directory}")

    validate_result(job)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cache_path = PROJECT / cfg["cache_dir"] / (
        f"{job['task']}_tr{cfg['train_simulations']}_"
        f"te{cfg['test_simulations']}_n{cfg['points_per_simulation']}.npz"
    )
    with np.load(cache_path) as cache:
        x = cache["test_x"]
        target = cache["test_y"]
        knn = cache["test_knn"]
        groups = cache["test_groups"]
        xmean, xstd = cache["xmean"], cache["xstd"]
        ymean, ystd = cache["ymean"], cache["ystd"]
        if x.shape[:2] != target.shape[:2] or target.shape[-1] != 4:
            raise ValueError(f"Unexpected test array shape in {cache_path}")
        model = model_for(job["model"], cfg).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        predictions = np.empty_like(target, dtype=np.float32)
        with torch.inference_mode():
            for index in range(len(x)):
                normalized_x = ((x[index] - xmean) / xstd).astype("float32")
                xt = torch.from_numpy(normalized_x)[None].to(device)
                kt = torch.from_numpy(knn[index])[None].to(device)
                gt = torch.from_numpy(groups[index])[None].to(device)
                with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                    output = model(xt, kt, gt)
                predictions[index] = (
                    output.float().cpu().numpy()[0] * ystd + ymean
                ).astype("float32")

        aggregate = check_aggregate(predictions, target, result)
        per_sample_rel_l2 = np.empty((len(x), 4), dtype=np.float64)
        per_sample_mse = np.empty((len(x), 4), dtype=np.float64)
        for index in range(len(x)):
            for channel in range(4):
                per_sample_rel_l2[index, channel] = rel_l2(
                    predictions[index, :, channel], target[index, :, channel]
                )
                per_sample_mse[index, channel] = float(
                    np.mean((predictions[index, :, channel] - target[index, :, channel]) ** 2)
                )

    temporary = array_path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                prediction=predictions,
                test_cache_index=np.arange(len(predictions), dtype=np.int32),
                per_sample_rel_l2=per_sample_rel_l2,
                per_sample_mse=per_sample_mse,
            )
        temporary.replace(array_path)
        manifest = {
            "model": job["model"],
            "task": job["task"],
            "training_seed": job["seed"],
            "data_seed": cfg["data_seed"],
            "checkpoint": str(checkpoint.relative_to(PROJECT)),
            "checkpoint_sha256": sha256(checkpoint),
            "test_cache": str(cache_path.relative_to(PROJECT)),
            "test_cache_sha256": sha256(cache_path),
            "test_identifier": "zero-based index in fixed test cache; original object IDs are not stored there",
            "prediction_units": "original AirfRANS targets, after inverse target standardization",
            "variables": list(VARIABLES),
            "prediction_shape": list(predictions.shape),
            "aggregate_check": aggregate,
            "stored_macro_rel_l2": result["macro_rel_l2"],
            "relative_l2_aggregation": "global across all test nodes, matching result.json; per-sample scores are separate",
            "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print(
        f"EXPORTED {job['task']} {job['model']} s{job['seed']} "
        f"samples={len(predictions)} macro={aggregate['macro_rel_l2']:.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    jobs = [
        make_job(model, task, seed)
        for task in TASKS
        for model in MODELS
        for seed in SEEDS
        if (args.model is None or model == args.model)
        and (args.task is None or task == args.task)
        and (args.seed is None or seed == args.seed)
    ]
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "outputs": [j["result_dir"] for j in jobs]}, indent=2))
        return

    state = json.loads(STATUS.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError("Training pipeline is not complete; refusing GPU inference during training")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    device = torch.device("cuda")
    for job in jobs:
        export_one(job, device, torch)
    print(f"AIRFRANS_PREDICTION_EXPORT_COMPLETE jobs={len(jobs)} output_root={OUT}", flush=True)


if __name__ == "__main__":
    main()
