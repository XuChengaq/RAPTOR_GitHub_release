"""Run the missing AirfRANS seeds serially without touching seed-42 outputs.

The existing formal model runners perform the actual training. This supervisor
only pins their configurations, records progress, and stops on the first error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs" / "airfrans_multiseed_41_43"
STATUS = OUT / "pipeline_status.json"
MANIFEST = OUT / "job_manifest.json"
SEEDS = (41, 43)
TASKS = ("reynolds", "aoa")
MODELS = (
    "raptor",
    "transolver_tiny",
    "linearno_tiny",
    "transolver_plus_tiny",
    "mono_tiny",
)

EXPECTED_PROTOCOL = {
    "train_simulations": 200,
    "test_simulations": 200,
    "validation_fraction": 0.10,
    "points_per_simulation": 4096,
    "surface_fraction": 0.20,
    "token_count": 64,
    "knn_k": 12,
    "hidden_dim": 64,
    "depth": 4,
    "heads": 4,
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.000001,
    "amp": True,
    "patience": 15,
    "data_seed": 2026,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_for(model: str, task: str) -> str:
    if model == "raptor":
        return "configs/formal.yaml" if task == "reynolds" else "configs/aoa_raptor_formal.yaml"
    if model == "transolver_tiny":
        return (
            "configs/reynolds_transolver_tiny_formal.yaml"
            if task == "reynolds"
            else "configs/aoa_transolver_tiny_formal.yaml"
        )
    if model == "linearno_tiny":
        return "configs/linearno_formal.yaml"
    if model == "transolver_plus_tiny":
        return (
            "configs/transolver_plus_fixed_formal.yaml"
            if task == "reynolds"
            else "configs/aoa_modern_formal.yaml"
        )
    if model == "mono_tiny":
        return "configs/modern_formal.yaml" if task == "reynolds" else "configs/aoa_modern_formal.yaml"
    raise ValueError(model)


def load_config(config_path: str, task: str, model: str) -> dict:
    cfg = yaml.safe_load((PROJECT / config_path).read_text(encoding="utf-8"))
    for key, expected in EXPECTED_PROTOCOL.items():
        if cfg.get(key) != expected:
            raise ValueError(f"{config_path}: {key}={cfg.get(key)!r}, expected {expected!r}")
    if task not in cfg["tasks"]:
        raise ValueError(f"{config_path}: task {task} not configured")
    if model in {"linearno_tiny", "transolver_plus_tiny", "mono_tiny"}:
        if model not in cfg["models"]:
            raise ValueError(f"{config_path}: model {model} not configured")
    if not (PROJECT / cfg["data_root"]).is_dir():
        raise FileNotFoundError(PROJECT / cfg["data_root"])
    if not (PROJECT / "cache" / f"{task}_tr200_te200_n4096.npz").is_file():
        raise FileNotFoundError(f"fixed data cache missing for {task}")
    return cfg


def make_job(model: str, task: str, seed: int) -> dict:
    config_path = config_for(model, task)
    cfg = load_config(config_path, task, model)
    if model in {"raptor", "transolver_tiny"}:
        result_dir = PROJECT / cfg["output_dir"] / f"{task}-{model}-s{seed}"
        runner = f"{model}/run.py"
        config_arg = f"../{config_path}"
        model_arg: list[str] = []
    else:
        result_dir = PROJECT / cfg["output_dir"] / model / f"{task}-{model}-s{seed}"
        runner = "run_modern_baselines.py"
        config_arg = config_path
        model_arg = ["--model", model]
    command = [
        sys.executable,
        "-s",
        runner,
        "--config",
        config_arg,
        *model_arg,
        "--task",
        task,
        "--seed",
        str(seed),
    ]
    return {
        "model": model,
        "task": task,
        "seed": seed,
        "config": config_path,
        "command": command,
        "result_dir": str(result_dir.relative_to(PROJECT)),
        "log": str((OUT / "logs" / f"{task}-{model}-s{seed}.log").relative_to(PROJECT)),
        "data_seed": cfg["data_seed"],
    }


def all_jobs() -> list[dict]:
    # Closest comparisons first; the other two modern baselines follow.
    order = (*MODELS[:3], *MODELS[3:])
    return [
        make_job(model, task, seed)
        for model in order
        for seed in SEEDS
        for task in TASKS
    ]


def result_path(job: dict) -> Path:
    return PROJECT / job["result_dir"] / "result.json"


def validate_result(job: dict) -> dict:
    path = result_path(job)
    checkpoint = path.with_name("best.pt")
    if not path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"result/checkpoint incomplete: {path.parent}")
    result = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "task", "seed"):
        if result.get(key) != job[key]:
            raise ValueError(f"{path}: {key}={result.get(key)!r}, expected {job[key]!r}")
    if result.get("device") != "cuda":
        raise ValueError(f"{path}: result was not trained on CUDA")
    macro = float(result["macro_rel_l2"])
    if not math.isfinite(macro) or macro < 0:
        raise ValueError(f"{path}: invalid macro relative L2: {macro}")
    return {
        "model": job["model"],
        "task": job["task"],
        "seed": job["seed"],
        "macro_rel_l2": macro,
        "epochs": result.get("epochs"),
        "peak_memory_mib": result.get("peak_memory_mib"),
        "train_seconds": result.get("train_seconds"),
        "result": str(path.relative_to(PROJECT)),
    }


def write_status(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def run_job(job: dict, state: dict, index: int, cooldown: float) -> None:
    destination = PROJECT / job["result_dir"]
    if result_path(job).exists():
        summary = validate_result(job)
        summary["completed_at"] = now()
        summary["reused_existing"] = True
        state["completed_jobs"].append(summary)
        write_status(state)
        print(f"JOB_REUSED {index}/{state['total_jobs']} {job['model']} {job['task']} s{job['seed']}", flush=True)
        return
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Incomplete output already exists; inspect before retrying: {destination}")

    destination.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT / job["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state["current_job"] = {**job, "index": index, "started_at": now()}
    write_status(state)
    print(f"JOB_START {index}/{state['total_jobs']} {job['model']} {job['task']} s{job['seed']} {now()}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            job["command"],
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        state["current_job"]["child_pid"] = process.pid
        write_status(state)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"Job {index} exited with code {returncode}; see {log_path}")
    summary = validate_result(job)
    summary["completed_at"] = now()
    summary["reused_existing"] = False
    state["completed_jobs"].append(summary)
    state["current_job"] = None
    write_status(state)
    print(
        f"JOB_DONE {index}/{state['total_jobs']} {job['model']} {job['task']}"
        f" s{job['seed']} macro={summary['macro_rel_l2']:.6f} {now()}",
        flush=True,
    )
    if index < state["total_jobs"] and cooldown > 0:
        time.sleep(cooldown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be nonnegative")
    jobs = all_jobs()
    if len(jobs) != 20 or len({(j["model"], j["task"], j["seed"]) for j in jobs}) != 20:
        raise AssertionError("Expected 20 unique model/task/seed jobs")

    # A complete seed-42 checkpoint is required for each three-seed group.
    for model in MODELS:
        for task in TASKS:
            validate_result(make_job(model, task, 42))

    if args.dry_run:
        print(json.dumps({"total_jobs": len(jobs), "jobs": jobs}, indent=2), flush=True)
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    if STATUS.exists():
        old = json.loads(STATUS.read_text(encoding="utf-8"))
        if old.get("status") == "running":
            raise RuntimeError("A pipeline already reports running; inspect its PID before restarting")

    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "total_jobs": len(jobs),
        "completed_jobs": [],
        "current_job": None,
        "cooldown_seconds": args.cooldown_seconds,
        "data_seed": 2026,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
    write_status(state)
    print(f"AIRFRANS_MULTISEED_START pid={os.getpid()} {now()}", flush=True)
    try:
        for index, job in enumerate(jobs, start=1):
            run_job(job, state, index, args.cooldown_seconds)
        subprocess.run(
            [sys.executable, "-s", "scripts/summarize_airfrans_multiseed.py"],
            cwd=PROJECT,
            check=True,
        )
        state["status"] = "complete"
        state["completed_at"] = now()
        write_status(state)
        print(f"AIRFRANS_MULTISEED_COMPLETE {state['completed_at']}", flush=True)
    except Exception as exc:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
        write_status(state)
        print(state["traceback"], file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
