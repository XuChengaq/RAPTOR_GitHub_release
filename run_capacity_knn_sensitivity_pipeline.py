"""Run capacity-control and K-sensitivity AirfRANS experiments sequentially."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs" / "capacity_knn_sensitivity"
STATUS = OUTPUT / "pipeline_status.json"

JOBS = [
    ("NODEWISE_REYNOLDS", "configs/capacity_control_nodewise.yaml", "reynolds"),
    ("NODEWISE_AOA", "configs/capacity_control_nodewise.yaml", "aoa"),
    ("K4_REYNOLDS", "configs/knn_sensitivity_k4.yaml", "reynolds"),
    ("K4_AOA", "configs/knn_sensitivity_k4.yaml", "aoa"),
    ("K8_REYNOLDS", "configs/knn_sensitivity_k8.yaml", "reynolds"),
    ("K8_AOA", "configs/knn_sensitivity_k8.yaml", "aoa"),
    ("K24_REYNOLDS", "configs/knn_sensitivity_k24.yaml", "reynolds"),
    ("K24_AOA", "configs/knn_sensitivity_k24.yaml", "aoa"),
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def result_path(config_path: str, task: str) -> Path:
    cfg = yaml.safe_load((PROJECT / config_path).read_text(encoding="utf-8"))
    return PROJECT / cfg["output_dir"] / f"{task}-raptor-s42" / "result.json"


def validate_result(config_path: str, task: str) -> dict:
    cfg = yaml.safe_load((PROJECT / config_path).read_text(encoding="utf-8"))
    path = result_path(config_path, task)
    if not path.exists():
        raise RuntimeError(f"Missing result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_mode = cfg.get("adapter_neighbor_mode", "knn_mean")
    if result.get("device") != "cuda":
        raise RuntimeError(f"Non-CUDA result rejected: {path}")
    if int(result.get("knn_k", -1)) != int(cfg["knn_k"]):
        raise RuntimeError(f"K mismatch in {path}")
    if result.get("adapter_neighbor_mode") != expected_mode:
        raise RuntimeError(f"Neighbor-mode mismatch in {path}")
    return {
        "task": task,
        "knn_k": result["knn_k"],
        "adapter_neighbor_mode": result["adapter_neighbor_mode"],
        "macro_rel_l2": result["macro_rel_l2"],
        "peak_memory_mib": result["peak_memory_mib"],
        "train_seconds": result["train_seconds"],
        "epochs": result["epochs"],
        "result": str(path.relative_to(PROJECT)),
    }


def write_status(state: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "total_jobs": len(JOBS),
        "completed_jobs": [],
        "current_job": None,
        "log": args.log,
    }
    write_status(state)
    print(f"CAPACITY_KNN_PIPELINE_START {state['started_at']} pid={os.getpid()}", flush=True)
    try:
        for index, (label, config, task) in enumerate(JOBS, start=1):
            state["current_job"] = {
                "index": index,
                "label": label,
                "config": config,
                "task": task,
                "started_at": now(),
                "expected_result": str(result_path(config, task).relative_to(PROJECT)),
            }
            write_status(state)
            print(f"JOB_START {index}/{len(JOBS)} {label} {now()}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-s",
                    "raptor/run.py",
                    "--config",
                    f"../{config}",
                    "--task",
                    task,
                    "--seed",
                    "42",
                ],
                cwd=PROJECT,
                check=True,
            )
            result = validate_result(config, task)
            result.update(label=label, completed_at=now())
            state["completed_jobs"].append(result)
            state["current_job"] = None
            write_status(state)
            print(f"JOB_DONE {index}/{len(JOBS)} {label} {now()}", flush=True)

        subprocess.run(
            [sys.executable, "-s", "scripts/summarize_capacity_knn_sensitivity.py"],
            cwd=PROJECT,
            check=True,
        )
        state["status"] = "complete"
        state["completed_at"] = now()
        state["current_job"] = None
        write_status(state)
        print(f"CAPACITY_KNN_PIPELINE_COMPLETE {state['completed_at']}", flush=True)
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
