"""Run the four authorized RAPTOR adapter ablations sequentially on CUDA."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent
JOBS = [
    ("REYNOLDS_NO_ADAPTER", "configs/reynolds_ablation_no_adapter.yaml"),
    ("REYNOLDS_NONZERO_INIT", "configs/reynolds_ablation_nonzero_init.yaml"),
    ("AOA_NO_ADAPTER", "configs/aoa_ablation_no_adapter.yaml"),
    ("AOA_NONZERO_INIT", "configs/aoa_ablation_nonzero_init.yaml"),
]


def validate_result(config_path: str) -> None:
    import yaml

    cfg = yaml.safe_load((PROJECT / config_path).read_text(encoding="utf-8"))
    task = cfg["tasks"][0]
    result_path = PROJECT / cfg["output_dir"] / f"{task}-raptor-s42" / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"Missing result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("device") != "cuda":
        raise RuntimeError(f"Non-CUDA result rejected: {result_path}")
    if result.get("adapter_enabled") != cfg["adapter_enabled"]:
        raise RuntimeError(f"Adapter switch mismatch: {result_path}")
    if float(result.get("adapter_alpha_init", -1)) != float(cfg["adapter_alpha_init"]):
        raise RuntimeError(f"Adapter initialization mismatch: {result_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    print(f"CORE_ABLATION_PIPELINE_START {datetime.now().isoformat(timespec='seconds')}", flush=True)
    for label, config in JOBS:
        print(f"JOB_START {label} {datetime.now().isoformat(timespec='seconds')}", flush=True)
        subprocess.run(
            [sys.executable, "-s", "raptor/run.py", "--config", f"../{config}"],
            cwd=PROJECT,
            check=True,
        )
        validate_result(config)
        print(f"JOB_DONE {label} {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"CORE_ABLATION_PIPELINE_COMPLETE {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
