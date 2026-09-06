"""Run four AoA seed-42 models, then bounded RAPTOR ablations before 07:00."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DEADLINE = datetime(2026, 8, 26, 7, 0, 0)


def run(label: str, args: list[str]) -> None:
    print(f"PIPELINE_START {label} {datetime.now().isoformat(timespec='seconds')}", flush=True)
    subprocess.run([sys.executable, "-s", *args], cwd=PROJECT, check=True)
    print(f"PIPELINE_DONE {label} {datetime.now().isoformat(timespec='seconds')}", flush=True)


def formal_results_complete() -> bool:
    paths = [
        PROJECT / "outputs/aoa_formal/raptor/aoa-raptor-s42/result.json",
        PROJECT / "outputs/aoa_formal/transolver_tiny/aoa-transolver_tiny-s42/result.json",
        PROJECT / "outputs/aoa_formal/modern/mono_tiny/aoa-mono_tiny-s42/result.json",
        PROJECT / "outputs/aoa_formal/modern/transolver_plus_tiny/aoa-transolver_plus_tiny-s42/result.json",
    ]
    return all(
        path.exists()
        and json.loads(path.read_text(encoding="utf-8"))["device"] == "cuda"
        for path in paths
    )


def main() -> None:
    run("RAPTOR_AOA", ["raptor/run.py", "--config", "../configs/aoa_raptor_formal.yaml"])
    run("TRANSOLVER_TINY_AOA", ["transolver_tiny/run.py", "--config", "../configs/aoa_transolver_tiny_formal.yaml"])
    run("MONO_TINY_AOA", ["run_modern_baselines.py", "--config", "configs/aoa_modern_formal.yaml", "--model", "mono_tiny"])
    run("TRANSOLVER_PLUS_TINY_AOA", ["run_modern_baselines.py", "--config", "configs/aoa_modern_formal.yaml", "--model", "transolver_plus_tiny"])
    if formal_results_complete() and datetime.now() < DEADLINE:
        print("ABLATION_GATE passed", flush=True)
        run("RAPTOR_NO_ADAPTER_AOA", ["raptor/run.py", "--config", "../configs/aoa_ablation_no_adapter.yaml"])
        if datetime.now() < DEADLINE:
            run("RAPTOR_NONZERO_INIT_AOA", ["raptor/run.py", "--config", "../configs/aoa_ablation_nonzero_init.yaml"])
        else:
            print("ABLATION_GATE deadline reached after first ablation", flush=True)
    else:
        print("ABLATION_GATE skipped: formal incomplete or deadline reached", flush=True)


if __name__ == "__main__":
    main()
