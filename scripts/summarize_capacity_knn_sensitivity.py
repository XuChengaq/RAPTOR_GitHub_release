"""Summarize the AirfRANS capacity-control and K-sensitivity runs."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs" / "capacity_knn_sensitivity"

RUNS = [
    ("nodewise", "reynolds", OUT / "nodewise" / "reynolds-raptor-s42" / "result.json"),
    ("nodewise", "aoa", OUT / "nodewise" / "aoa-raptor-s42" / "result.json"),
    ("knn_k4", "reynolds", OUT / "knn_k4" / "reynolds-raptor-s42" / "result.json"),
    ("knn_k4", "aoa", OUT / "knn_k4" / "aoa-raptor-s42" / "result.json"),
    ("knn_k8", "reynolds", OUT / "knn_k8" / "reynolds-raptor-s42" / "result.json"),
    ("knn_k8", "aoa", OUT / "knn_k8" / "aoa-raptor-s42" / "result.json"),
    ("knn_k12", "reynolds", PROJECT / "outputs" / "formal" / "reynolds-raptor-s42" / "result.json"),
    ("knn_k12", "aoa", PROJECT / "outputs" / "aoa_formal" / "raptor" / "aoa-raptor-s42" / "result.json"),
    ("knn_k24", "reynolds", OUT / "knn_k24" / "reynolds-raptor-s42" / "result.json"),
    ("knn_k24", "aoa", OUT / "knn_k24" / "aoa-raptor-s42" / "result.json"),
]

FIELDS = [
    "variant",
    "task",
    "knn_k",
    "adapter_neighbor_mode",
    "macro_rel_l2",
    "rel_l2_ux",
    "rel_l2_uy",
    "rel_l2_p",
    "rel_l2_nut",
    "parameters",
    "peak_memory_mib",
    "inference_ms_per_sim",
    "train_seconds",
    "epochs",
]


def collect() -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for variant, task, path in RUNS:
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT)))
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {key: result.get(key) for key in FIELDS}
        row.update(
            variant=variant,
            task=task,
            adapter_neighbor_mode=result.get("adapter_neighbor_mode", "knn_mean"),
        )
        rows.append(row)
    return rows, missing


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = collect()
    payload = {
        "protocol": "AirfRANS fixed split, seed 42, N=4096",
        "rows": rows,
        "missing": missing,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)
    if missing:
        print("ACCURACY_COST_PROFILE_SKIPPED incomplete experiment results", flush=True)
        return
    subprocess.run(
        [sys.executable, "-s", "scripts/profile_accuracy_cost_tradeoff.py"],
        cwd=PROJECT,
        check=True,
    )


if __name__ == "__main__":
    main()
