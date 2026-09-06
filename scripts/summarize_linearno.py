"""Summarize matched seed-42 modern baselines without fabricating missing data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = [
    "task", "model", "rel_l2_ux", "rel_l2_uy", "rel_l2_p", "rel_l2_nut",
    "macro_rel_l2", "parameters", "train_seconds", "peak_memory_mib",
    "inference_ms_per_sim", "device",
]


def main() -> None:
    roots = [
        PROJECT / "outputs/linearno_formal",
        PROJECT / "outputs/modern_formal",
        PROJECT / "outputs/modern_formal_fixed",
        PROJECT / "outputs/aoa_formal",
        PROJECT / "outputs/formal",
    ]
    rows, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("result.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record.get("task"), record.get("model"), record.get("seed"))
            if record.get("seed") != 42 or key in seen:
                continue
            seen.add(key)
            rows.append({field: record.get(field) for field in FIELDS})
    rows.sort(key=lambda row: (str(row["task"]), str(row["model"])))
    output = PROJECT / "outputs/linearno_seed42_comparison.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

