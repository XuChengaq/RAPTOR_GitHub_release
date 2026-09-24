"""Summarize three AirfRANS training seeds without significance claims."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run_airfrans_multiseed_pipeline import MODELS, OUT, TASKS, make_job, validate_result  # noqa: E402


SEEDS = (41, 42, 43)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    per_seed: list[dict] = []
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for task in TASKS:
        for model in MODELS:
            grouped[(task, model)] = {}
            for seed in SEEDS:
                job = make_job(model, task, seed)
                row = validate_result(job)
                result_path = PROJECT / row["result"]
                result = json.loads(result_path.read_text(encoding="utf-8"))
                row.update(
                    **{f"rel_l2_{name}": result[f"rel_l2_{name}"] for name in ("ux", "uy", "p", "nut")},
                    best_validation_mse=result.get("best_validation_mse"),
                    checkpoint=str(result_path.with_name("best.pt").relative_to(PROJECT)),
                    history_exists=result_path.with_name("history.csv").is_file(),
                    recovered_evaluation=bool(result.get("recovered_evaluation", False)),
                )
                per_seed.append(row)
                grouped[(task, model)][seed] = row["macro_rel_l2"]

    aggregates = []
    for task in TASKS:
        for model in MODELS:
            values = [grouped[(task, model)][seed] for seed in SEEDS]
            aggregates.append(
                {
                    "task": task,
                    "model": model,
                    "n_training_seeds": len(values),
                    "training_seeds": "41/42/43",
                    "macro_rel_l2_mean": statistics.mean(values),
                    "macro_rel_l2_sample_sd": statistics.stdev(values),
                    "seed_41": values[0],
                    "seed_42": values[1],
                    "seed_43": values[2],
                }
            )

    paired = []
    for task in TASKS:
        raptor = grouped[(task, "raptor")]
        for comparator in MODELS:
            if comparator == "raptor":
                continue
            baseline = grouped[(task, comparator)]
            improvements = [
                (baseline[seed] - raptor[seed]) / baseline[seed] * 100.0
                for seed in SEEDS
            ]
            paired.append(
                {
                    "task": task,
                    "comparator": comparator,
                    "n_training_seeds": 3,
                    "paired_improvement_mean_percent": statistics.mean(improvements),
                    "paired_improvement_sample_sd_percent": statistics.stdev(improvements),
                    "seed_41_percent": improvements[0],
                    "seed_42_percent": improvements[1],
                    "seed_43_percent": improvements[2],
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "per_seed_results.csv",
        [
            "task", "model", "seed", "rel_l2_ux", "rel_l2_uy", "rel_l2_p",
            "rel_l2_nut", "macro_rel_l2", "best_validation_mse", "epochs",
            "peak_memory_mib", "train_seconds", "checkpoint", "history_exists",
            "recovered_evaluation", "result",
        ],
        per_seed,
    )
    write_csv(
        OUT / "three_seed_summary.csv",
        ["task", "model", "n_training_seeds", "training_seeds", "macro_rel_l2_mean", "macro_rel_l2_sample_sd", "seed_41", "seed_42", "seed_43"],
        aggregates,
    )
    write_csv(
        OUT / "paired_improvements.csv",
        ["task", "comparator", "n_training_seeds", "paired_improvement_mean_percent", "paired_improvement_sample_sd_percent", "seed_41_percent", "seed_42_percent", "seed_43_percent"],
        paired,
    )
    payload = {
        "protocol": {
            "tasks": list(TASKS),
            "training_seeds": list(SEEDS),
            "fixed_data_seed": 2026,
            "nodes": 4096,
            "accuracy_statistic": "mean and sample standard deviation over three training seeds",
            "paired_improvement": "computed within each shared training seed, then summarized descriptively",
            "significance_claim": "none",
            "train_seconds_scope": "wall-clock from training start through final checkpoint test evaluation",
            "per_sample_predictions": "export separately from best.pt after the training pipeline completes",
        },
        "aggregate_rows": aggregates,
        "paired_improvement_rows": paired,
    }
    (OUT / "three_seed_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"AIRFRANS_THREE_SEED_SUMMARY_COMPLETE rows={len(aggregates)} paired={len(paired)}", flush=True)


if __name__ == "__main__":
    main()
