from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs" / "formal_100car"
OUT = ROOT / "analysis"
MODELS = ["pointnet", "transolver_tiny", "no_adapter", "raptor"]
LABELS = {
    "pointnet": "PointNet",
    "transolver_tiny": "Transolver-Tiny",
    "no_adapter": "RAPTOR w/o adapter",
    "raptor": "RAPTOR",
}


def percentile_ci(values: np.ndarray, rng: np.random.Generator,
                  resamples: int = 20000) -> tuple[float, float]:
    n = len(values)
    draws = rng.choice(values, size=(resamples, n), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {m: np.load(ROOT / m / "test_predictions.npz") for m in MODELS}
    results = {
        m: json.loads((ROOT / m / "result.json").read_text(encoding="utf-8"))
        for m in MODELS
    }
    truth = arrays["raptor"]["y_true"]
    per_vehicle = {
        m: np.mean((arrays[m]["y_pred"] - truth) ** 2, axis=(1, 2)) for m in MODELS
    }

    with (OUT / "per_vehicle_normalized_mse.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_vehicle_index", *MODELS])
        for i in range(len(truth)):
            writer.writerow([i, *[f"{per_vehicle[m][i]:.10f}" for m in MODELS]])

    rng = np.random.default_rng(20260827)
    comparisons = {}
    for baseline in ["pointnet", "transolver_tiny", "no_adapter"]:
        # Positive values mean that RAPTOR has lower error.
        improvement = (per_vehicle[baseline] - per_vehicle["raptor"]) / per_vehicle[baseline] * 100
        lo, hi = percentile_ci(improvement, rng)
        wins = int(np.sum(per_vehicle["raptor"] < per_vehicle[baseline]))
        ties = int(np.sum(np.isclose(per_vehicle["raptor"], per_vehicle[baseline], rtol=1e-7, atol=1e-9)))
        comparisons[baseline] = {
            "mean_paired_improvement_percent": float(improvement.mean()),
            "median_paired_improvement_percent": float(np.median(improvement)),
            "bootstrap_95_ci_percent": [lo, hi],
            "strict_wins": wins,
            "losses": int(len(improvement) - wins - ties),
            "ties": ties,
            "win_rate_percent": float(wins / len(improvement) * 100),
        }

    summary = {
        "protocol": "100 vehicles; 80/10/10 split; 4096 nodes; seed 42",
        "test_vehicles": int(len(truth)),
        "model_metrics": {
            m: {
                "mean_test_normalized_mse_from_predictions": float(per_vehicle[m].mean()),
                "median_vehicle_normalized_mse": float(np.median(per_vehicle[m])),
                "std_vehicle_normalized_mse": float(per_vehicle[m].std(ddof=1)),
                "result_json_test_normalized_mse": float(results[m]["test_normalized_mse"]),
                "best_validation_normalized_mse": float(results[m]["best_validation_normalized_mse"]),
                "parameters": int(results[m]["parameters"]),
                "peak_memory_mib": float(results[m]["peak_memory_mib"]),
                "train_seconds": float(results[m]["train_seconds_this_invocation"]),
            }
            for m in MODELS
        },
        "raptor_paired_comparisons": comparisons,
        "adapter_conclusion": {
            "reference": "no_adapter",
            **comparisons["no_adapter"],
        },
    }
    (OUT / "formal_100car_statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (OUT / "raptor_paired_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "mean_improvement_percent", "median_improvement_percent",
                         "ci_low", "ci_high", "wins", "losses", "ties", "win_rate_percent"])
        for baseline, row in comparisons.items():
            writer.writerow([baseline, row["mean_paired_improvement_percent"],
                             row["median_paired_improvement_percent"],
                             *row["bootstrap_95_ci_percent"], row["strict_wins"],
                             row["losses"], row["ties"], row["win_rate_percent"]])

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), constrained_layout=True)
    x = np.arange(len(truth))
    palette = {"pointnet": "#d4a72c", "transolver_tiny": "#52789e",
               "no_adapter": "#9aa5ad", "raptor": "#087f8c"}

    for m in MODELS:
        axes[0].plot(x + 1, per_vehicle[m], marker="o", markersize=4.2,
                     linewidth=1.6, color=palette[m], label=LABELS[m])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Test vehicle")
    axes[0].set_ylabel("Normalized MSE (log scale) ↓")
    axes[0].set_title("(a) Per-vehicle reconstruction error", loc="left", weight="bold")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)

    means = [per_vehicle[m].mean() for m in MODELS]
    stds = [per_vehicle[m].std(ddof=1) for m in MODELS]
    bars = axes[1].bar(np.arange(4), means, yerr=stds, capsize=3,
                       color=[palette[m] for m in MODELS], width=0.66)
    axes[1].set_xticks(np.arange(4), ["PointNet", "Transolver\nTiny", "No adapter", "RAPTOR"])
    axes[1].set_ylabel("Mean normalized MSE ↓")
    axes[1].set_title("(b) Test-set mean ± vehicle SD", loc="left", weight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                     f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)

    refs = ["pointnet", "transolver_tiny", "no_adapter"]
    imp = [comparisons[r]["mean_paired_improvement_percent"] for r in refs]
    ci = np.array([comparisons[r]["bootstrap_95_ci_percent"] for r in refs])
    y = np.arange(3)
    axes[2].errorbar(imp, y, xerr=np.vstack([np.array(imp) - ci[:, 0], ci[:, 1] - np.array(imp)]),
                     fmt="o", color="#087f8c", ecolor="#5b7083", capsize=4, markersize=7)
    axes[2].axvline(0, color="#64748b", linewidth=1, linestyle="--")
    axes[2].set_yticks(y, [LABELS[r] for r in refs])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("RAPTOR paired improvement (%) →")
    axes[2].set_title("(c) Mean improvement with 95% bootstrap CI", loc="left", weight="bold")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    for i, r in enumerate(refs):
        axes[2].text(ci[i, 1] + 0.8, i,
                     f"{comparisons[r]['strict_wins']}/10 wins", va="center", fontsize=8.5)

    fig.suptitle("100-Vehicle RAPTOR Evaluation and Local-Adapter Contribution",
                 fontsize=15, weight="bold")
    png = OUT / "Figure_formal_100car_statistics.png"
    fig.savefig(png, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(png)


if __name__ == "__main__":
    main()
