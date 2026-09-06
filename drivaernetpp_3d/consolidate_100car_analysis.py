"""Consolidated, vehicle-level statistical analysis for the formal 100-car study."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs" / "formal_100car"
OUT = ROOT / "analysis_consolidated_20260906"
MODELS = ["pointnet", "transolver_tiny", "no_adapter", "raptor"]
LABELS = {"pointnet": "PointNet", "transolver_tiny": "Transolver-Tiny",
          "no_adapter": "RAPTOR w/o adapter", "raptor": "RAPTOR"}
COLORS = {"pointnet": "#D8A12E", "transolver_tiny": "#587D9F",
          "no_adapter": "#98A6AE", "raptor": "#087F8C"}


def boot_ci_mean(x: np.ndarray, rng: np.random.Generator, n: int = 50000):
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {m: np.load(ROOT / m / "test_predictions.npz") for m in MODELS}
    results = {m: json.loads((ROOT / m / "result.json").read_text(encoding="utf-8"))
               for m in MODELS}
    truth = arrays["raptor"]["y_true"]
    for m in MODELS:
        if not np.allclose(arrays[m]["y_true"], truth):
            raise ValueError(f"Test targets differ for {m}")
    errors = {m: np.mean((arrays[m]["y_pred"] - truth) ** 2, axis=(1, 2)) for m in MODELS}
    rng = np.random.default_rng(20260906)

    summary_rows, summary = [], {}
    order = sorted(MODELS, key=lambda m: errors[m].mean())
    previous_mean, displayed_rank = None, 0
    for position, m in enumerate(order, start=1):
        x = errors[m]
        if previous_mean is None or not np.isclose(x.mean(), previous_mean, rtol=1e-10, atol=1e-12):
            displayed_rank = position
        rank = displayed_rank
        previous_mean = x.mean()
        lo, hi = boot_ci_mean(x, rng)
        r = results[m]
        summary[m] = {"rank": rank, "mean": float(x.mean()), "median": float(np.median(x)),
                      "std": float(x.std(ddof=1)), "q1": float(np.quantile(x, .25)),
                      "q3": float(np.quantile(x, .75)), "ci95": [lo, hi],
                      "parameters": int(r["parameters"]),
                      "peak_memory_mib": float(r["peak_memory_mib"]),
                      "train_minutes": float(r["train_seconds_this_invocation"] / 60),
                      "best_val_mse": float(r["best_validation_normalized_mse"])}
        summary_rows.append([rank, LABELS[m], x.mean(), np.median(x), x.std(ddof=1), lo, hi,
                             r["best_validation_normalized_mse"], r["parameters"],
                             r["peak_memory_mib"], r["train_seconds_this_invocation"] / 60])

    paired_rows, paired = [], {}
    for b in ["pointnet", "transolver_tiny", "no_adapter"]:
        imp = 100 * (errors[b] - errors["raptor"]) / errors[b]
        lo, hi = boot_ci_mean(imp, rng)
        diff = errors[b] - errors["raptor"]
        try:
            p = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")
        wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0)); ties = len(diff)-wins-losses
        paired[b] = {"mean_improvement_percent": float(imp.mean()),
                     "median_improvement_percent": float(np.median(imp)), "ci95": [lo, hi],
                     "wins": wins, "losses": losses, "ties": ties,
                     "wilcoxon_one_sided_p": p}
        paired_rows.append([LABELS[b], imp.mean(), np.median(imp), lo, hi,
                            f"{wins}/{losses}/{ties}", p])

    write_csv(OUT / "Table_100car_model_summary.csv",
              ["Rank", "Model", "Mean test MSE", "Median", "SD", "CI low", "CI high",
               "Best val MSE", "Parameters", "Peak memory MiB", "Training min"], summary_rows)
    write_csv(OUT / "Table_RAPTOR_paired_comparisons.csv",
              ["Reference", "Mean improvement %", "Median improvement %", "CI low", "CI high",
               "W/L/T", "One-sided Wilcoxon p"], paired_rows)
    write_csv(OUT / "Per_vehicle_normalized_MSE.csv",
              ["Test vehicle", *[LABELS[m] for m in MODELS]],
              [[i+1, *[errors[m][i] for m in MODELS]] for i in range(len(truth))])

    tex1 = [r"\begin{table}[t]", r"\centering", r"\caption{Vehicle-level accuracy and resource statistics for the 100-car experiment. The confidence intervals are 95\% vehicle-level bootstrap intervals for the mean normalized MSE.}", r"\label{tab:car_summary}", r"\small", r"\begin{tabular}{clccccrrr}", r"\toprule", r"Rank & Model & Mean & Median & SD & 95\% CI & Params. & Memory & Time \\", r" & & & & & & & (MiB) & (min) \\", r"\midrule"]
    for row in summary_rows:
        rank, label, mean, med, sd, lo, hi, _, params, memory, minutes = row
        style = r"\textbf{" + label + "}" if label == "RAPTOR" else label
        tex1.append(f"{rank} & {style} & {mean:.4f} & {med:.4f} & {sd:.4f} & [{lo:.4f}, {hi:.4f}] & {params:,} & {memory:.1f} & {minutes:.1f}" + r" \\")
    tex1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "Table_100car_model_summary.tex").write_text("\n".join(tex1), encoding="utf-8")

    tex2 = [r"\begin{table}[t]", r"\centering", r"\caption{Paired vehicle-level comparison of RAPTOR with the baselines. Positive improvement denotes a lower normalized MSE for RAPTOR.}", r"\label{tab:car_paired}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule", r"Reference & Mean improvement & 95\% CI & W/L/T & $p$ \\", r"\midrule"]
    for row in paired_rows:
        label, mean, _, lo, hi, wlt, p = row
        tex2.append(f"{label} & {mean:.2f}\\% & [{lo:.2f}, {hi:.2f}]\\% & {wlt} & {p:.4f}" + r" \\")
    tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "Table_RAPTOR_paired_comparisons.tex").write_text("\n".join(tex2), encoding="utf-8")

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
    pos = np.arange(1, 5)
    vals = [errors[m] for m in MODELS]
    bp = ax[0,0].boxplot(vals, positions=pos, widths=.55, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COLORS[m]); patch.set_alpha(.78)
    for j, (m, x) in enumerate(zip(MODELS, vals), start=1):
        jitter = np.linspace(-.12, .12, len(x)); ax[0,0].scatter(j+jitter, x, s=22, color="#17202A", alpha=.72, zorder=3)
    ax[0,0].set_xticks(pos, ["PointNet", "Transolver\nTiny", "No adapter", "RAPTOR"])
    ax[0,0].set_ylabel("Vehicle normalized MSE ↓")
    ax[0,0].set_title("(a) Distribution across 10 held-out vehicles", loc="left", weight="bold")

    for i in range(len(truth)):
        ax[0,1].plot([0,1], [errors["transolver_tiny"][i], errors["raptor"][i]],
                     color="#A7B2BC", linewidth=1.1, marker="o", markersize=3.8)
    ax[0,1].plot([0,1], [errors["transolver_tiny"].mean(), errors["raptor"].mean()],
                 color="#087F8C", linewidth=3, marker="o", markersize=7, label="Mean")
    ax[0,1].set_xticks([0,1], ["Transolver-Tiny", "RAPTOR"])
    ax[0,1].set_ylabel("Vehicle normalized MSE ↓")
    ax[0,1].set_title("(b) Paired change on identical vehicles", loc="left", weight="bold")
    ax[0,1].legend(frameon=False)

    imp = 100*(errors["no_adapter"]-errors["raptor"])/errors["no_adapter"]
    colors = np.where(imp >= 0, "#087F8C", "#C44E52")
    ax[1,0].bar(np.arange(1,11), imp, color=colors)
    ax[1,0].axhline(0, color="#2F3A45", linewidth=.9)
    ax[1,0].axhline(imp.mean(), color="#087F8C", linestyle="--", linewidth=1.4,
                    label=f"Mean = {imp.mean():.2f}%")
    ax[1,0].set_xlabel("Test vehicle")
    ax[1,0].set_ylabel("Adapter improvement (%) ↑")
    ax[1,0].set_title("(c) Local-adapter contribution by vehicle", loc="left", weight="bold")
    ax[1,0].legend(frameon=False)

    offsets = {"pointnet": (6, 6), "transolver_tiny": (6, 12),
               "no_adapter": (6, -2), "raptor": (-76, -2)}
    for m in MODELS:
        ax[1,1].scatter(results[m]["peak_memory_mib"], errors[m].mean(), s=95,
                        color=COLORS[m], label=LABELS[m], edgecolor="white", linewidth=.8)
        ax[1,1].annotate(LABELS[m], (results[m]["peak_memory_mib"], errors[m].mean()),
                         xytext=offsets[m], textcoords="offset points", fontsize=8)
    ax[1,1].set_xlabel("Peak GPU memory (MiB) ↓")
    ax[1,1].set_ylabel("Mean test normalized MSE ↓")
    ax[1,1].set_title("(d) Accuracy–memory trade-off", loc="left", weight="bold")
    for a in ax.flat:
        a.grid(alpha=.22); a.spines[["top","right"]].set_visible(False)
    fig.suptitle("100-Car Evaluation: Accuracy, Paired Robustness, and Adapter Contribution",
                 fontsize=15, weight="bold")
    png = OUT / "Figure_100car_consolidated_statistics.png"
    pdf = OUT / "Figure_100car_consolidated_statistics.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)

    report = {"protocol": "100 cars; fixed 80/10/10 split; 4096 nodes; seed 42",
              "statistical_unit": "held-out vehicle (n=10)", "model_summary": summary,
              "paired_raptor_comparisons": paired, "figure": str(png)}
    (OUT / "consolidated_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
