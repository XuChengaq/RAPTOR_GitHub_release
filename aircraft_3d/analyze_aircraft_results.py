from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "formal_100aircraft"
ANALYSIS = OUTPUT / "analysis"
MODELS = ("transolver_tiny", "raptor")
MODEL_LABELS = {"transolver_tiny": "Transolver-Tiny", "raptor": "RAPTOR"}


def load_inputs() -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    manifest = json.loads((HERE / "cache" / "split_manifest.json").read_text(encoding="utf-8"))
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        with np.load(OUTPUT / model / "test_predictions.npz") as payload:
            predictions[model] = {key: payload[key].copy() for key in payload.files}
    return manifest, predictions


def family_from_name(name: str) -> str:
    return name.replace("\\", "/").split("/")[1]


def sample_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_pred - y_true) ** 2, axis=(1, 2))


def compute_statistics(manifest: dict, predictions: dict[str, dict[str, np.ndarray]]):
    names = manifest["test"]
    families = np.asarray([family_from_name(name) for name in names])
    reference = predictions["raptor"]["y_true"]
    for model in MODELS:
        if not np.allclose(reference, predictions[model]["y_true"]):
            raise RuntimeError(f"test targets do not match for {model}")
        if not np.allclose(predictions["raptor"]["x"], predictions[model]["x"]):
            raise RuntimeError(f"test coordinates do not match for {model}")

    errors = {model: sample_mse(reference, predictions[model]["y_pred"]) for model in MODELS}
    improvement = 100.0 * (errors["transolver_tiny"] - errors["raptor"]) / errors["transolver_tiny"]

    rows = []
    for index, name in enumerate(names):
        rows.append({
            "test_index": index,
            "sample": name,
            "family": families[index],
            "transolver_tiny_normalized_mse": float(errors["transolver_tiny"][index]),
            "raptor_normalized_mse": float(errors["raptor"][index]),
            "raptor_relative_improvement_percent": float(improvement[index]),
            "raptor_wins": bool(improvement[index] > 0),
        })

    total_error_reduction = float(np.sum(errors["transolver_tiny"] - errors["raptor"]))
    summary = []
    family_order = list(manifest["families"])
    for family in family_order:
        mask = families == family
        trans = errors["transolver_tiny"][mask]
        raptor = errors["raptor"][mask]
        paired = improvement[mask]
        family_error_reduction = float(np.sum(trans - raptor))
        summary.append({
            "family": family,
            "test_samples": int(mask.sum()),
            "transolver_tiny_mean_normalized_mse": float(trans.mean()),
            "raptor_mean_normalized_mse": float(raptor.mean()),
            "family_mean_mse_improvement_percent": float(100.0 * (trans.mean() - raptor.mean()) / trans.mean()),
            "mean_vehicle_paired_improvement_percent": float(paired.mean()),
            "absolute_error_reduction_sum": family_error_reduction,
            "share_of_overall_error_reduction_percent": float(
                100.0 * family_error_reduction / total_error_reduction
            ),
            "raptor_wins": int(np.sum(paired > 0)),
            "raptor_losses": int(np.sum(paired <= 0)),
        })

    overall = {
        "test_samples": len(names),
        "transolver_tiny_mean_normalized_mse": float(errors["transolver_tiny"].mean()),
        "raptor_mean_normalized_mse": float(errors["raptor"].mean()),
        "overall_mse_improvement_percent": float(
            100.0 * (errors["transolver_tiny"].mean() - errors["raptor"].mean())
            / errors["transolver_tiny"].mean()
        ),
        "mean_vehicle_paired_improvement_percent": float(improvement.mean()),
        "median_vehicle_paired_improvement_percent": float(np.median(improvement)),
        "raptor_wins": int(np.sum(improvement > 0)),
        "raptor_losses": int(np.sum(improvement <= 0)),
        "all_families_improved_by_mean_mse": bool(all(row["family_mean_mse_improvement_percent"] > 0 for row in summary)),
    }
    return rows, summary, overall, errors, improvement, families


def write_outputs(rows: list[dict], summary: list[dict], overall: dict) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for filename, records in (("per_aircraft_metrics.csv", rows), ("family_summary.csv", summary)):
        with (ANALYSIS / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (ANALYSIS / "summary.json").write_text(
        json.dumps({"overall": overall, "by_family": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def equal_3d_axes(ax, xyz: np.ndarray) -> None:
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    center = (low + high) / 2
    radius = max((high - low).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 0.65))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()


def plot_surface_comparison(manifest, predictions, errors, improvement, families) -> None:
    family_order = list(manifest["families"])
    y_mean = float(predictions["raptor"]["y_mean"].reshape(-1)[0])
    y_std = float(predictions["raptor"]["y_std"].reshape(-1)[0])
    y_true = predictions["raptor"]["y_true"] * y_std + y_mean
    pred_t = predictions["transolver_tiny"]["y_pred"] * y_std + y_mean
    pred_r = predictions["raptor"]["y_pred"] * y_std + y_mean
    xyz_all = predictions["raptor"]["x"]

    representatives = []
    for family in family_order:
        indices = np.flatnonzero(families == family)
        family_values = improvement[indices]
        target = np.median(family_values)
        representatives.append(int(indices[np.argmin(np.abs(family_values - target))]))

    fig = plt.figure(figsize=(16.5, 11.2), constrained_layout=True)
    subfigs = fig.subfigures(len(family_order), 1, hspace=0.025)
    column_titles = ("Ground truth", "Transolver-Tiny", "|Error|: Transolver", "RAPTOR", "|Error|: RAPTOR")
    for row_index, (family, sample_index) in enumerate(zip(family_order, representatives)):
        subfig = subfigs[row_index]
        axes = subfig.subplots(1, 5, subplot_kw={"projection": "3d"})
        xyz = xyz_all[sample_index]
        truth = y_true[sample_index, :, 0]
        trans = pred_t[sample_index, :, 0]
        raptor = pred_r[sample_index, :, 0]
        field_min, field_max = np.percentile(truth, [1, 99])
        error_t, error_r = np.abs(trans - truth), np.abs(raptor - truth)
        error_max = max(np.percentile(error_t, 99), np.percentile(error_r, 99), 1e-8)
        panels = (
            (truth, "turbo", Normalize(field_min, field_max)),
            (trans, "turbo", Normalize(field_min, field_max)),
            (error_t, "magma", Normalize(0, error_max)),
            (raptor, "turbo", Normalize(field_min, field_max)),
            (error_r, "magma", Normalize(0, error_max)),
        )
        for col, (ax, (values, cmap, norm)) in enumerate(zip(axes, panels)):
            scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap=cmap, norm=norm,
                                 s=2.7, linewidths=0, rasterized=True)
            equal_3d_axes(ax, xyz)
            if row_index == 0:
                ax.set_title(column_titles[col], fontsize=11, fontweight="semibold", pad=0)
            if col in (1, 3):
                value = errors[MODELS[0] if col == 1 else MODELS[1]][sample_index]
                ax.text2D(0.03, 0.02, f"MSE={value:.3f}", transform=ax.transAxes, fontsize=8,
                          bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2))
            if col in (2, 4):
                cb = subfig.colorbar(scatter, ax=ax, fraction=0.035, pad=-0.01, shrink=0.58)
                cb.ax.tick_params(labelsize=6, width=0.5)
        sample_name = Path(manifest["test"][sample_index]).stem
        subfig.suptitle(
            f"{family}  |  sample {sample_name}  |  RAPTOR paired improvement {improvement[sample_index]:+.1f}%",
            x=0.005, y=0.93, ha="left", fontsize=10.5, fontweight="bold",
        )

# =============================================================================
#     fig.suptitle("Aircraft surface-field reconstruction: ground truth, predictions, and absolute errors",
#                  fontsize=15, fontweight="bold")
# =============================================================================
    for suffix in ("png", "pdf"):
        fig.savefig(ANALYSIS / f"aircraft_truth_prediction_error.{suffix}", dpi=400 if suffix == "png" else 300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_paired_statistics(rows: list[dict], summary: list[dict], overall: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    labels = [f"{row['family']}\n{Path(row['sample']).stem}" for row in rows]
    values = np.asarray([row["raptor_relative_improvement_percent"] for row in rows])
    colors = ["#16837a" if value > 0 else "#d95f59" for value in values]
    axes[0].bar(np.arange(len(rows)), values, color=colors, edgecolor="white", linewidth=0.7)
    axes[0].axhline(0, color="#263238", linewidth=1)
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("RAPTOR paired MSE improvement (%)")
    axes[0].set_title(f"Per-aircraft paired comparison ({overall['raptor_wins']}/{overall['test_samples']} wins)",
                      fontweight="bold")
    axes[0].grid(axis="y", alpha=0.22)

    families = [row["family"] for row in summary]
    trans = np.asarray([row["transolver_tiny_mean_normalized_mse"] for row in summary])
    raptor = np.asarray([row["raptor_mean_normalized_mse"] for row in summary])
    x = np.arange(len(summary)); width = 0.36
    axes[1].bar(x - width / 2, trans, width, label="Transolver-Tiny", color="#7f95b5")
    axes[1].bar(x + width / 2, raptor, width, label="RAPTOR", color="#16837a")
    axes[1].set_xticks(x, families)
    axes[1].set_ylabel("Mean normalized MSE")
    axes[1].set_title("Mean error by aircraft family", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    for i, row in enumerate(summary):
        axes[1].text(i, max(trans[i], raptor[i]) * 1.035,
                     f"{row['family_mean_mse_improvement_percent']:+.1f}%",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle("Does RAPTOR's overall gain generalize across aircraft families?", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(ANALYSIS / f"aircraft_family_paired_statistics.{suffix}",
                    dpi=400 if suffix == "png" else 300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    manifest, predictions = load_inputs()
    rows, summary, overall, errors, improvement, families = compute_statistics(manifest, predictions)
    write_outputs(rows, summary, overall)
    plot_surface_comparison(manifest, predictions, errors, improvement, families)
    plot_paired_statistics(rows, summary, overall)
    print(json.dumps({"overall": overall, "by_family": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
