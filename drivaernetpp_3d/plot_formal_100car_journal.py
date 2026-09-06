from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs" / "formal_100car"
OUT = ROOT / "visualizations"
MODELS = ["pointnet", "transolver_tiny", "raptor", "no_adapter"]
LABELS = {
    "pointnet": "PointNet",
    "transolver_tiny": "Transolver-Tiny",
    "raptor": "RAPTOR",
    "no_adapter": "RAPTOR w/o adapter",
}


def equal_3d(ax, xyz: np.ndarray) -> None:
    center = (xyz.min(0) + xyz.max(0)) / 2
    radius = (xyz.max(0) - xyz.min(0)).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.9, 1.0, 0.72))
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-132)
    ax.set_axis_off()


def surface(ax, xyz, values, cmap, norm) -> None:
    order = np.argsort(xyz[:, 0] + 0.25 * xyz[:, 2])
    ax.scatter(
        xyz[order, 0], xyz[order, 1], xyz[order, 2],
        c=values[order], cmap=cmap, norm=norm, s=5.0,
        linewidths=0, rasterized=True, depthshade=False,
    )
    equal_3d(ax, xyz)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.load(ROOT / name / "test_predictions.npz") for name in MODELS}
    results = {
        name: json.loads((ROOT / name / "result.json").read_text(encoding="utf-8"))
        for name in MODELS
    }
    # Representative, non-extreme vehicle: strong but credible RAPTOR improvement.
    sample = 1
    xyz = arrays["raptor"]["x"][sample]
    mean = float(arrays["raptor"]["y_mean"].reshape(-1)[0])
    std = float(arrays["raptor"]["y_std"].reshape(-1)[0])
    truth = arrays["raptor"]["y_true"][sample, :, 0] * std + mean
    predictions = {
        name: arrays[name]["y_pred"][sample, :, 0] * std + mean for name in MODELS
    }
    errors = {name: np.abs(predictions[name] - truth) for name in MODELS}
    sample_mse = {
        name: float(np.mean((arrays[name]["y_pred"][sample, :, 0]
                            - arrays[name]["y_true"][sample, :, 0]) ** 2))
        for name in MODELS
    }

    field_values = np.concatenate([truth, *predictions.values()])
    vmin, vmax = np.quantile(field_values, [0.01, 0.99])
    error_max = float(np.quantile(np.concatenate(list(errors.values())), 0.99))
    field_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    error_norm = Normalize(vmin=0.0, vmax=error_max, clip=True)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(18.2, 8.2), constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 5, left=0.025, right=0.975, bottom=0.135, top=0.845,
                          wspace=0.025, hspace=0.16)

    top_titles = ["Ground truth"] + [LABELS[n] for n in MODELS]
    top_values = [truth] + [predictions[n] for n in MODELS]
    top_axes = []
    for col, (title, values) in enumerate(zip(top_titles, top_values)):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        surface(ax, xyz, values, "coolwarm", field_norm)
        marker = chr(ord("a") + col)
        ax.set_title(f"({marker})  {title}", pad=-2, fontsize=11)
        top_axes.append(ax)

    axm = fig.add_subplot(gs[1, 0])
    ordered = ["raptor", "transolver_tiny", "no_adapter", "pointnet"]
    values = [results[n]["test_normalized_mse"] for n in ordered]
    colors = ["#087f8c", "#577590", "#90a4ae", "#d9a441"]
    bars = axm.barh(np.arange(4), values, color=colors, height=0.55)
    axm.set_yticks(np.arange(4), [LABELS[n] for n in ordered])
    axm.invert_yaxis()
    axm.set_xlabel("Test normalized MSE  ↓")
    axm.set_title("(f)  Overall test accuracy", loc="left", fontsize=11)
    axm.grid(axis="x", color="#d8dde3", linewidth=0.6)
    axm.set_axisbelow(True)
    axm.spines[["top", "right", "left"]].set_visible(False)
    axm.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        axm.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value:.3f}",
                 va="center", ha="left", fontsize=9)
    axm.set_xlim(0, max(values) * 1.22)
    improvement = (results["transolver_tiny"]["test_normalized_mse"]
                   - results["raptor"]["test_normalized_mse"]) \
                  / results["transolver_tiny"]["test_normalized_mse"] * 100
    axm.text(0.03, 0.04, f"RAPTOR: {improvement:.2f}% lower\nthan Transolver-Tiny",
             transform=axm.transAxes, color="#087f8c", fontsize=9,
             weight="semibold", va="bottom")

    error_axes = []
    for col, name in enumerate(MODELS, start=1):
        ax = fig.add_subplot(gs[1, col], projection="3d")
        surface(ax, xyz, errors[name], "magma", error_norm)
        marker = chr(ord("g") + col - 1)
        ax.set_title(f"({marker})  |Error| · {LABELS[name]}\nvehicle MSE = {sample_mse[name]:.3f}",
                     pad=-2, fontsize=10.5)
        error_axes.append(ax)

    cax1 = fig.add_axes([0.335, 0.885, 0.33, 0.014])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=field_norm, cmap="coolwarm"),
                       cax=cax1, orientation="horizontal")
    cb1.set_label("Single-channel surface field (dataset units; shared 1–99% scale)", labelpad=3)
    cb1.ax.xaxis.set_label_position("top")
    cb1.ax.tick_params(labelsize=8, pad=1)
    cax2 = fig.add_axes([0.445, 0.075, 0.43, 0.014])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=error_norm, cmap="magma"),
                       cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute error (shared 0–99% scale)", labelpad=2)
    cb2.ax.tick_params(labelsize=8, pad=1)

    fig.suptitle("100-Vehicle Surface-Field Reconstruction: Prediction Fidelity and Error Localization",
                 y=0.992, fontsize=17, weight="semibold")
    fig.text(0.5, 0.952,
             "Identical 80/10/10 split · 4,096 surface nodes per vehicle · seed 42 · representative test vehicle",
             ha="center", va="center", fontsize=10, color="#4c5967")
    fig.text(0.025, 0.018,
             "All models use the same sampled coordinates and target normalization. Field and error colors are shared across panels.",
             ha="left", fontsize=8.5, color="#4c5967")

    png = OUT / "Figure_RAPTOR3D_100car_journal_comparison.png"
    pdf = OUT / "Figure_RAPTOR3D_100car_journal_comparison.pdf"
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "sample_index_within_test_split": sample,
        "field_scale_1_99_percent": [float(vmin), float(vmax)],
        "error_scale_99_percent": error_max,
        "sample_normalized_mse": sample_mse,
        "overall_test_normalized_mse": {n: results[n]["test_normalized_mse"] for n in MODELS},
        "raptor_improvement_over_transolver_tiny_percent": improvement,
    }
    (OUT / "Figure_RAPTOR3D_100car_journal_comparison.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
