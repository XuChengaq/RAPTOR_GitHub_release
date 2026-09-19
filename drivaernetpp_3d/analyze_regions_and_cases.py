from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from plot_fullmesh_paper_style import (
    ROOT, HERE, load_full_mesh, draw_mesh, face_values,
)


OUT = ROOT / "analysis"
MODELS = ["pointnet", "transolver_tiny", "raptor"]
REGIONS = ["Front", "Roof", "Side", "Lower / wheel zone", "Rear"]


def region_masks(xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Create disjoint, vehicle-relative regions from coordinate quantiles."""
    x, z = xyz[:, 0], xyz[:, 2]
    x20, x80 = np.quantile(x, [0.20, 0.80])
    z25, z70 = np.quantile(z, [0.25, 0.70])
    front = x <= x20
    rear = x >= x80
    middle = ~(front | rear)
    roof = middle & (z >= z70)
    lower = middle & (z <= z25)
    side = middle & ~(roof | lower)
    return {
        "Front": front,
        "Roof": roof,
        "Side": side,
        "Lower / wheel zone": lower,
        "Rear": rear,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {m: np.load(ROOT / m / "test_predictions.npz") for m in MODELS}
    manifest = json.loads((HERE / "cache" / "split_manifest.json").read_text(encoding="utf-8"))
    truth = arrays["raptor"]["y_true"][:, :, 0]
    per_vehicle = {
        m: np.mean((arrays[m]["y_pred"][:, :, 0] - truth) ** 2, axis=1)
        for m in MODELS
    }
    improvement = (per_vehicle["transolver_tiny"] - per_vehicle["raptor"]) \
                  / per_vehicle["transolver_tiny"] * 100

    region_errors = {m: np.zeros((len(truth), len(REGIONS))) for m in MODELS}
    region_counts = np.zeros((len(truth), len(REGIONS)), dtype=int)
    for i in range(len(truth)):
        masks = region_masks(arrays["raptor"]["x"][i])
        for j, region in enumerate(REGIONS):
            mask = masks[region]
            region_counts[i, j] = int(mask.sum())
            for m in MODELS:
                residual = arrays[m]["y_pred"][i, :, 0] - truth[i]
                region_errors[m][i, j] = float(np.mean(residual[mask] ** 2))

    with (OUT / "regional_normalized_mse.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_vehicle_index", "source_member", "region", "node_count", *MODELS])
        for i, member in enumerate(manifest["test"]):
            for j, region in enumerate(REGIONS):
                writer.writerow([i, member, region, region_counts[i, j],
                                 *[f"{region_errors[m][i, j]:.10f}" for m in MODELS]])

    order = np.argsort(improvement)
    worst = int(order[0])
    best = int(order[-1])
    median = int(np.argmin(np.abs(improvement - np.median(improvement))))
    cases = [(best, "Largest RAPTOR gain"), (median, "Median RAPTOR gain"),
             (worst, "RAPTOR regression case")]

    mean_regions = {m: region_errors[m].mean(axis=0) for m in MODELS}
    region_improvement = (mean_regions["transolver_tiny"] - mean_regions["raptor"]) \
                         / mean_regions["transolver_tiny"] * 100
    report = {
        "region_definition": {
            "Front": "lowest 20% of longitudinal x coordinates",
            "Rear": "highest 20% of longitudinal x coordinates",
            "Roof": "middle 60% in x and highest 30% in z",
            "Lower / wheel zone": "middle 60% in x and lowest 25% in z",
            "Side": "remaining middle-body nodes",
        },
        "mean_region_normalized_mse": {m: dict(zip(REGIONS, map(float, mean_regions[m]))) for m in MODELS},
        "raptor_improvement_over_transolver_tiny_percent": dict(zip(REGIONS, map(float, region_improvement))),
        "representative_cases": [
            {"type": label, "test_index": i, "source_member": manifest["test"][i],
             "raptor_improvement_percent": float(improvement[i]),
             "transolver_mse": float(per_vehicle["transolver_tiny"][i]),
             "raptor_mse": float(per_vehicle["raptor"][i])}
            for i, label in cases
        ],
    }
    (OUT / "regional_and_case_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), constrained_layout=True)
    x = np.arange(len(REGIONS))
    width = 0.24
    colors = {"pointnet": "#d4a72c", "transolver_tiny": "#52789e", "raptor": "#087f8c"}
    for k, m in enumerate(MODELS):
        axes[0].bar(x + (k - 1) * width, mean_regions[m], width=width,
                    color=colors[m], label={"pointnet":"PointNet", "transolver_tiny":"Transolver-Tiny", "raptor":"RAPTOR"}[m])
    axes[0].set_xticks(x, ["Front", "Roof", "Side", "Lower /\nwheel zone", "Rear"])
    axes[0].set_ylabel("Regional sMSE ↓")
    axes[0].set_title("(a) Geometry-region reconstruction error", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Okabe--Ito inspired colors: blue for improvement and orange for degradation.
    bar_colors = ["#0072B2" if v >= 0 else "#D55E00" for v in region_improvement]
    bars = axes[1].barh(x, region_improvement, color=bar_colors, height=0.58)
    axes[1].axvline(0, color="#4B5563", linestyle="--", linewidth=1)
    axes[1].set_yticks(x, REGIONS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RAPTOR improvement over Transolver-Tiny (%) →")
    axes[1].set_title("(b) Local-adapter contribution by region", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].set_xlim(-25, 26)
    for bar, value in zip(bars, region_improvement):
        y_text = bar.get_y() + bar.get_height() / 2
        if abs(value) >= 5:
            # Put labels inside sufficiently long bars using high-contrast white text.
            x_text = value - 0.8 if value > 0 else value + 0.8
            ha = "right" if value > 0 else "left"
            color = "white"
        else:
            # Move labels outside short bars so that the zero line does not obscure them.
            x_text = value + 0.7 if value > 0 else value - 0.7
            ha = "left" if value > 0 else "right"
            color = "#111827"
        axes[1].text(x_text, y_text, f"{value:+.1f}%", va="center", ha=ha,
                     fontsize=9, fontweight="bold", color=color)
# =============================================================================
#     fig.suptitle("RAPTOR 3D Surface Reconstruction: Regional Error and Adapter Gain",
#                  fontsize=15, weight="bold")
# =============================================================================
    region_png = OUT / "Figure_100car_regional_adapter_analysis.png"
    fig.savefig(region_png, dpi=350, bbox_inches="tight")
    fig.savefig(region_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(17.5, 10.2), facecolor="white")
    gs = fig.add_gridspec(3, 5, left=0.015, right=0.985, bottom=0.08, top=0.91,
                          wspace=0.008, hspace=0.12)
    row_cache = []
    all_fields, all_errors = [], []
    for index, label in cases:
        full = load_full_mesh(manifest["test"][index])
        xyz = np.asarray(full["nodes_list"], dtype=float)
        faces = np.asarray(full["elems_list"][:, 1:4], dtype=int)
        full_truth = np.asarray(full["features_list"], dtype=float).reshape(-1)
        sampled_xyz = arrays["raptor"]["x"][index]
        nearest = cKDTree(sampled_xyz).query(xyz, k=1)[1]
        mean = float(arrays["raptor"]["y_mean"].reshape(-1)[0])
        std = float(arrays["raptor"]["y_std"].reshape(-1)[0])
        trans = (arrays["transolver_tiny"]["y_pred"][index, :, 0] * std + mean)[nearest]
        raptor = (arrays["raptor"]["y_pred"][index, :, 0] * std + mean)[nearest]
        terr, rerr = np.abs(trans - full_truth), np.abs(raptor - full_truth)
        row_cache.append((index, label, xyz, faces, full_truth, trans, raptor, terr, rerr))
        all_fields.extend([full_truth, trans, raptor])
        all_errors.extend([terr, rerr])
    field_norm = Normalize(*np.quantile(np.concatenate(all_fields), [0.01, 0.99]), clip=True)
    error_norm = Normalize(0, float(np.quantile(np.concatenate(all_errors), 0.99)), clip=True)
    for row, (index, label, xyz, faces, full_truth, trans, raptor, terr, rerr) in enumerate(row_cache):
        panels = [
            ("Ground truth", face_values(full_truth, faces), "coolwarm", field_norm),
            ("Transolver-Tiny", face_values(trans, faces), "coolwarm", field_norm),
            ("RAPTOR", face_values(raptor, faces), "coolwarm", field_norm),
            ("Transolver |error|", face_values(terr, faces), "Reds", error_norm),
            ("RAPTOR |error|", face_values(rerr, faces), "Reds", error_norm),
        ]
        for col, (title, values, cmap, norm) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            draw_mesh(ax, xyz, faces, values, cmap, norm)
            ax.set_title(title, fontsize=9.5, weight="bold", pad=-6)
        fig.text(0.012, 0.82 - row * 0.275,
                 f"{label}\nvehicle {index + 1}\nΔ={improvement[index]:+.1f}%",
                 fontsize=9, weight="bold", color="#334155", va="center")
    cax1 = fig.add_axes([0.24, 0.94, 0.28, 0.012])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=field_norm, cmap="coolwarm"), cax=cax1, orientation="horizontal")
    cb1.set_label("Surface field", labelpad=1); cb1.ax.xaxis.set_label_position("top"); cb1.ax.tick_params(labelsize=7)
    cax2 = fig.add_axes([0.62, 0.94, 0.22, 0.012])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=error_norm, cmap="Reds"), cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute error", labelpad=1); cb2.ax.xaxis.set_label_position("top"); cb2.ax.tick_params(labelsize=7)
    fig.suptitle("Representative 3D Test Vehicles: Best, Median, and Regression Cases",
                 y=0.992, fontsize=16, weight="bold")
    fig.text(0.015, 0.015,
             "Full-mesh colors use visualization-only nearest-neighbor interpolation from the original 4,096 sampled prediction nodes; all numerical comparisons remain on those nodes.",
             fontsize=8, color="#475569")
    cases_png = OUT / "Figure_100car_best_median_regression_fullmesh.png"
    fig.savefig(cases_png, dpi=320, bbox_inches="tight")
    fig.savefig(cases_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(region_png)
    print(cases_png)


if __name__ == "__main__":
    main()
