from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm

from make_boeing737_high_quality import (
    ANALYSIS,
    OUTPUT,
    add_surface,
    face_normals,
    interpolate_to_full,
    load_prediction,
    read_original_sample,
    shaded_colors,
)


def load_extreme_cases() -> list[dict[str, object]]:
    metrics_path = ANALYSIS / "per_aircraft_metrics.csv"
    manifest_path = Path(__file__).resolve().parent / "cache" / "split_manifest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run analyze_aircraft_results.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("per_aircraft_metrics.csv is empty")

    for row in rows:
        row["test_index"] = int(row["test_index"])
        row["raptor_relative_improvement_percent"] = float(
            row["raptor_relative_improvement_percent"]
        )
    best = max(rows, key=lambda row: row["raptor_relative_improvement_percent"])
    worst = min(rows, key=lambda row: row["raptor_relative_improvement_percent"])

    display_names = {
        "J20": "J-20",
        "P180": "P180",
        "boeing737": "Boeing 737",
        "erj": "ERJ",
    }
    for label, row in (("Best case", best), ("Worst case", worst)):
        test_index = int(row["test_index"])
        expected_sample = manifest["test"][test_index]
        if expected_sample.replace("\\", "/") != str(row["sample"]).replace("\\", "/"):
            raise RuntimeError(f"Manifest/metrics mismatch at test index {test_index}")
        row["case_label"] = label
        row["display_name"] = display_names.get(str(row["family"]), str(row["family"]))
    return [best, worst]


def prepare_case(row: dict[str, object]) -> dict[str, object]:
    sample = str(row["sample"])
    test_index = int(row["test_index"])
    nodes, faces, truth = read_original_sample(sample)
    sampled_t, pred_t_sampled, mse_t = load_prediction("transolver_tiny", test_index)
    sampled_r, pred_r_sampled, mse_r = load_prediction("raptor", test_index)
    if not np.allclose(sampled_t, sampled_r):
        raise RuntimeError(f"Prediction coordinates do not match for {sample}")

    pred_t = interpolate_to_full(nodes, sampled_t, pred_t_sampled)
    pred_r = interpolate_to_full(nodes, sampled_r, pred_r_sampled)
    error_t = np.abs(pred_t - truth)
    error_r = np.abs(pred_r - truth)
    normals = face_normals(nodes, faces)

    field_lo, field_hi = np.percentile(truth, [1, 99])
    error_hi = max(np.percentile(error_t, 97.5), np.percentile(error_r, 97.5), 1e-12)
    field_norm = Normalize(field_lo, field_hi)
    error_norm = PowerNorm(gamma=0.45, vmin=0, vmax=error_hi, clip=True)

    geometry_light = np.clip(
        np.abs(normals @ np.array([-0.35, -0.45, 0.82])), 0, 1
    )
    geometry_gray = 0.55 + 0.35 * geometry_light
    geometry_colors = np.column_stack(
        [geometry_gray, geometry_gray, geometry_gray, np.ones_like(geometry_gray)]
    )
    face_values = {
        "mesh": geometry_colors,
        "truth": shaded_colors(truth[faces].mean(axis=1), field_norm, "turbo", normals),
        "trans": shaded_colors(pred_t[faces].mean(axis=1), field_norm, "turbo", normals),
        "trans_error": shaded_colors(
            error_t[faces].mean(axis=1), error_norm, "turbo", normals, 0.06
        ),
        "raptor": shaded_colors(pred_r[faces].mean(axis=1), field_norm, "turbo", normals),
        "raptor_error": shaded_colors(
            error_r[faces].mean(axis=1), error_norm, "turbo", normals, 0.06
        ),
    }
    return {
        **row,
        "nodes": nodes,
        "faces": faces,
        "face_values": face_values,
        "field_norm": field_norm,
        "error_norm": error_norm,
        "mse_t": mse_t,
        "mse_r": mse_r,
        "improvement": 100.0 * (mse_t - mse_r) / mse_t,
    }


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cases = [prepare_case(row) for row in load_extreme_cases()]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(19.2, 7.8), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.875, bottom=0.145,
                        wspace=0.005, hspace=0.15)
    axes = np.asarray([
        [fig.add_subplot(2, 6, row * 6 + col + 1, projection="3d") for col in range(6)]
        for row in range(2)
    ])
    titles = (
        "Input surface mesh",
        "Ground truth",
        "Transolver-Tiny prediction",
        "Transolver-Tiny |error|",
        "RAPTOR prediction",
        "RAPTOR |error|",
    )
    keys = ("mesh", "truth", "trans", "trans_error", "raptor", "raptor_error")

    for row_index, case in enumerate(cases):
        for col, (title, key) in enumerate(zip(titles, keys)):
            ax = axes[row_index, col]
            add_surface(
                ax,
                case["nodes"],
                case["faces"],
                case["face_values"][key],
            )
            if row_index == 0:
                ax.set_title(f"({chr(97 + col)}) {title}", fontsize=10.8, pad=-3)

        axes[row_index, 0].text2D(
            0.01,
            0.92,
            f"{case['case_label']}: {case['display_name']} / "
            f"sample {Path(str(case['sample'])).stem}",
            transform=axes[row_index, 0].transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
        axes[row_index, 2].text2D(
            0.04,
            0.02,
            f"MSE = {case['mse_t']:.3f}",
            transform=axes[row_index, 2].transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.90,
                      edgecolor="#78909c"),
        )
        gain_color = "#16837a" if case["improvement"] >= 0 else "#c74440"
        gain_fill = "#e5f6f3" if case["improvement"] >= 0 else "#fdecea"
        axes[row_index, 4].text2D(
            0.04,
            0.02,
            f"MSE = {case['mse_r']:.3f}  ({case['improvement']:+.1f}%)",
            transform=axes[row_index, 4].transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=gain_fill, alpha=0.94,
                      edgecolor=gain_color),
        )

        # Compact row-specific legends preserve detail while maintaining fair
        # Transolver/RAPTOR comparisons within each aircraft.
        y = 0.515 if row_index == 0 else 0.078
        field_cax = fig.add_axes([0.205, y, 0.245, 0.014])
        error_cax = fig.add_axes([0.700, y, 0.245, 0.014])
        field_cb = fig.colorbar(
            ScalarMappable(norm=case["field_norm"], cmap="turbo"),
            cax=field_cax,
            orientation="horizontal",
        )
        error_cb = fig.colorbar(
            ScalarMappable(norm=case["error_norm"], cmap="turbo"),
            cax=error_cax,
            orientation="horizontal",
        )
        field_cb.set_label("Surface-field value (row-wise shared scale)", fontsize=7.7)
        error_cb.set_label("Absolute error (row-wise shared power scale)", fontsize=7.7)
        for colorbar in (field_cb, error_cb):
            colorbar.ax.tick_params(labelsize=6.8, width=0.5, pad=1)

# =============================================================================
#     fig.suptitle(
#         "Best- and Worst-Case Aircraft Surface-Field Reconstruction",
#         fontsize=17,
#         fontweight="bold",
#         y=0.965,
#     )
#     fig.text(
#         0.5,
#         0.008,
#         "Cases are selected automatically from the held-out test set by paired MSE improvement; "
#         "full-mesh colors are interpolated from the matched 4,096-point seed-42 predictions.",
#         ha="center",
#         fontsize=8.3,
#         color="#455a64",
#     )
# =============================================================================

    stem = "aircraft_best_worst_2x6"
    for suffix in ("png", "pdf"):
        fig.savefig(
            ANALYSIS / f"{stem}.{suffix}",
            dpi=500 if suffix == "png" else 300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print({
        "output": str(ANALYSIS / stem),
        "best": {"sample": cases[0]["sample"], "improvement": cases[0]["improvement"]},
        "worst": {"sample": cases[1]["sample"], "improvement": cases[1]["improvement"]},
    })


if __name__ == "__main__":
    main()
