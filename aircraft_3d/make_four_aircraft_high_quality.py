from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "outputs" / "formal_100aircraft" / "analysis"

SAMPLES = (
    ("J20 0037", "j20_0037_full_surface_fields.npz", 0.7629244, 0.7150293),
    ("P180 0747", "p180_0747_full_surface_fields.npz", 0.1875787, 0.2008434),
    ("B737 0648", "boeing737_0648_full_surface_fields.npz", 1.3861819, 1.1499671),
    ("ERJ 0695", "erj_0695_full_surface_fields.npz", 0.1200628, 0.1100475),
)


def face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = nodes[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def shaded_colors(values: np.ndarray, norm, cmap_name: str, normals: np.ndarray,
                  strength: float = 0.16) -> np.ndarray:
    colors = plt.get_cmap(cmap_name)(norm(values))
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    intensity = np.clip(np.abs(normals @ light), 0, 1)
    colors[:, :3] *= ((1.0 - strength) + strength * intensity)[:, None]
    return colors


def set_camera(ax, nodes: np.ndarray) -> None:
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    center = (low + high) / 2
    radius = max(high - low) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.46, center[2] + radius * 0.46)
    ax.set_box_aspect((2.25, 1.25, 0.55))
    ax.view_init(elev=24, azim=-56)
    ax.set_axis_off()


def add_surface(ax, nodes: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    surface = Poly3DCollection(
        nodes[faces], facecolors=colors, edgecolors="none", linewidths=0,
        antialiased=True, rasterized=True,
    )
    surface.set_zsort("average")
    ax.add_collection3d(surface)
    set_camera(ax, nodes)


def main() -> None:
    datasets = []
    for family, filename, mse_t, mse_r in SAMPLES:
        with np.load(ANALYSIS / filename) as payload:
            item = {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}
        item.update(family=family, mse_t=mse_t, mse_r=mse_r)
        datasets.append(item)

    all_truth = np.concatenate([item["ground_truth"] for item in datasets])
    all_errors = np.concatenate([
        item["transolver_abs_error"] for item in datasets
    ] + [
        item["raptor_abs_error"] for item in datasets
    ])
    field_lo, field_hi = np.percentile(all_truth, [1.0, 99.0])
    error_hi = max(float(np.percentile(all_errors, 97.5)), 1e-12)
    field_norm = Normalize(field_lo, field_hi, clip=True)
    error_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=error_hi, clip=True)

    fig = plt.figure(figsize=(18.0, 12.0), facecolor="white")
    grid = fig.add_gridspec(
        4, 5, left=0.070, right=0.990, top=0.955, bottom=0.105,
        wspace=0.008, hspace=0.040,
    )
    column_titles = (
        "Ground\ntruth", "Transolver-Tiny\nprediction",
        "Transolver-Tiny\nabsolute error", "RAPTOR\nprediction",
        "RAPTOR\nabsolute error",
    )

    for row, item in enumerate(datasets):
        nodes = item["nodes"]
        faces = item["faces"].astype(np.int64)
        normals = face_normals(nodes, faces)
        face_truth = item["ground_truth"][faces].mean(axis=1)
        face_t = item["transolver_tiny"][faces].mean(axis=1)
        face_r = item["raptor"][faces].mean(axis=1)
        face_et = item["transolver_abs_error"][faces].mean(axis=1)
        face_er = item["raptor_abs_error"][faces].mean(axis=1)

        panels = (
            shaded_colors(face_truth, field_norm, "turbo", normals),
            shaded_colors(face_t, field_norm, "turbo", normals),
            shaded_colors(face_et, error_norm, "turbo", normals, 0.05),
            shaded_colors(face_r, field_norm, "turbo", normals),
            shaded_colors(face_er, error_norm, "turbo", normals, 0.05),
        )

        axes = []
        for col, colors in enumerate(panels):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            axes.append(ax)
            add_surface(ax, nodes, faces, colors)
            if row == 0:
                ax.set_title(column_titles[col], fontsize=17.0, fontweight="bold", pad=-2)

        improvement = 100.0 * (item["mse_t"] - item["mse_r"]) / item["mse_t"]
        family_color = "#16837a" if improvement >= 0 else "#c74b50"
        axes[0].text2D(
            -0.11, 0.50, item["family"], transform=axes[0].transAxes,
            rotation=90, va="center", ha="center", fontsize=15.0,
            fontweight="bold", color="#17223b",
        )
        axes[1].text2D(
            0.03, 0.035, f"sMSE = {item['mse_t']:.3f}", transform=axes[1].transAxes,
            fontsize=12.5, bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                    alpha=0.90, edgecolor="#78909c"),
        )
        axes[3].text2D(
            0.03, 0.035, f"sMSE = {item['mse_r']:.3f}  ({improvement:+.1f}%)",
            transform=axes[3].transAxes, fontsize=12.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.92,
                      edgecolor=family_color), color=family_color,
        )

    field_map = ScalarMappable(norm=field_norm, cmap="turbo")
    error_map = ScalarMappable(norm=error_norm, cmap="turbo")
    field_cax = fig.add_axes([0.18, 0.050, 0.29, 0.018])
    error_cax = fig.add_axes([0.56, 0.050, 0.29, 0.018])
    field_cb = fig.colorbar(field_map, cax=field_cax, orientation="horizontal")
    error_cb = fig.colorbar(error_map, cax=error_cax, orientation="horizontal")
    field_cb.set_label(r"$C_p$ (shared scale across all aircraft)", fontsize=12)
    error_cb.set_label("Absolute error (shared power scale; 97.5th-percentile cap)", fontsize=12)
    error_cb.set_ticks(np.linspace(0.0, error_hi, 5))
    for colorbar in (field_cb, error_cb):
        colorbar.ax.tick_params(labelsize=10.5, width=0.6)
    for suffix in ("png", "pdf"):
        fig.savefig(
            ANALYSIS / f"four_aircraft_truth_prediction_error_high_quality.{suffix}",
            dpi=450 if suffix == "png" else 300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
