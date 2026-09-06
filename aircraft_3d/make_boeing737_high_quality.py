from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "drivaernetpp_3d"))
from audit_dataset import SliceReader, nested_data_offset


ARCHIVE = HERE.parents[1] / "DrivAerNetPP_100" / "mixed_3d.zip"
OUTPUT = HERE / "outputs" / "formal_100aircraft"
ANALYSIS = OUTPUT / "analysis"
def read_original_sample(sample: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = zipfile.ZipFile(ARCHIVE)
    nested = next(item for item in outer.infolist() if item.filename.lower().endswith(".zip"))
    start = nested_data_offset(ARCHIVE, nested)
    outer.close()
    raw = SliceReader(ARCHIVE, start, nested.file_size)
    buffered = io.BufferedReader(raw)
    inner = zipfile.ZipFile(buffered)
    try:
        with inner.open(sample) as handle:
            payload = np.load(io.BytesIO(handle.read()), allow_pickle=False)
            nodes = np.asarray(payload["nodes_list"], dtype=np.float64)
            faces = np.asarray(payload["elems_list"], dtype=np.int64)
            truth = np.asarray(payload["features_list"], dtype=np.float64).reshape(-1)
    finally:
        inner.close()
        buffered.close()
        raw.close()
    if not np.all(faces[:, 0] == 3):
        raise ValueError("expected VTK-style triangular faces [3, i, j, k]")
    return nodes, faces[:, 1:4], truth


def load_prediction(model: str, test_index: int) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(OUTPUT / model / "test_predictions.npz") as payload:
        sampled_nodes = payload["x"][test_index].astype(np.float64)
        mean = float(payload["y_mean"].reshape(-1)[0])
        std = float(payload["y_std"].reshape(-1)[0])
        prediction = payload["y_pred"][test_index, :, 0].astype(np.float64) * std + mean
        normalized_truth = payload["y_true"][test_index, :, 0].astype(np.float64)
        normalized_prediction = payload["y_pred"][test_index, :, 0].astype(np.float64)
        mse = float(np.mean((normalized_prediction - normalized_truth) ** 2))
    return sampled_nodes, prediction, mse


def interpolate_to_full(full_nodes: np.ndarray, sampled_nodes: np.ndarray,
                        sampled_values: np.ndarray, neighbors: int = 4) -> np.ndarray:
    distances, indices = cKDTree(sampled_nodes).query(full_nodes, k=neighbors, workers=-1)
    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / distances**2
    interpolated = np.sum(weights * sampled_values[indices], axis=1) / np.sum(weights, axis=1)
    exact = distances[:, 0] < 1e-10
    interpolated[exact] = sampled_values[indices[exact, 0]]
    return interpolated


def face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = nodes[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def shaded_colors(values: np.ndarray, norm: Normalize, cmap_name: str,
                  normals: np.ndarray, strength: float = 0.18) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(values))
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    intensity = np.clip(np.abs(normals @ light), 0, 1)
    shade = (1.0 - strength) + strength * intensity
    colors[:, :3] *= shade[:, None]
    return colors


def set_camera(ax, nodes: np.ndarray) -> None:
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    center = (low + high) / 2
    span = high - low
    radius = max(span[0], span[1], span[2]) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.46, center[2] + radius * 0.46)
    ax.set_box_aspect((2.25, 1.25, 0.55))
    ax.view_init(elev=24, azim=-56)
    ax.set_axis_off()


def add_surface(ax, nodes, faces, face_colors, rasterized=True) -> None:
    collection = Poly3DCollection(nodes[faces], facecolors=face_colors, edgecolors="none",
                                  linewidths=0, antialiased=True, rasterized=rasterized)
    collection.set_zsort("average")
    ax.add_collection3d(collection)
    set_camera(ax, nodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-quality full-surface aircraft comparison")
    parser.add_argument("--sample", default="Plane/boeing737/0648.npz")
    parser.add_argument("--test-index", type=int, default=6)
    parser.add_argument("--display-name", default="Boeing 737")
    parser.add_argument("--slug", default="boeing737")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    nodes, faces, truth = read_original_sample(args.sample)
    sampled_t, pred_t_sampled, mse_t = load_prediction("transolver_tiny", args.test_index)
    sampled_r, pred_r_sampled, mse_r = load_prediction("raptor", args.test_index)
    if not np.allclose(sampled_t, sampled_r):
        raise RuntimeError("model prediction coordinates do not match")

    pred_t = interpolate_to_full(nodes, sampled_t, pred_t_sampled)
    pred_r = interpolate_to_full(nodes, sampled_r, pred_r_sampled)
    error_t = np.abs(pred_t - truth)
    error_r = np.abs(pred_r - truth)
    normals = face_normals(nodes, faces)

    field_lo, field_hi = np.percentile(truth, [1, 99])
    # A shared robust upper bound prevents a few extreme vertices from
    # compressing nearly the entire error surface into the darkest colors.
    error_hi = max(np.percentile(error_t, 97.5), np.percentile(error_r, 97.5), 1e-12)
    field_norm = Normalize(field_lo, field_hi)
    # Power normalization expands low-to-moderate errors while preserving one
    # common scale for the two models. Values above error_hi are saturated.
    error_norm = PowerNorm(gamma=0.45, vmin=0, vmax=error_hi, clip=True)
    face_truth = truth[faces].mean(axis=1)
    face_pred_t = pred_t[faces].mean(axis=1)
    face_pred_r = pred_r[faces].mean(axis=1)
    face_error_t = error_t[faces].mean(axis=1)
    face_error_r = error_r[faces].mean(axis=1)

    geometry_light = np.clip(np.abs(normals @ np.array([-0.35, -0.45, 0.82])), 0, 1)
    geometry_gray = 0.55 + 0.35 * geometry_light
    geometry_colors = np.column_stack([geometry_gray, geometry_gray, geometry_gray,
                                       np.ones_like(geometry_gray)])

    fig = plt.figure(figsize=(15.8, 8.6), facecolor="white")
    fig.subplots_adjust(left=0.018, right=0.982, top=0.91, bottom=0.13,
                        wspace=0.015, hspace=0.07)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    panels = (
        ("(a) Input surface mesh", geometry_colors),
        ("(b) Ground truth", shaded_colors(face_truth, field_norm, "turbo", normals)),
        ("(c) Transolver-Tiny prediction", shaded_colors(face_pred_t, field_norm, "turbo", normals)),
        ("(d) Transolver-Tiny absolute error", shaded_colors(face_error_t, error_norm, "turbo", normals, 0.06)),
        ("(e) RAPTOR prediction", shaded_colors(face_pred_r, field_norm, "turbo", normals)),
        ("(f) RAPTOR absolute error", shaded_colors(face_error_r, error_norm, "turbo", normals, 0.06)),
    )
    for ax, (title, colors) in zip(axes, panels):
        add_surface(ax, nodes, faces, colors)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=-4)

    axes[2].text2D(0.04, 0.03, f"Normalized MSE = {mse_t:.3f}", transform=axes[2].transAxes,
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88,
                                        edgecolor="#78909c"))
    improvement = 100.0 * (mse_t - mse_r) / mse_t
    axes[4].text2D(0.04, 0.03, f"Normalized MSE = {mse_r:.3f}  ({improvement:+.1f}%)",
                   transform=axes[4].transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5f6f3", alpha=0.92,
                             edgecolor="#16837a"))

    field_map = ScalarMappable(norm=field_norm, cmap="turbo")
    error_map = ScalarMappable(norm=error_norm, cmap="turbo")
    field_cax = fig.add_axes([0.18, 0.062, 0.29, 0.020])
    error_cax = fig.add_axes([0.56, 0.062, 0.29, 0.020])
    field_cb = fig.colorbar(field_map, cax=field_cax, orientation="horizontal")
    field_cb.set_label("Surface-field value (shared scale)", fontsize=9)
    error_cb = fig.colorbar(error_map, cax=error_cax, orientation="horizontal")
    error_cb.set_label("Absolute error (shared power scale; 97.5th-percentile cap)", fontsize=9)
    error_cb.locator = MaxNLocator(nbins=6)
    error_cb.update_ticks()
    for colorbar in (field_cb, error_cb):
        colorbar.ax.tick_params(labelsize=8, width=0.6)

    sample_id = Path(args.sample).stem
    fig.suptitle(f"{args.display_name} surface-field reconstruction on test sample {sample_id}",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.012,
             "Original 27,192-node / 54,371-triangle surface; predictions interpolated from the matched 4,096-point protocol.",
             ha="center", fontsize=8.5, color="#455a64")
    for suffix in ("png", "pdf"):
        fig.savefig(ANALYSIS / f"{args.slug}_{sample_id}_high_quality.{suffix}",
                    dpi=500 if suffix == "png" else 300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        ANALYSIS / f"{args.slug}_{sample_id}_full_surface_fields.npz",
        nodes=nodes.astype(np.float32), faces=faces.astype(np.int32),
        ground_truth=truth.astype(np.float32), transolver_tiny=pred_t.astype(np.float32),
        raptor=pred_r.astype(np.float32), transolver_abs_error=error_t.astype(np.float32),
        raptor_abs_error=error_r.astype(np.float32),
    )
    print({"sample": args.sample, "nodes": len(nodes), "triangles": len(faces),
           "transolver_mse": mse_t, "raptor_mse": mse_r,
           "raptor_improvement_percent": improvement})


if __name__ == "__main__":
    main()
