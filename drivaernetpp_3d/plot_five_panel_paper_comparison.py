from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from plot_fullmesh_paper_style import (
    HERE, ROOT, OUT, load_full_mesh, face_values, lit_facecolors, style_3d,
)


# Median RAPTOR-gain case: representative rather than cherry-picked.
SAMPLE_INDEX = 9


def draw_neutral_mesh(ax, xyz: np.ndarray, faces: np.ndarray) -> None:
    tri = xyz[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.45, -0.30, 0.84])
    light /= np.linalg.norm(light)
    shade = 0.70 + 0.25 * np.abs(normals @ light)
    colors = np.column_stack([shade * 0.88, shade * 0.87, shade * 0.83,
                              np.ones(len(shade))])
    mesh = Poly3DCollection(
        tri, facecolors=colors, edgecolors=(0.15, 0.16, 0.17, 0.08),
        linewidths=0.025, antialiased=True, rasterized=True,
    )
    ax.add_collection3d(mesh)
    style_3d(ax, xyz)


def draw_scalar(ax, xyz, faces, values, cmap, norm) -> None:
    colors = lit_facecolors(xyz, faces, values, cmap, norm)
    mesh = Poly3DCollection(
        xyz[faces], facecolors=colors, edgecolors=(0.04, 0.05, 0.06, 0.05),
        linewidths=0.02, antialiased=True, rasterized=True,
    )
    ax.add_collection3d(mesh)
    style_3d(ax, xyz)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "cache" / "split_manifest.json").read_text(encoding="utf-8"))
    member = manifest["test"][SAMPLE_INDEX]
    full = load_full_mesh(member)
    xyz = np.asarray(full["nodes_list"], dtype=float)
    faces = np.asarray(full["elems_list"][:, 1:4], dtype=int)
    truth = np.asarray(full["features_list"], dtype=float).reshape(-1)

    trans = np.load(ROOT / "transolver_tiny" / "test_predictions.npz")
    raptor = np.load(ROOT / "raptor" / "test_predictions.npz")
    sampled_xyz = raptor["x"][SAMPLE_INDEX]
    nearest = cKDTree(sampled_xyz).query(xyz, k=1)[1]
    mean = float(raptor["y_mean"].reshape(-1)[0])
    std = float(raptor["y_std"].reshape(-1)[0])
    trans_full = (trans["y_pred"][SAMPLE_INDEX, :, 0] * std + mean)[nearest]
    raptor_full = (raptor["y_pred"][SAMPLE_INDEX, :, 0] * std + mean)[nearest]
    error_full = np.abs(raptor_full - truth)

    field_pool = np.concatenate([truth, trans_full, raptor_full])
    field_norm = Normalize(*np.quantile(field_pool, [0.01, 0.99]), clip=True)
    error_norm = Normalize(0, float(np.quantile(error_full, 0.99)), clip=True)
    truth_f = face_values(truth, faces)
    trans_f = face_values(trans_full, faces)
    raptor_f = face_values(raptor_full, faces)
    error_f = face_values(error_full, faces)

    raptor_mse = float(np.mean((raptor["y_pred"][SAMPLE_INDEX, :, 0]
                                - raptor["y_true"][SAMPLE_INDEX, :, 0]) ** 2))
    trans_mse = float(np.mean((trans["y_pred"][SAMPLE_INDEX, :, 0]
                               - trans["y_true"][SAMPLE_INDEX, :, 0]) ** 2))

    mpl.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(16.2, 4.15), facecolor="white")
    gs = fig.add_gridspec(1, 5, left=0.018, right=0.982, bottom=0.24, top=0.84,
                          wspace=0.01)
    titles = ["Input Mesh", "GT Surface Field", "Transolver-Tiny",
              "RAPTOR", "RAPTOR |Error|"]
    values = [None, truth_f, trans_f, raptor_f, error_f]
    for col, (title, value) in enumerate(zip(titles, values)):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        if value is None:
            draw_neutral_mesh(ax, xyz, faces)
        elif col == 4:
            draw_scalar(ax, xyz, faces, value, "Reds", error_norm)
        else:
            draw_scalar(ax, xyz, faces, value, "turbo", field_norm)
        ax.set_title(title, fontsize=11.5, pad=-4)

    cax_field = fig.add_axes([0.285, 0.165, 0.38, 0.022])
    cb_field = fig.colorbar(
        mpl.cm.ScalarMappable(norm=field_norm, cmap="turbo"),
        cax=cax_field, orientation="horizontal",
    )
    cb_field.set_label("Surface scalar field (shared scale)", fontsize=9, labelpad=2)
    cb_field.ax.tick_params(labelsize=7, pad=1)
    cax_error = fig.add_axes([0.805, 0.165, 0.15, 0.022])
    cb_error = fig.colorbar(
        mpl.cm.ScalarMappable(norm=error_norm, cmap="Reds"),
        cax=cax_error, orientation="horizontal",
    )
    cb_error.set_label("Absolute error", fontsize=9, labelpad=2)
    cb_error.ax.tick_params(labelsize=7, pad=1)

    fig.suptitle("Full-Mesh Automotive Surface-Field Reconstruction",
                 y=0.985, fontsize=16, weight="semibold")
    fig.text(0.5, 0.915,
             f"Representative test vehicle · seed 42 · Transolver-Tiny MSE={trans_mse:.3f} · RAPTOR MSE={raptor_mse:.3f}",
             ha="center", fontsize=9.2, color="#475569")
    fig.text(0.018, 0.025,
             "Full-mesh colors are visualization-only nearest-neighbor interpolation from 4,096 sampled prediction nodes; numerical MSEs remain on the sampled nodes.",
             fontsize=8, color="#475569")

    stem = OUT / "Figure_RAPTOR3D_five_panel_paper_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {
        "source_member": member,
        "test_index": SAMPLE_INDEX,
        "selection": "median RAPTOR-gain case",
        "vertices": int(len(xyz)),
        "triangular_faces": int(len(faces)),
        "transolver_tiny_normalized_mse": trans_mse,
        "raptor_normalized_mse": raptor_mse,
        "full_mesh_interpolation": "nearest neighbor from original 4096 sampled nodes",
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
