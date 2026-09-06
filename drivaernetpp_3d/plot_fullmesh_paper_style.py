from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "DrivAerNetPP_100"
ARCHIVE = DATA_ROOT / "mixed_3d.zip"
ROOT = HERE / "outputs" / "formal_100car"
OUT = ROOT / "visualizations"
SAMPLE_INDEX = 1
MODELS = ["pointnet", "transolver_tiny", "raptor"]
LABELS = {
    "pointnet": "PointNet",
    "transolver_tiny": "Transolver-Tiny",
    "raptor": "RAPTOR",
}

sys.path.insert(0, str(DATA_ROOT))
from inspect_nested_zip import SliceReader, nested_data_offset  # noqa: E402


def load_full_mesh(member: str) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(ARCHIVE) as outer:
        nested = next(item for item in outer.infolist() if item.filename.lower().endswith(".zip"))
        start = nested_data_offset(ARCHIVE, nested)
    with SliceReader(ARCHIVE, start, nested.file_size) as raw:
        with zipfile.ZipFile(io.BufferedReader(raw)) as inner:
            payload = np.load(io.BytesIO(inner.read(member)), allow_pickle=False)
            return {key: payload[key] for key in payload.files}


def face_values(node_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return node_values[faces].mean(axis=1)


def lit_facecolors(xyz: np.ndarray, faces: np.ndarray, values: np.ndarray,
                   cmap: str, norm: Normalize) -> np.ndarray:
    tri = xyz[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    light = np.array([-0.45, -0.30, 0.84])
    light /= np.linalg.norm(light)
    illumination = 0.62 + 0.38 * np.abs(normals @ light)
    rgba = mpl.colormaps[cmap](norm(values))
    rgba[:, :3] *= illumination[:, None]
    rgba[:, :3] = np.clip(rgba[:, :3], 0, 1)
    return rgba


def style_3d(ax, xyz: np.ndarray) -> None:
    low, high = xyz.min(0), xyz.max(0)
    center = (low + high) / 2
    span = high - low
    ax.set_xlim(low[0] - 0.02 * span[0], high[0] + 0.02 * span[0])
    ax.set_ylim(low[1] - 0.05 * span[1], high[1] + 0.05 * span[1])
    ax.set_zlim(low[2] - 0.08 * span[2], high[2] + 0.10 * span[2])
    ax.set_box_aspect((2.05, 1.0, 0.70), zoom=1.10)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()


def draw_mesh(ax, xyz: np.ndarray, faces: np.ndarray, values: np.ndarray | None,
              cmap: str = "coolwarm", norm: Normalize | None = None) -> None:
    if values is None:
        colors = np.tile(np.array([[0.32, 0.43, 0.55, 1.0]]), (len(faces), 1))
        colors = lit_facecolors(xyz, faces, np.zeros(len(faces)), "Blues", Normalize(-1, 1))
    else:
        colors = lit_facecolors(xyz, faces, values, cmap, norm)
    mesh = Poly3DCollection(
        xyz[faces], facecolors=colors,
        edgecolors=(0.04, 0.06, 0.09, 0.10), linewidths=0.035,
        antialiased=True, rasterized=True,
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
    full_truth = np.asarray(full["features_list"], dtype=float).reshape(-1)

    arrays = {name: np.load(ROOT / name / "test_predictions.npz") for name in MODELS}
    sampled_xyz = arrays["raptor"]["x"][SAMPLE_INDEX]
    mean = float(arrays["raptor"]["y_mean"].reshape(-1)[0])
    std = float(arrays["raptor"]["y_std"].reshape(-1)[0])
    tree = cKDTree(sampled_xyz)
    _, nearest = tree.query(xyz, k=1)
    predictions = {
        name: arrays[name]["y_pred"][SAMPLE_INDEX, :, 0] * std + mean
        for name in MODELS
    }
    full_predictions = {name: values[nearest] for name, values in predictions.items()}
    full_errors = {name: np.abs(values - full_truth) for name, values in full_predictions.items()}

    field_pool = np.concatenate([full_truth, *full_predictions.values()])
    vmin, vmax = np.quantile(field_pool, [0.01, 0.99])
    emax = float(np.quantile(np.concatenate(list(full_errors.values())), 0.99))
    field_norm = Normalize(vmin, vmax, clip=True)
    error_norm = Normalize(0, emax, clip=True)
    truth_faces = face_values(full_truth, faces)
    prediction_faces = {name: face_values(values, faces) for name, values in full_predictions.items()}
    error_faces = {name: face_values(values, faces) for name, values in full_errors.items()}

    sample_mse = {
        name: float(np.mean((arrays[name]["y_pred"][SAMPLE_INDEX, :, 0]
                            - arrays[name]["y_true"][SAMPLE_INDEX, :, 0]) ** 2))
        for name in MODELS
    }

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(16.8, 8.4), facecolor="white")
    gs = fig.add_gridspec(2, 4, left=0.018, right=0.982, bottom=0.11, top=0.89,
                          wspace=0.01, hspace=0.08)

    top = [
        ("(a)  Surface mesh", None),
        ("(b)  Ground truth", truth_faces),
        ("(c)  Transolver-Tiny", prediction_faces["transolver_tiny"]),
        ("(d)  RAPTOR", prediction_faces["raptor"]),
    ]
    for j, (title, values) in enumerate(top):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        draw_mesh(ax, xyz, faces, values, "coolwarm", field_norm)
        ax.set_title(title, fontsize=11, weight="semibold", pad=-5)

    bottom = [
        ("(e)  PointNet prediction", prediction_faces["pointnet"], "coolwarm", field_norm),
        (f"(f)  PointNet |error|\nMSE = {sample_mse['pointnet']:.3f}", error_faces["pointnet"], "Reds", error_norm),
        (f"(g)  Transolver-Tiny |error|\nMSE = {sample_mse['transolver_tiny']:.3f}", error_faces["transolver_tiny"], "Reds", error_norm),
        (f"(h)  RAPTOR |error|\nMSE = {sample_mse['raptor']:.3f}", error_faces["raptor"], "Reds", error_norm),
    ]
    for j, (title, values, cmap, norm) in enumerate(bottom):
        ax = fig.add_subplot(gs[1, j], projection="3d")
        draw_mesh(ax, xyz, faces, values, cmap, norm)
        ax.set_title(title, fontsize=10.5, weight="semibold", pad=-5)

    cax1 = fig.add_axes([0.29, 0.925, 0.42, 0.014])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=field_norm, cmap="coolwarm"),
                       cax=cax1, orientation="horizontal")
    cb1.set_label("Surface scalar field (shared 1–99% scale)", labelpad=2)
    cb1.ax.xaxis.set_label_position("top")
    cb1.ax.tick_params(labelsize=8, pad=1)
    cax2 = fig.add_axes([0.31, 0.060, 0.38, 0.014])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=error_norm, cmap="Reds"),
                       cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute error (shared 0–99% scale)", labelpad=2)
    cb2.ax.xaxis.set_label_position("top")
    cb2.ax.tick_params(labelsize=8, pad=1)

# =============================================================================
#     fig.suptitle("Full-Mesh 3D Automotive Surface-Field Reconstruction",
#                  y=0.992, fontsize=18, weight="semibold")
# =============================================================================
# =============================================================================
#     fig.text(0.5, 0.958,
#              f"{member}  ·  {len(xyz):,} vertices  ·  {len(faces):,} triangular faces  ·  seed 42",
#              ha="center", fontsize=9.5, color="#475569")
# =============================================================================
 

    png = OUT / "Figure_RAPTOR3D_100car_fullmesh_light_error.png"
    pdf = OUT / "Figure_RAPTOR3D_100car_fullmesh_light_error.pdf"
    fig.savefig(png, dpi=350, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "source_member": member,
        "sample_index_within_test_split": SAMPLE_INDEX,
        "vertices": int(len(xyz)), "triangular_faces": int(len(faces)),
        "visualization_interpolation": "nearest neighbor from 4096 sampled nodes to full mesh",
        "sample_normalized_mse_on_original_4096_nodes": sample_mse,
    }
    (OUT / "Figure_RAPTOR3D_100car_fullmesh_paper_style.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(png)


if __name__ == "__main__":
    main()
