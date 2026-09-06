"""Journal-style 2x3 car field, error, and RAPTOR improvement visualization."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.spatial import cKDTree

from plot_five_panel_paper_comparison import (
    HERE, ROOT, OUT, SAMPLE_INDEX, load_full_mesh, face_values, draw_scalar,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((HERE / "cache" / "split_manifest.json").read_text(encoding="utf-8"))
    member = split["test"][SAMPLE_INDEX]
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
    trans_error = np.abs(trans_full - truth)
    raptor_error = np.abs(raptor_full - truth)
    # Positive values mean that RAPTOR has the smaller pointwise absolute error.
    improvement = trans_error - raptor_error

    field_norm = Normalize(*np.quantile(np.concatenate([truth, trans_full, raptor_full]),
                                        [0.01, 0.99]), clip=True)
    error_norm = Normalize(0, float(np.quantile(np.concatenate([trans_error, raptor_error]), .99)),
                           clip=True)
    improvement_limit = float(np.quantile(np.abs(improvement), .99))
    improvement_norm = TwoSlopeNorm(vmin=-improvement_limit, vcenter=0, vmax=improvement_limit)
    fields = [face_values(v, faces) for v in
              [truth, trans_full, raptor_full, trans_error, raptor_error, improvement]]

    trans_mse = float(np.mean((trans["y_pred"][SAMPLE_INDEX, :, 0]
                               - trans["y_true"][SAMPLE_INDEX, :, 0]) ** 2))
    raptor_mse = float(np.mean((raptor["y_pred"][SAMPLE_INDEX, :, 0]
                                - raptor["y_true"][SAMPLE_INDEX, :, 0]) ** 2))
    gain = 100.0 * (trans_mse - raptor_mse) / trans_mse

    mpl.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(14.6, 7.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=.025, right=.975, bottom=.15, top=.86,
                          wspace=.005, hspace=.13)
    titles = [
        "(a) Ground truth", "(b) Transolver-Tiny", "(c) RAPTOR",
        f"(d) Transolver-Tiny |error|\nMSE = {trans_mse:.3f}",
        f"(e) RAPTOR |error|\nMSE = {raptor_mse:.3f}",
        f"(f) RAPTOR local improvement\nrepresentative gain = {gain:.2f}%",
    ]
    cmaps = ["turbo", "turbo", "turbo", "magma", "magma", "RdYlGn"]
    norms = [field_norm, field_norm, field_norm, error_norm, error_norm, improvement_norm]
    for k, (title, values, cmap, norm) in enumerate(zip(titles, fields, cmaps, norms)):
        ax = fig.add_subplot(gs[k // 3, k % 3], projection="3d")
        draw_scalar(ax, xyz, faces, values, cmap, norm)
        ax.set_title(title, fontsize=11.2, weight="semibold", pad=-3)

    # Center the shared field colorbar on the full figure canvas.
    cax1 = fig.add_axes([.30, .905, .40, .016])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=field_norm, cmap="turbo"),
                       cax=cax1, orientation="horizontal")
    cb1.set_label("Surface scalar field (shared 1–99% scale)", fontsize=8.6, labelpad=1)
    cb1.ax.xaxis.set_label_position("top"); cb1.ax.tick_params(labelsize=7, pad=1)
    cax2 = fig.add_axes([.17, .075, .27, .016])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=error_norm, cmap="magma"),
                       cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute error (shared 0–99% scale)", fontsize=8.6, labelpad=1)
    cb2.ax.xaxis.set_label_position("top"); cb2.ax.tick_params(labelsize=7, pad=1)
    cax3 = fig.add_axes([.61, .075, .24, .016])
    cb3 = fig.colorbar(mpl.cm.ScalarMappable(norm=improvement_norm, cmap="RdYlGn"),
                       cax=cax3, orientation="horizontal")
    cb3.set_label(r"$|e_{Transolver}|-|e_{RAPTOR}|$  (green: RAPTOR improves)",
                  fontsize=8.6, labelpad=1)
    cb3.ax.xaxis.set_label_position("top"); cb3.ax.tick_params(labelsize=7, pad=1)

# =============================================================================
#     fig.suptitle("Full-Mesh Automotive Surface-Field Reconstruction and Error Localization",
#                  y=.985, fontsize=17, weight="semibold")
#     fig.text(.5, .945,
#              f"Median-gain representative test vehicle · seed 42 · {len(xyz):,} vertices · {len(faces):,} triangular faces",
#              ha="center", fontsize=9.2, color="#475569")
#     fig.text(.025, .012,
#              "Full-mesh colors use visualization-only nearest-neighbor interpolation from 4,096 sampled nodes; MSEs are computed on the original sampled nodes.",
#              fontsize=8, color="#475569")
# =============================================================================

    stem = OUT / "Figure_RAPTOR3D_car_field_error_improvement_2x3"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {"source_member": member, "test_index": SAMPLE_INDEX,
            "selection": "median RAPTOR-gain case", "transolver_mse": trans_mse,
            "raptor_mse": raptor_mse, "representative_gain_percent": gain,
            "improvement_definition": "abs_error_transolver - abs_error_raptor",
            "interpolation": "nearest neighbor from 4096 sampled nodes; visualization only"}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
