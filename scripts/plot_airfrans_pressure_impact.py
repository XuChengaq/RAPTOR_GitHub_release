"""Journal-style near-field pressure comparison for all AirfRANS baselines."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
import matplotlib.tri as mtri
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
LEGACY_DIR = WORKSPACE / "AirfRANS_engineering_validation"
sys.path.insert(0, str(PROJECT))

from linearno_tiny import LinearNOTiny  # noqa: E402
from mono_tiny import MoNoTiny  # noqa: E402
from raptor.model import RAPTOR  # noqa: E402
from transolver_plus_tiny import TransolverPlusTiny  # noqa: E402
from transolver_tiny.model import TransolverTiny  # noqa: E402

spec = importlib.util.spec_from_file_location("legacy_airfrans_models", LEGACY_DIR / "models.py")
legacy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(legacy)

OUT = PROJECT / "figs" / "airfrans_pressure_impact"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "reynolds": {
        "title": "Reynolds-number extrapolation",
        "index": 27,
        "cache": PROJECT / "cache" / "reynolds_tr200_te200_n4096.npz",
        "roots": {
            "MLP": LEGACY_DIR / "outputs/formal/reynolds-mlp-s42",
            "GraphSAGE": LEGACY_DIR / "outputs/formal/reynolds-graphsage-s42",
            "PointNet": LEGACY_DIR / "outputs/official_remaining_formal/reynolds-pointnet-s42",
            "Graph U-Net": LEGACY_DIR / "outputs/official_remaining_formal/reynolds-gunet-s42",
            "Sparse Token": LEGACY_DIR / "outputs/formal/reynolds-sparse_token-s42",
            "Transolver-Tiny": PROJECT / "outputs/baselines/transolver_tiny/reynolds-transolver_tiny-s42",
            "Transolver++-Tiny": PROJECT / "outputs/modern_formal_fixed/transolver_plus_tiny/reynolds-transolver_plus_tiny-s42",
            "MoNo-Tiny": PROJECT / "outputs/modern_formal/mono_tiny/reynolds-mono_tiny-s42",
            "LinearNO-Tiny": PROJECT / "outputs/linearno_formal/linearno_tiny/reynolds-linearno_tiny-s42",
            "RAPTOR": PROJECT / "outputs/formal/reynolds-raptor-s42",
        },
    },
    "aoa": {
        "title": "Angle-of-attack extrapolation",
        "index": 119,
        "cache": PROJECT / "cache" / "aoa_tr200_te200_n4096.npz",
        "roots": {
            "MLP": LEGACY_DIR / "outputs/formal/aoa-mlp-s42",
            "GraphSAGE": LEGACY_DIR / "outputs/formal/aoa-graphsage-s42",
            "PointNet": LEGACY_DIR / "outputs/official_remaining_formal/aoa-pointnet-s42",
            "Graph U-Net": LEGACY_DIR / "outputs/official_remaining_formal/aoa-gunet-s42",
            "Sparse Token": LEGACY_DIR / "outputs/formal/aoa-sparse_token-s42",
            "Transolver-Tiny": PROJECT / "outputs/aoa_formal/transolver_tiny/aoa-transolver_tiny-s42",
            "Transolver++-Tiny": PROJECT / "outputs/aoa_formal/modern/transolver_plus_tiny/aoa-transolver_plus_tiny-s42",
            "MoNo-Tiny": PROJECT / "outputs/aoa_formal/modern/mono_tiny/aoa-mono_tiny-s42",
            "LinearNO-Tiny": PROJECT / "outputs/linearno_formal/linearno_tiny/aoa-linearno_tiny-s42",
            "RAPTOR": PROJECT / "outputs/aoa_formal/raptor/aoa-raptor-s42",
        },
    },
}

ORDER = [
    "Ground truth", "MLP", "GraphSAGE", "PointNet",
    "Graph U-Net", "Sparse Token", "Transolver-Tiny", "Transolver++-Tiny",
    "MoNo-Tiny", "LinearNO-Tiny", "RAPTOR", "Pressure error ranking",
]

COLORS = {"traditional": "#7B8794", "modern": "#D97706", "raptor": "#0B5CAD"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10.5, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def build_model(name: str):
    cfg = {"hidden_dim": 64, "depth": 4, "token_count": 64, "sparse_keep": 0.25,
           "knn_k": 12, "heads": 4}
    if name == "MLP":
        return legacy.build_model("mlp", cfg)
    if name == "GraphSAGE":
        return legacy.build_model("graphsage", cfg)
    if name == "PointNet":
        return legacy.build_model("pointnet", cfg)
    if name == "Graph U-Net":
        return legacy.build_model("gunet", cfg)
    if name == "Sparse Token":
        return legacy.build_model("sparse_token", cfg)
    if name == "Transolver-Tiny":
        return TransolverTiny(dim=64, depth=4, heads=4, slices=64)
    if name == "Transolver++-Tiny":
        return TransolverPlusTiny(dim=64, depth=4, heads=4, slices=64)
    if name == "MoNo-Tiny":
        return MoNoTiny(modes=64, dim=64, layers=2)
    if name == "LinearNO-Tiny":
        return LinearNOTiny(dim=64, depth=4, heads=4, states=64)
    if name == "RAPTOR":
        return RAPTOR(dim=64, depth=4, heads=4, slices=64, knn_k=12)
    raise KeyError(name)


def predict_one(name: str, root: Path, data, index: int, device: torch.device) -> np.ndarray:
    model = build_model(name)
    model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    x = data["test_x"][index]
    xn = ((x - data["xmean"]) / data["xstd"]).astype("float32")
    xt = torch.from_numpy(xn)[None].to(device)
    kt = torch.from_numpy(data["test_knn"][index])[None].to(device)
    gt = torch.from_numpy(data["test_groups"][index])[None].to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model(xt, kt, gt)
    pred = out.float().cpu().numpy()[0] * data["ystd"] + data["ymean"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred[:, 2]


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + 1e-12))


def bounds(xy: np.ndarray, surface: np.ndarray):
    s = xy[surface]
    xmin, xmax = s[:, 0].min(), s[:, 0].max()
    chord = max(xmax - xmin, 1e-8)
    cy = 0.5 * (s[:, 1].min() + s[:, 1].max())
    return (xmin - .22 * chord, xmax + 1.05 * chord, cy - .42 * chord, cy + .42 * chord, chord)


def triangulation(xy: np.ndarray, box) -> mtri.Triangulation:
    active = ((xy[:, 0] >= box[0]) & (xy[:, 0] <= box[1]) &
              (xy[:, 1] >= box[2]) & (xy[:, 1] <= box[3]))
    ids = np.flatnonzero(active)
    tri = mtri.Triangulation(xy[ids, 0], xy[ids, 1])
    pts = xy[ids][tri.triangles]
    edges = np.stack([
        np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1),
    ], axis=1)
    tri.set_mask(edges.max(axis=1) > 0.13 * box[4])
    tri._source_ids = ids
    return tri


def airfoil_patch(ax, xy: np.ndarray, surface: np.ndarray) -> None:
    s = xy[surface]
    center = s.mean(axis=0)
    angle = np.arctan2(s[:, 1] - center[1], s[:, 0] - center[0])
    poly = s[np.argsort(angle)]
    ax.add_patch(Polygon(poly, closed=True, facecolor="#121826", edgecolor="white",
                         linewidth=0.55, zorder=8))


def panel(ax, tri, xy, surface, values, norm, title, error=None, highlight=False):
    art = ax.tripcolor(tri, values[tri._source_ids], shading="gouraud", cmap="RdBu_r",
                       norm=norm, rasterized=True)
    airfoil_patch(ax, xy, surface)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontweight="bold", color="#0B5CAD" if highlight else "#182230", pad=7)
    for spine in ax.spines.values():
        spine.set_visible(highlight)
        if highlight:
            spine.set_color("#0B5CAD"); spine.set_linewidth(2.0)
    if error is not None:
        ax.text(.025, .045, rf"pressure rel. $L_2$ = {error:.3f}", transform=ax.transAxes,
                fontsize=7.5, fontweight="bold" if highlight else "normal",
                bbox=dict(boxstyle="round,pad=.24", fc="white", ec="#D7DEE7", alpha=.92))
    return art


def ranking_panel(ax, roots: dict) -> list[tuple[str, float]]:
    rows = []
    for name, root in roots.items():
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        rows.append((name, float(result["rel_l2_p"])))
    rows.sort(key=lambda item: item[1])
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [COLORS["raptor"] if n == "RAPTOR" else
              COLORS["modern"] if n in {"Sparse Token", "Transolver-Tiny", "Transolver++-Tiny", "MoNo-Tiny", "LinearNO-Tiny"}
              else COLORS["traditional"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=.62)
    ax.set_yticks([])
    ax.invert_yaxis()
    xmax = float(vals.max())
    ax.set_xlim(-0.30 * xmax, 1.14 * xmax)
    ax.set_xlabel(r"Test-set pressure rel. $L_2$", fontsize=8)
    ax.set_title("Pressure error ranking", fontweight="bold", pad=7)
    ax.grid(axis="x", color="#E5EAF0", linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, (name, val) in enumerate(zip(names, vals)):
        ax.text(-0.018 * xmax, yi, name, ha="right", va="center", fontsize=7.0)
        ax.text(val + .012 * vals.max(), yi, f"{val:.3f}", va="center", fontsize=7)
    return rows


def make_figure(slug: str, cfg: dict, device: torch.device) -> dict:
    data = np.load(cfg["cache"])
    index = cfg["index"]
    xy = data["test_x"][index, :, :2]
    surface = data["test_surface"][index] > .5
    truth = data["test_y"][index, :, 2]
    preds = {name: predict_one(name, root, data, index, device)
             for name, root in cfg["roots"].items()}
    box = bounds(xy, surface)
    tri = triangulation(xy, box)
    extent = max(abs(float(np.percentile(truth, 1))), abs(float(np.percentile(truth, 99))))
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)

    fig, axes = plt.subplots(3, 4, figsize=(15.8, 8.0))
    plt.subplots_adjust(left=.045, right=.94, bottom=.085, top=.89, wspace=.19, hspace=.34)
    pressure_errors = {}
    art = None
    for ax, name in zip(axes.flat, ORDER):
        if name == "Pressure error ranking":
            ranking = ranking_panel(ax, cfg["roots"])
            continue
        values = truth if name == "Ground truth" else preds[name]
        error = None if name == "Ground truth" else rel_l2(values, truth)
        if error is not None:
            pressure_errors[name] = error
        art = panel(ax, tri, xy, surface, values, norm, name, error,
                    highlight=name == "RAPTOR")
    cax = fig.add_axes([.953, .18, .014, .62])
    cb = fig.colorbar(art, cax=cax)
    cb.set_label(r"Pressure $p$ · shared physical scale", fontsize=9)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f"AirfRANS {cfg['title']} · near-field pressure reconstruction",
        x=.045, y=.965, ha="left", fontsize=16, fontweight="bold", color="#182230",
    )
    fig.text(.045, .025,
             "All predictions share the ground-truth pressure scale. The representative case follows the same "
             "aggregate-matched selection rule used in the four-variable analysis.",
             fontsize=8, color="#536273")
    png = OUT / f"Figure_AirfRANS_{slug}_pressure_all_models.png"
    pdf = OUT / f"Figure_AirfRANS_{slug}_pressure_all_models.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"task": slug, "test_index": index, "sample_pressure_errors": pressure_errors,
            "aggregate_pressure_ranking": ranking, "png": str(png), "pdf": str(pdf)}


def main() -> None:
    style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {"device": str(device), "selection": "aggregate-matched representative case", "figures": []}
    for slug, cfg in TASKS.items():
        for root in cfg["roots"].values():
            for filename in ("best.pt", "result.json"):
                if not (root / filename).exists():
                    raise FileNotFoundError(root / filename)
        manifest["figures"].append(make_figure(slug, cfg, device))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
