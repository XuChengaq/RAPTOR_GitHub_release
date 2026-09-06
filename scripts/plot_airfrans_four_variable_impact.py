"""Create publication-grade four-variable AirfRANS comparison figures.

The representative case for each task is selected objectively: its paired
RAPTOR-over-Transolver improvement is closest to the task-level aggregate
improvement reported by the formal evaluation. Aggregate values in the
summary panel are read from formal result files rather than recomputed from
the displayed case.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from raptor.model import RAPTOR  # noqa: E402
from transolver_tiny.model import TransolverTiny  # noqa: E402


OUT = PROJECT / "figs" / "airfrans_four_variable"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "Reynolds-number extrapolation": {
        "cache": PROJECT / "cache" / "reynolds_tr200_te200_n4096.npz",
        "raptor_ckpt": PROJECT / "outputs" / "formal" / "reynolds-raptor-s42" / "best.pt",
        "trans_ckpt": PROJECT / "outputs" / "baselines" / "transolver_tiny" / "reynolds-transolver_tiny-s42" / "best.pt",
        "raptor_result": PROJECT / "outputs" / "formal" / "reynolds-raptor-s42" / "result.json",
        "trans_result": PROJECT / "outputs" / "baselines" / "transolver_tiny" / "reynolds-transolver_tiny-s42" / "result.json",
        "slug": "reynolds",
    },
    "AoA extrapolation": {
        "cache": PROJECT / "cache" / "aoa_tr200_te200_n4096.npz",
        "raptor_ckpt": PROJECT / "outputs" / "aoa_formal" / "raptor" / "aoa-raptor-s42" / "best.pt",
        "trans_ckpt": PROJECT / "outputs" / "aoa_formal" / "transolver_tiny" / "aoa-transolver_tiny-s42" / "best.pt",
        "raptor_result": PROJECT / "outputs" / "aoa_formal" / "raptor" / "aoa-raptor-s42" / "result.json",
        "trans_result": PROJECT / "outputs" / "aoa_formal" / "transolver_tiny" / "aoa-transolver_tiny-s42" / "result.json",
        "slug": "aoa",
    },
}

VARIABLES = [
    (r"$u_x$", "ux", "cividis"),
    (r"$u_y$", "uy", "RdBu_r"),
    (r"$p$", "p", "RdBu_r"),
    (r"$\nu_t$", "nut", "viridis"),
]
BLUE = "#1764AB"
RED = "#D1495B"
DARK = "#182230"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_model(kind: str, checkpoint: Path, device: torch.device):
    if kind == "raptor":
        model = RAPTOR(dim=64, depth=4, heads=4, slices=64, knn_k=12)
    else:
        model = TransolverTiny(dim=64, depth=4, heads=4, slices=64)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, data, device: torch.device) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for x, knn, groups in zip(data["test_x"], data["test_knn"], data["test_groups"]):
            xn = ((x - data["xmean"]) / data["xstd"]).astype("float32")
            xt = torch.from_numpy(xn)[None].to(device)
            kt = torch.from_numpy(knn)[None].to(device)
            gt = torch.from_numpy(groups)[None].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(xt, kt, gt)
            pred = out.float().cpu().numpy()[0] * data["ystd"] + data["ymean"]
            rows.append(pred.astype("float32"))
    return np.stack(rows)


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, pred.ndim))
    num = np.sqrt(np.sum((pred - truth) ** 2, axis=axes, dtype=np.float64))
    den = np.sqrt(np.sum(truth**2, axis=axes, dtype=np.float64)) + 1e-12
    return num / den


def select_aggregate_matched_case(
    trans_pred: np.ndarray,
    raptor_pred: np.ndarray,
    truth: np.ndarray,
    target_gain: float,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    trans_errors = np.stack(
        [rel_l2(trans_pred[..., i], truth[..., i]) for i in range(4)], axis=1
    )
    raptor_errors = np.stack(
        [rel_l2(raptor_pred[..., i], truth[..., i]) for i in range(4)], axis=1
    )
    trans_macro = trans_errors.mean(axis=1)
    raptor_macro = raptor_errors.mean(axis=1)
    paired_gain = (trans_macro - raptor_macro) / np.maximum(trans_macro, 1e-12)
    index = int(np.argmin(np.abs(paired_gain - target_gain)))
    return index, trans_errors[index], raptor_errors[index], float(paired_gain[index])


def near_field_bounds(xy: np.ndarray, surface: np.ndarray) -> tuple[float, float, float, float]:
    pts = xy[surface]
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    chord = max(xmax - xmin, 1e-6)
    cy = 0.5 * (ymin + ymax)
    return xmin - 0.10 * chord, xmax + 1.15 * chord, cy - 0.42 * chord, cy + 0.42 * chord


def clean_field_axis(ax, bounds) -> None:
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def field_norm(values: np.ndarray, cmap: str):
    lo, hi = np.nanpercentile(values, [1, 99])
    if hi <= lo:
        hi = lo + 1e-12
    if cmap == "RdBu_r" and lo < 0 < hi:
        extent = max(abs(lo), abs(hi))
        return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    return Normalize(lo, hi)


def plot_task(title: str, cfg: dict, data, trans_pred, raptor_pred, target_gain: float) -> dict:
    truth = data["test_y"]
    index, sample_trans_errors, sample_raptor_errors, paired_gain = select_aggregate_matched_case(
        trans_pred, raptor_pred, truth, target_gain
    )
    xy = data["test_x"][index, :, :2]
    surface = data["test_surface"][index] > 0.5
    bounds = near_field_bounds(xy, surface)

    fig = plt.figure(figsize=(15.8, 10.4))
    gs = fig.add_gridspec(
        4, 7, width_ratios=[1, 1, 1, 0.055, 1, 1, 0.055],
        left=0.055, right=0.965, bottom=0.065, top=0.90,
        wspace=0.08, hspace=0.14,
    )
    headers = ["Ground truth", "Transolver-Tiny", "RAPTOR", "", "Transolver-Tiny", "RAPTOR"]
    for col, header in zip([0, 1, 2, 4, 5], [headers[i] for i in [0, 1, 2, 4, 5]]):
        axh = fig.add_subplot(gs[0, col], frame_on=False)
        axh.set_title(header, fontsize=11, fontweight="bold", color=DARK, pad=12)
        axh.set_axis_off()

    for row, (display, _, cmap) in enumerate(VARIABLES):
        gt = truth[index, :, row]
        tp = trans_pred[index, :, row]
        rp = raptor_pred[index, :, row]
        norm = field_norm(gt, cmap)
        robust_range = max(np.percentile(gt, 99) - np.percentile(gt, 1), 1e-12)
        te = np.abs(tp - gt) / robust_range
        re = np.abs(rp - gt) / robust_range
        err_hi = max(float(np.percentile(np.concatenate([te, re]), 99)), 1e-8)
        err_norm = Normalize(0, err_hi)

        field_axes = [fig.add_subplot(gs[row, c]) for c in (0, 1, 2)]
        for ax, values in zip(field_axes, (gt, tp, rp)):
            art = ax.scatter(
                xy[:, 0], xy[:, 1], c=values, s=4.0, cmap=cmap, norm=norm,
                linewidths=0, rasterized=True,
            )
            clean_field_axis(ax, bounds)
        field_axes[0].set_ylabel(display, rotation=0, labelpad=18, fontsize=13, fontweight="bold")
        field_axes[1].text(
            0.02, 0.04, f"rel. $L_2$ = {rel_l2(tp[None], gt[None])[0]:.3f}",
            transform=field_axes[1].transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.88),
        )
        field_axes[2].text(
            0.02, 0.04, f"rel. $L_2$ = {rel_l2(rp[None], gt[None])[0]:.3f}",
            transform=field_axes[2].transAxes, fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.88),
        )
        cax = fig.add_subplot(gs[row, 3])
        cb = fig.colorbar(art, cax=cax)
        cb.ax.tick_params(length=2, labelsize=6.5)
        if row == 0:
            cb.ax.set_title("field", fontsize=7, pad=4)

        error_axes = [fig.add_subplot(gs[row, c]) for c in (4, 5)]
        for ax, values in zip(error_axes, (te, re)):
            err_art = ax.scatter(
                xy[:, 0], xy[:, 1], c=values, s=4.0, cmap="magma",
                norm=err_norm, linewidths=0, rasterized=True,
            )
            clean_field_axis(ax, bounds)
        caxe = fig.add_subplot(gs[row, 6])
        cbe = fig.colorbar(err_art, cax=caxe)
        cbe.ax.tick_params(length=2, labelsize=6.5)
        if row == 0:
            cbe.ax.set_title(r"$|e|/R_{99}$", fontsize=7, pad=4)

        if row == 0:
            field_axes[0].text(-0.18, 1.20, "(a)", transform=field_axes[0].transAxes,
                               fontsize=13, fontweight="bold")
            error_axes[0].text(-0.18, 1.20, "(b)", transform=error_axes[0].transAxes,
                               fontsize=13, fontweight="bold")

    fig.text(0.31, 0.935, "Physical fields · shared truth scale", ha="center",
             fontsize=11, fontweight="bold", color=BLUE)
    fig.text(0.73, 0.935, "Normalized absolute error · shared model scale", ha="center",
             fontsize=11, fontweight="bold", color=RED)
    fig.suptitle(
        f"{title}  |  aggregate-matched representative test case #{index}",
        x=0.055, y=0.985, ha="left", fontsize=15, fontweight="bold", color=DARK,
    )
    fig.text(
        0.055, 0.025,
        "Each prediction uses the same color scale as its ground truth. Error maps use a common scale "
        "for both models within each variable; the displayed case is selected by the aggregate-matched rule.",
        fontsize=8, color="#4D5966",
    )

    png = OUT / f"Figure_AirfRANS_{cfg['slug']}_four_variable.png"
    pdf = OUT / f"Figure_AirfRANS_{cfg['slug']}_four_variable.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "index": index,
        "sample_transolver_errors": sample_trans_errors.tolist(),
        "sample_raptor_errors": sample_raptor_errors.tolist(),
        "sample_paired_improvement_percent": 100.0 * paired_gain,
        "png": str(png),
        "pdf": str(pdf),
    }


def plot_summary(records: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    labels = [r"$u_x$", r"$u_y$", r"$p$", r"$\nu_t$", "Macro"]
    keys = ["rel_l2_ux", "rel_l2_uy", "rel_l2_p", "rel_l2_nut", "macro_rel_l2"]
    for panel, (ax, record) in enumerate(zip(axes, records)):
        trans = np.array([record["trans_result"][k] for k in keys])
        raptor = np.array([record["raptor_result"][k] for k in keys])
        y = np.arange(len(labels))
        ax.hlines(y, np.minimum(trans, raptor), np.maximum(trans, raptor), color="#CBD5E1", lw=3)
        ax.scatter(trans, y, s=52, color="#9AA6B2", edgecolor="white", linewidth=.8,
                   zorder=3, label="Transolver-Tiny")
        ax.scatter(raptor, y, s=62, color=BLUE, edgecolor="white", linewidth=.8,
                   zorder=4, label="RAPTOR")
        for yi, (tv, rv) in enumerate(zip(trans, raptor)):
            gain = 100.0 * (tv - rv) / tv
            color = "#0B7A53" if gain >= 0 else RED
            ax.text(max(tv, rv) + 0.012 * max(trans.max(), raptor.max()), yi,
                    f"{gain:+.1f}%", va="center", fontsize=8, color=color, fontweight="bold")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"Relative $L_2$ error  (lower is better)")
        ax.set_title(f"({chr(99 + panel)}) {record['title']}", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E5EAF0", linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Aggregate error fingerprint across all four aerodynamic variables",
                 fontsize=14, fontweight="bold", color=DARK)
    png = OUT / "Figure_AirfRANS_four_variable_quantitative_summary.png"
    pdf = OUT / "Figure_AirfRANS_four_variable_quantitative_summary.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    configure_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "selection_rule": "sample paired improvement closest to the formal task-level aggregate improvement",
        "device": str(device),
        "tasks": {},
    }
    summary_records = []
    for title, cfg in TASKS.items():
        for path_key in ("cache", "raptor_ckpt", "trans_ckpt", "raptor_result", "trans_result"):
            if not cfg[path_key].exists():
                raise FileNotFoundError(cfg[path_key])
        data = np.load(cfg["cache"])
        trans_model = load_model("transolver", cfg["trans_ckpt"], device)
        trans_pred = predict(trans_model, data, device)
        del trans_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raptor_model = load_model("raptor", cfg["raptor_ckpt"], device)
        raptor_pred = predict(raptor_model, data, device)
        del raptor_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raptor_result = json.loads(cfg["raptor_result"].read_text(encoding="utf-8"))
        trans_result = json.loads(cfg["trans_result"].read_text(encoding="utf-8"))
        target_gain = (
            trans_result["macro_rel_l2"] - raptor_result["macro_rel_l2"]
        ) / trans_result["macro_rel_l2"]
        task_manifest = plot_task(
            title, cfg, data, trans_pred, raptor_pred, target_gain
        )
        task_manifest["aggregate_raptor_macro"] = raptor_result["macro_rel_l2"]
        task_manifest["aggregate_transolver_macro"] = trans_result["macro_rel_l2"]
        manifest["tasks"][cfg["slug"]] = task_manifest
        summary_records.append({"title": title, "raptor_result": raptor_result, "trans_result": trans_result})
    summary_png, summary_pdf = plot_summary(summary_records)
    manifest["summary_png"] = str(summary_png)
    manifest["summary_pdf"] = str(summary_pdf)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
