"""High-contrast 2x3 AirfRANS figure: p/rho truth and top-two error maps.

For each task, the displayed case is selected with the same pressure metric
reported in the panel annotations.  Let e_q^m denote the relative L2 pressure
error of model m on test case q and let epsilon^m denote the pooled test-set
pressure error.  The selected case minimizes

    |(e_q^b - e_q^R) / e_q^b - (epsilon^b - epsilon^R) / epsilon^b|,

where R is RAPTOR and b is the other displayed (macro-rank-2) model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

import plot_airfrans_pressure_impact as base
from plot_airfrans_top3_macro_pressure import macro_error


OUT = base.PROJECT / "figs" / "airfrans_pressure_impact"


def predict_all_pressure(name: str, root: Path, data, device: torch.device) -> np.ndarray:
    """Predict p/rho for every test case while loading the checkpoint once."""
    model = base.build_model(name)
    model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
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
            rows.append(pred[:, 2].astype("float32"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack(rows)


def sample_relative_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-case relative L2 error over all sampled nodes."""
    num = np.sqrt(np.sum((pred - truth) ** 2, axis=1, dtype=np.float64))
    den = np.sqrt(np.sum(truth**2, axis=1, dtype=np.float64)) + 1e-12
    return num / den


def main() -> None:
    base.style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 5.5))
    plt.subplots_adjust(left=.045, right=.91, bottom=.12, top=.86,
                        wspace=.10, hspace=.42)
    manifest = {
        "device": str(device),
        "map_error_definition": "100*abs(pred-truth)/(p99(truth)-p1(truth))",
        "selection_metric": "pressure relative L2 over all 4096 sampled nodes",
        "selection_rule": (
            "argmin_q abs(((e_base_q-e_raptor_q)/e_base_q) - "
            "((eps_base-eps_raptor)/eps_base))"
        ),
        "tasks": [],
    }

    for row, (slug, cfg) in enumerate(base.TASKS.items()):
        data = np.load(cfg["cache"])
        ranked = sorted(
            ((name, root, macro_error(root / "result.json"))
             for name, root in cfg["roots"].items()), key=lambda x: x[2])[:2]
        if ranked[0][0] != "RAPTOR":
            raise RuntimeError(f"Expected RAPTOR to be macro rank 1 for {slug}; got {ranked}")
        raptor_name, raptor_root, _ = ranked[0]
        base_name, base_root, _ = ranked[1]
        all_predictions = {
            name: predict_all_pressure(name, root, data, device)
            for name, root, _ in ranked
        }
        truth_all = data["test_y"][..., 2]
        sample_errors = {
            name: sample_relative_l2(all_predictions[name], truth_all)
            for name, _, _ in ranked
        }
        raptor_result = json.loads((raptor_root / "result.json").read_text(encoding="utf-8"))
        base_result = json.loads((base_root / "result.json").read_text(encoding="utf-8"))
        aggregate_gain = (
            float(base_result["rel_l2_p"]) - float(raptor_result["rel_l2_p"])
        ) / max(float(base_result["rel_l2_p"]), 1e-12)
        sample_gains = (
            sample_errors[base_name] - sample_errors[raptor_name]
        ) / np.maximum(sample_errors[base_name], 1e-12)
        idx = int(np.argmin(np.abs(sample_gains - aggregate_gain)))
        xy = data["test_x"][idx, :, :2]
        surface = data["test_surface"][idx] > .5
        truth = data["test_y"][idx, :, 2]
        box = base.bounds(xy, surface)
        tri = base.triangulation(xy, box)
        ids = tri._source_ids
        pscale = max(abs(float(np.percentile(truth, 1))),
                     abs(float(np.percentile(truth, 99))))
        pnorm = TwoSlopeNorm(vmin=-pscale, vcenter=0, vmax=pscale)
        prange = max(float(np.percentile(truth, 99) - np.percentile(truth, 1)), 1e-12)
        predictions = [
            (name, macro, all_predictions[name][idx])
            for name, _, macro in ranked
        ]
        errors = [100.0 * np.abs(pred - truth) / prange for _, _, pred in predictions]
        emax = max(float(np.percentile(err[ids], 99)) for err in errors)

        ax = axes[row, 0]
        p_art = ax.tripcolor(tri, truth[ids], shading="gouraud", cmap="RdBu_r",
                             norm=pnorm, rasterized=True)
        base.airfoil_patch(ax, xy, surface)
        ax.set_title(r"Ground truth · $p/\rho$", fontsize=11, fontweight="bold")
        ax.text(.025, .045, rf"physical scale: $[-{pscale:.0f},\,{pscale:.0f}]$",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=.24", fc="white", ec="#D7DEE7", alpha=.94))
        task_rows = []

        for col, ((name, macro, pred), error) in enumerate(zip(predictions, errors), start=1):
            ax_e = axes[row, col]
            e_art = ax_e.tripcolor(tri, error[ids], shading="gouraud", cmap="magma",
                                   vmin=0, vmax=emax, rasterized=True)
            base.airfoil_patch(ax_e, xy, surface)
            rank = col
            ax_e.set_title(f"Macro rank #{rank} · {name}", fontsize=11,
                           fontweight="bold", color="#0B5CAD" if name == "RAPTOR" else "#182230")
            sample_p = float(sample_errors[name][idx])
            gain_text = (
                rf"  |  paired $\Delta_p$={100.0 * sample_gains[idx]:+.1f}\%"
                if name == "RAPTOR" else ""
            )
            ax_e.text(.025, .045,
                      rf"sample $p/\rho$ rel. $L_2$={sample_p:.3f}{gain_text}",
                      transform=ax_e.transAxes, fontsize=8,
                      fontweight="bold" if name == "RAPTOR" else "normal",
                      bbox=dict(boxstyle="round,pad=.24", fc="white",
                                ec="#0B5CAD" if name == "RAPTOR" else "#D7DEE7", alpha=.94))
            for spine in ax_e.spines.values():
                spine.set_visible(name == "RAPTOR")
                if name == "RAPTOR":
                    spine.set_color("#0B5CAD"); spine.set_linewidth(2.2)
            task_rows.append({"rank": rank, "model": name, "macro_rel_l2": macro,
                              "sample_pressure_rel_l2": sample_p})

        for ax_i in axes[row]:
            ax_i.set_aspect("equal", adjustable="box")
            ax_i.set_xlim(box[0], box[1]); ax_i.set_ylim(box[2], box[3])
            ax_i.set_xticks([]); ax_i.set_yticks([])
        axes[row, 0].text(-.07, .5, "Reynolds" if slug == "reynolds" else "AoA",
                          transform=axes[row, 0].transAxes, rotation=90, va="center",
                          ha="center", fontsize=12, fontweight="bold", color="#44546A")
        cax = fig.add_axes([.925, .555 if row == 0 else .17, .012, .22])
        cb = fig.colorbar(e_art, cax=cax)
        cb.set_label(r"Normalized $|p/\rho-\widehat{p/\rho}|$ (\%)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        manifest["tasks"].append({
            "task": slug,
            "test_index": idx,
            "comparison_model": base_name,
            "aggregate_pressure_improvement_percent": 100.0 * aggregate_gain,
            "selected_sample_pressure_improvement_percent": 100.0 * float(sample_gains[idx]),
            "selection_gap_percentage_points": 100.0 * abs(float(sample_gains[idx] - aggregate_gain)),
            "error_color_limit_percent": emax,
            "top2": task_rows,
        })


    png = OUT / "Figure_AirfRANS_truth_top2_pressure_error_2x3.png"
    pdf = OUT / "Figure_AirfRANS_truth_top2_pressure_error_2x3.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.update({"png": str(png), "pdf": str(pdf)})
    (OUT / "truth_top2_pressure_error_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
