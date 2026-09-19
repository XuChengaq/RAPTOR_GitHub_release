"""High-contrast 2x3 AirfRANS figure: p/rho truth and top-two error maps."""
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


def main() -> None:
    base.style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 5.5))
    plt.subplots_adjust(left=.045, right=.91, bottom=.12, top=.86,
                        wspace=.10, hspace=.42)
    manifest = {"device": str(device), "error_definition":
                "100*abs(pred-truth)/(p99(truth)-p1(truth))", "tasks": []}

    for row, (slug, cfg) in enumerate(base.TASKS.items()):
        data = np.load(cfg["cache"])
        idx = cfg["index"]
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
        ranked = sorted(
            ((name, root, macro_error(root / "result.json"))
             for name, root in cfg["roots"].items()), key=lambda x: x[2])[:2]
        predictions = [(name, macro, base.predict_one(name, root, data, idx, device))
                       for name, root, macro in ranked]
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
            sample_p = base.rel_l2(pred, truth)
            ax_e.text(.025, .045, rf"macro $L_2$={macro:.3f}  |  sample $p/\rho$ rel. $L_2$={sample_p:.3f}",
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
        manifest["tasks"].append({"task": slug, "test_index": idx,
                                  "error_color_limit_percent": emax, "top2": task_rows})


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
