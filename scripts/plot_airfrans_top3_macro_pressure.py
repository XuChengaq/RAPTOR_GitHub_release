"""Compact 2x3 AirfRANS pressure figure: truth and top two by macro error."""
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


OUT = base.PROJECT / "figs" / "airfrans_pressure_impact"


def macro_error(result_path: Path) -> float:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "macro_rel_l2" in result:
        return float(result["macro_rel_l2"])
    return float(np.mean([
        result["rel_l2_ux"], result["rel_l2_uy"],
        result["rel_l2_p"], result["rel_l2_nut"],
    ]))


def main() -> None:
    base.style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 5.35))
    plt.subplots_adjust(left=.045, right=.91, bottom=.12, top=.86,
                        wspace=.10, hspace=.36)
    manifest = {"device": str(device), "ranking_metric": "four-variable macro relative L2",
                "layout": "ground truth plus top two models per task", "tasks": []}
    row_arts = []

    for row, (slug, cfg) in enumerate(base.TASKS.items()):
        data = np.load(cfg["cache"])
        idx = cfg["index"]
        xy = data["test_x"][idx, :, :2]
        surface = data["test_surface"][idx] > .5
        truth = data["test_y"][idx, :, 2]
        box = base.bounds(xy, surface)
        tri = base.triangulation(xy, box)
        extent = max(abs(float(np.percentile(truth, 1))),
                     abs(float(np.percentile(truth, 99))))
        norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)

        ranked = sorted(
            ((name, root, macro_error(root / "result.json"))
             for name, root in cfg["roots"].items()),
            key=lambda item: item[2],
        )[:2]
        task_info = {"task": slug, "test_index": idx, "top2": []}

        truth_ax = axes[row, 0]
        last_art = truth_ax.tripcolor(
            tri, truth[tri._source_ids], shading="gouraud",
            cmap="RdBu_r", norm=norm, rasterized=True,
        )
        base.airfoil_patch(truth_ax, xy, surface)
        truth_ax.set_aspect("equal", adjustable="box")
        truth_ax.set_xlim(box[0], box[1]); truth_ax.set_ylim(box[2], box[3])
        truth_ax.set_xticks([]); truth_ax.set_yticks([])
        truth_ax.set_title("Ground Truth", fontsize=11.5, fontweight="bold",
                           color="#182230", pad=7)
        for spine in truth_ax.spines.values():
            spine.set_visible(False)

        for rank, (name, root, macro) in enumerate(ranked, start=1):
            pred = base.predict_one(name, root, data, idx, device)
            pressure = base.rel_l2(pred, truth)
            ax = axes[row, rank]
            last_art = ax.tripcolor(
                tri, pred[tri._source_ids], shading="gouraud",
                cmap="RdBu_r", norm=norm, rasterized=True,
            )
            base.airfoil_patch(ax, xy, surface)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Macro rank #{rank}  {name}", fontsize=11.5, fontweight="bold",
                         color="#0B5CAD" if name == "RAPTOR" else "#182230", pad=7)
            for spine in ax.spines.values():
                spine.set_visible(name == "RAPTOR")
                if name == "RAPTOR":
                    spine.set_color("#0B5CAD"); spine.set_linewidth(2.2)
            ax.text(.025, .045,
                    rf"macro $L_2$ = {macro:.3f}  |  sample $p$ $L_2$ = {pressure:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    fontweight="bold" if name == "RAPTOR" else "normal",
                    bbox=dict(boxstyle="round,pad=.25", fc="white",
                              ec="#0B5CAD" if name == "RAPTOR" else "#D7DEE7", alpha=.94))
            task_info["top2"].append({"rank": rank, "model": name,
                                      "macro_rel_l2": macro,
                                      "sample_pressure_rel_l2": pressure})

        axes[row, 0].text(-.07, .5, "Reynolds" if slug == "reynolds" else "AoA",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=12, fontweight="bold",
                          color="#44546A")
        row_arts.append(last_art)
        manifest["tasks"].append(task_info)

    for art, y0 in zip(row_arts, (.535, .175)):
        cax = fig.add_axes([.925, y0, .012, .245])
        cb = fig.colorbar(art, cax=cax)
        cb.set_label(r"Pressure $p$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("AirfRANS pressure reconstruction · ground truth and top two models",
                 x=.045, y=.965, ha="left", fontsize=16, fontweight="bold", color="#182230")
    fig.text(.045, .035,
             "Ranks and macro errors use the complete test set; pressure errors refer only to the displayed representative case. "
             "Ground truth and both predictions within each task share the same physical pressure scale.",
             fontsize=8.2, color="#536273")

    png = OUT / "Figure_AirfRANS_truth_top2_macro_pressure_2x3.png"
    pdf = OUT / "Figure_AirfRANS_truth_top2_macro_pressure_2x3.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.update({"png": str(png), "pdf": str(pdf)})
    (OUT / "truth_top2_macro_pressure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
