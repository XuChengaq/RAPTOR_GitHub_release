from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent


def box(ax, xy, wh, title, lines, face, edge, title_color="#10243e",
        fontsize=8.3, mono=False, radius=0.012):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=1.35, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.028, title, ha="left", va="top",
            fontsize=10.2, fontweight="bold", color=title_color, zorder=3)
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    ax.text(x + 0.014, y + h - 0.069, "\n".join(lines), ha="left", va="top",
            fontsize=fontsize, color="#263746", family=family,
            linespacing=1.34, zorder=3)
    return patch


def arrow(ax, start, end, color="#64748b", width=1.45, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=width, color=color, shrinkA=3, shrinkB=3, zorder=1,
        connectionstyle="arc3,rad=0",
    ))


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 10.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "RAPTOR Experimental Code Structure and End-to-End Workflow",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#0b2451")
    ax.text(0.5, 0.928,
            "Shared implementation and protocol feeding three reproducible aerodynamic reconstruction pipelines",
            ha="center", va="center", fontsize=10.5, color="#526477")

    # Column headings
    headings = ((0.095, "DATA & PROTOCOL"), (0.335, "SHARED CORE"),
                (0.615, "EXPERIMENT PIPELINES"), (0.885, "EVIDENCE & FIGURES"))
    for x, text in headings:
        ax.text(x, 0.885, text, ha="center", va="center", fontsize=9.2,
                fontweight="bold", color="#64748b")

    box(ax, (0.02, 0.705), (0.18, 0.145), "AirfRANS 2D", [
        "Official Reynolds / AoA tasks",
        "180 / 20 / 200 split",
        "4,096 nodes; 7 → 4 variables",
        "configs/*.yaml",
    ], "#eef5ff", "#4f78b8")
    box(ax, (0.02, 0.475), (0.18, 0.145), "Automobile surfaces", [
        "100 vehicles; 80 / 10 / 10",
        "4,096 sampled surface nodes",
        "single-channel surface field",
        "--archive PATH_TO_ARCHIVE",
    ], "#eefaf6", "#23856f")
    box(ax, (0.02, 0.245), (0.18, 0.145), "Aircraft surfaces", [
        "100 stratified aircraft",
        "J-20 / P180 / B737 / ERJ",
        "80 / 10 / 10; 4,096 nodes",
        "--archive PATH_TO_ARCHIVE",
    ], "#fff6ea", "#cb7a20")

    box(ax, (0.245, 0.69), (0.205, 0.175), "Unified representations", [
        "data.py",
        "training-only normalization",
        "deterministic sampling + split",
        "64-D latent projection",
        "64 physics tokens; 12-NN",
    ], "#f4f2ff", "#7458b5", mono=True)
    box(ax, (0.245, 0.445), (0.205, 0.175), "Models and optimization", [
        "raptor/model.py",
        "transolver_tiny/ · point baselines",
        "modern Tiny baseline wrappers",
        "AdamW + AMP + clipping",
        "seed 42 + early stopping",
    ], "#f4f2ff", "#7458b5", mono=True)
    box(ax, (0.245, 0.20), (0.205, 0.175), "Verification boundary", [
        "tests/ · CUDA smoke tests",
        "serial GPU execution",
        "result.json resume markers",
        "no CPU fallback",
        "no overwrite of completed runs",
    ], "#f4f2ff", "#7458b5", mono=True)

    row_colors = [
        ("#edf4ff", "#356cb6"),
        ("#eaf8f4", "#16837a"),
        ("#fff4e5", "#c86f12"),
    ]
    pipelines = [
        (0.705, "AirfRANS extrapolation", [
            "raptor/run.py",
            "run_aoa_pipeline.py",
            "run_core_ablation_pipeline.py",
            "Reynolds + AoA + 4 ablations",
        ]),
        (0.475, "100-car matched benchmark", [
            "drivaernetpp_3d/",
            "run_formal_100car.py",
            "PointNet / Transolver / RAPTOR",
            "paired + regional analysis",
        ]),
        (0.245, "100-aircraft transfer", [
            "aircraft_3d/",
            "run_formal_100aircraft.py",
            "Transolver-Tiny vs RAPTOR",
            "family + best/worst analysis",
        ]),
    ]
    for (y, title, lines), (face, edge) in zip(pipelines, row_colors):
        box(ax, (0.505, y), (0.225, 0.145), title, lines, face, edge, mono=True)

    evidence = [
        (0.705, "2D evidence", [
            "outputs/**/result.json",
            "four-variable errors + resources",
            "scripts/plot_airfrans_*.py",
            "figs/airfrans_*/",
        ]),
        (0.475, "Automobile evidence", [
            "per-vehicle MSE + paired tests",
            "geometry-region contributions",
            "full-mesh field/error figures",
            "analysis_consolidated_*/",
        ]),
        (0.245, "Aircraft evidence", [
            "per-aircraft + family statistics",
            "four-family overview",
            "best/worst 2 × 6 comparison",
            "analysis/*.csv · *.json · *.pdf",
        ]),
    ]
    for (y, title, lines), (face, edge) in zip(evidence, row_colors):
        box(ax, (0.785, y), (0.195, 0.145), title, lines, face, edge, mono=True,
            fontsize=7.8)

    # Horizontal data flow.
    for y, color in ((0.777, "#4f78b8"), (0.547, "#23856f"), (0.317, "#cb7a20")):
        arrow(ax, (0.20, y), (0.245, y), color)
        arrow(ax, (0.45, y), (0.505, y), color)
        arrow(ax, (0.73, y), (0.785, y), color)

    # Shared-core vertical coupling.
    arrow(ax, (0.3475, 0.69), (0.3475, 0.62), "#7458b5")
    arrow(ax, (0.3475, 0.445), (0.3475, 0.375), "#7458b5")

    box(ax, (0.13, 0.055), (0.74, 0.09), "Common reproducibility contract", [
        "Identical split · identical node count · identical seed · matched width/depth · validation selection · compact JSON/CSV evidence · publication PDF figures",
    ], "#f8fafc", "#94a3b8", fontsize=8.5)
    for x, color in ((0.11, "#4f78b8"), (0.35, "#7458b5"),
                     (0.615, "#16837a"), (0.885, "#c86f12")):
        arrow(ax, (x, 0.20), (x, 0.145), color, width=1.1)

    fig.savefig(HERE / "RAPTOR_experiment_code_structure.png", dpi=360,
                bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "RAPTOR_experiment_code_structure.pdf", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
