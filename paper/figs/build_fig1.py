from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent

NAVY = "#082F7D"
NAVY_DARK = "#071E4B"
BLUE = "#075BD8"
TEAL = "#008E8A"
TEAL_DARK = "#006E6A"
TEAL_LIGHT = "#F0FBFA"
ORANGE = "#F04E23"
PURPLE = "#6F2BD1"
INK = "#111827"
PANEL = "#FCFDFE"


def rounded_box(ax, x, y, w, h, text="", *, face="white", edge=INK,
                color=INK, fontsize=9.5, linewidth=1.15, radius=0.07,
                weight="normal", zorder=3, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight=weight,
                linespacing=1.18, zorder=zorder + 1)
    return patch


def sum_node(ax, x, y, r=0.145, *, edge=INK, symbol="+"):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=edge,
                        linewidth=1.25, zorder=7))
    ax.text(x, y - 0.004, symbol, ha="center", va="center",
            fontsize=14, color=edge, fontweight="bold", zorder=8)


def arrow(ax, start, end, *, color=INK, lw=1.25, mutation=10, zorder=5,
          linestyle="-"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=mutation, shrinkA=0, shrinkB=0,
                                linestyle=linestyle), zorder=zorder)


def orth_arrow(ax, points, *, color=INK, lw=1.20, mutation=10, zorder=5,
               linestyle="-"):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle=linestyle, solid_capstyle="butt", zorder=zorder)
    arrow(ax, points[-2], points[-1], color=color, lw=lw,
          mutation=mutation, zorder=zorder, linestyle=linestyle)


def dot(ax, x, y, color=INK, r=0.026):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=7))


def square_grid(ax, x, y, rows=4, cols=4, *, size=0.105, gap=0.055,
                color=NAVY, edge="white"):
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(Rectangle(
                (x + col * (size + gap), y + (rows - 1 - row) * (size + gap)),
                size, size, facecolor=color, edgecolor=edge,
                linewidth=0.35, zorder=5,
            ))


def vector_squares(ax, x, y, *, n=6, size=0.12, gap=0.045,
                   color=TEAL, ellipsis=True):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (size + gap), y), size, size,
                               facecolor=color, edgecolor=TEAL_DARK,
                               linewidth=0.45, zorder=5))
    if ellipsis:
        ax.text(x + n * (size + gap) + 0.02, y + size / 2, r"$\cdots$",
                ha="left", va="center", fontsize=10, color=INK, zorder=6)


def token_rows(ax, x, y):
    colors = ["#0A4FA7", "#0B5AB8", "#0B66C8", "#0A4FA7", "#0B5AB8"]
    for row_y in (y + 0.28, y):
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((x + i * 0.18, row_y), 0.15, 0.15,
                                   facecolor=color, edgecolor=NAVY_DARK,
                                   linewidth=0.35, zorder=5))
        ax.text(x + 0.90, row_y + 0.075, r"$\cdots$", ha="left", va="center",
                fontsize=9.5, color=NAVY_DARK, zorder=6)


def airfoil_points(ax):
    # Compact schematic point cloud (not a sampled training object).
    t = np.linspace(0.04, 0.96, 27)
    x = 0.58 + 2.02 * t
    camber = 5.85 + 0.22 * np.sin(np.pi * t) - 0.10 * t
    thick = 0.24 * np.sin(np.pi * t) ** 0.72
    ax.scatter(x, camber + thick, s=8, color=NAVY, zorder=5)
    ax.scatter(x, camber - thick, s=8, color=NAVY, zorder=5)


def raptor_block(ax, x, y, w=3.30, h=1.80):
    rounded_box(ax, x, y, w, h, face=PANEL, edge=INK,
                linewidth=1.00, radius=0.09, zorder=1)
    branch_x = x + 0.38
    main_x = x + 0.72
    box_w = 1.72
    plus_x = x + w - 0.37
    cy_top = y + 1.28
    cy_bottom = y + 0.54

    rounded_box(ax, main_x, cy_top - 0.28, box_w, 0.56,
                "Physics-Attention\nBackbone", face=NAVY, edge=NAVY,
                color="white", fontsize=9.3, weight="semibold")
    rounded_box(ax, main_x, cy_bottom - 0.28, box_w, 0.56,
                "Local Residual\nAdapter (12-NN)", face=TEAL, edge=TEAL_DARK,
                color="white", fontsize=8.9, weight="semibold")

    dot(ax, branch_x, y + h / 2)
    orth_arrow(ax, [(branch_x, y + h / 2), (branch_x, cy_top),
                    (main_x, cy_top)], lw=1.15)
    orth_arrow(ax, [(branch_x, y + h / 2), (branch_x, cy_bottom),
                    (main_x, cy_bottom)], lw=1.15)

    sum_node(ax, plus_x, y + h / 2, r=0.14)
    orth_arrow(ax, [(main_x + box_w, cy_top), (plus_x - 0.28, cy_top),
                    (plus_x - 0.28, y + h / 2 + 0.12),
                    (plus_x - 0.13, y + h / 2 + 0.05)], lw=1.12)
    orth_arrow(ax, [(main_x + box_w, cy_bottom), (plus_x - 0.28, cy_bottom),
                    (plus_x - 0.28, y + h / 2 - 0.12),
                    (plus_x - 0.13, y + h / 2 - 0.05)], color=TEAL, lw=1.12)
    return (x, y + h / 2), (plus_x + 0.14, y + h / 2)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(18.2, 8.25), facecolor="white")
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 8.25)
    ax.axis("off")

    rounded_box(ax, 0.08, 0.08, 18.02, 8.07, face="white", edge=INK,
                linewidth=0.85, radius=0.12, linestyle=(0, (6, 4)), zorder=0)
    ax.plot([0.10, 18.08], [4.15, 4.15], color=INK, linewidth=0.75,
            linestyle=(0, (6, 4)), zorder=1)
    ax.plot([9.65, 9.65], [0.10, 4.15], color=INK, linewidth=0.75,
            linestyle=(0, (6, 4)), zorder=1)

    # (a) Overall architecture.
    ax.text(0.20, 7.96, "(a)  RAPTOR Overall Architecture", fontsize=17.2,
            color=NAVY, fontweight="bold", va="top")

    rounded_box(ax, 0.27, 5.15, 2.55, 1.78, face="white", edge=INK,
                linewidth=1.0, radius=0.08)
    ax.text(1.545, 6.76, "AirfRANS Input:", ha="center", va="center",
            fontsize=11.0, color=INK)
    ax.text(1.545, 6.47, r"4096 nodes $\times$ 7 features", ha="center",
            va="center", fontsize=10.3, color=INK)
    airfoil_points(ax)
    ax.text(1.545, 5.31, r"$(N=4096)$", ha="center", va="top",
            fontsize=9.7, color=INK)

    rounded_box(ax, 3.52, 5.62, 1.24, 0.92,
                "Input\nProjection\n" + r"$7\;\rightarrow\;64$",
                face="white", edge=BLUE, color=NAVY, fontsize=9.4,
                linewidth=1.1)
    arrow(ax, (2.82, 6.08), (3.52, 6.08), lw=1.2)

    ax.text(8.92, 7.58, r"RAPTOR Blocks $\times$ 4", ha="center",
            va="center", fontsize=15.0, color=NAVY, fontweight="bold")
    in1, out1 = raptor_block(ax, 5.28, 5.17)
    in2, out2 = raptor_block(ax, 9.78, 5.17)
    arrow(ax, (4.76, 6.07), in1, lw=1.2)
    arrow(ax, out1, (8.72, 6.07), lw=1.2)
    ax.text(9.20, 6.07, r"$\cdots$", fontsize=20, color=INK,
            ha="center", va="center")
    arrow(ax, (9.60, 6.07), in2, lw=1.2)

    rounded_box(ax, 13.55, 5.67, 1.14, 0.80, "Output\nHead",
                face="white", edge=PURPLE, fontsize=10.0, linewidth=1.1)
    arrow(ax, out2, (13.55, 6.07), lw=1.2)

    arrow(ax, (14.69, 6.07), (15.10, 6.07), lw=1.15)
    ax.plot([15.10, 15.10], [5.18, 6.90], color=INK, linewidth=1.1, zorder=4)
    outputs = [
        (6.84, r"$u_x$", "(x-velocity)"),
        (6.31, r"$u_y$", "(y-velocity)"),
        (5.78, r"$p/\rho$", "(pressure/density)"),
        (5.25, r"$\nu_t$", "(turbulent viscosity)"),
    ]
    for oy, symbol, label in outputs:
        arrow(ax, (15.10, oy), (15.48, oy), lw=1.05)
        rounded_box(ax, 15.48, oy - 0.20, 0.78, 0.40, symbol,
                    face="white", edge=PURPLE, fontsize=10.5,
                    linewidth=0.95, radius=0.055)
        ax.text(16.43, oy, label, ha="left", va="center", fontsize=8.4,
                color=INK)

    # (b) Physics-Attention backbone.
    ax.text(0.20, 3.92, "(b)  Physics-Attention Backbone", fontsize=15.2,
            color=NAVY, fontweight="bold", va="top")

    ax.text(0.61, 3.35, "Node\nfeatures", ha="center", va="center",
            fontsize=8.9, color=INK)
    square_grid(ax, 0.32, 2.45, rows=4, cols=4, size=0.105, gap=0.055)
    ax.text(0.64, 2.24, r"$N\times64$", ha="center", va="top",
            fontsize=8.9, color=INK)

    rounded_box(ax, 1.47, 2.38, 1.15, 0.87,
                "Soft\nPoint-to-Slice\nAssignment", face="white", edge=BLUE,
                color=NAVY, fontsize=8.4, linewidth=1.0)
    arrow(ax, (0.98, 2.80), (1.47, 2.80), lw=1.12)

    ax.text(3.60, 3.34, "64 Physics Tokens\n(slices)", ha="center",
            va="center", fontsize=8.9, color=INK)
    token_rows(ax, 3.05, 2.43)
    ax.text(3.57, 2.23, r"$64\times64$", ha="center", va="top",
            fontsize=8.7, color=INK)
    arrow(ax, (2.62, 2.80), (3.05, 2.80), lw=1.12)

    rounded_box(ax, 4.50, 2.38, 1.15, 0.87,
                "Multi-Head\nSlice Attention\n(4 heads)", face="white",
                edge=BLUE, color=NAVY, fontsize=8.2, linewidth=1.0)
    arrow(ax, (4.10, 2.80), (4.50, 2.80), lw=1.12)

    rounded_box(ax, 5.94, 2.43, 0.83, 0.77, "Deslicing\nto Nodes",
                face="white", edge=BLUE, color=NAVY, fontsize=8.0,
                linewidth=1.0)
    arrow(ax, (5.65, 2.80), (5.94, 2.80), lw=1.12)

    square_grid(ax, 7.03, 2.52, rows=4, cols=3, size=0.085, gap=0.04)
    ax.text(7.20, 3.20, r"$N\times64$", ha="center", va="bottom",
            fontsize=8.7, color=INK)
    arrow(ax, (6.77, 2.80), (7.03, 2.80), lw=1.10)
    sum_node(ax, 7.77, 2.80, r=0.135)
    arrow(ax, (7.43, 2.80), (7.635, 2.80), lw=1.08)

    rounded_box(ax, 8.06, 2.38, 0.83, 0.84, "FFN\n(2-layer\nMLP)",
                face="white", edge=BLUE, color=NAVY, fontsize=8.1,
                linewidth=1.0)
    arrow(ax, (7.905, 2.80), (8.06, 2.80), lw=1.08)
    sum_node(ax, 9.24, 2.80, r=0.135)
    arrow(ax, (8.89, 2.80), (9.105, 2.80), lw=1.08)

    dot(ax, 0.98, 2.80)
    orth_arrow(ax, [(0.98, 2.80), (0.98, 1.42), (7.77, 1.42),
                    (7.77, 2.665)], lw=1.05)
    orth_arrow(ax, [(7.92, 2.80), (7.94, 2.80), (7.94, 1.09),
                    (9.24, 1.09), (9.24, 2.665)], lw=1.05)

    rounded_box(ax, 1.70, 0.55, 1.34, 0.53,
                "Score matrix\n" + r"$N\times64$", face="white", edge=BLUE,
                color=INK, fontsize=7.9, linewidth=0.9, radius=0.055,
                linestyle=(0, (4, 3)))
    orth_arrow(ax, [(2.37, 1.08), (2.37, 1.80), (2.05, 1.80),
                    (2.05, 2.38)], color=BLUE, lw=0.9, mutation=8,
               linestyle=(0, (4, 3)))
    rounded_box(ax, 4.55, 0.55, 1.38, 0.53,
                "Attention matrix\n" + r"$64\times64$", face="white", edge=BLUE,
                color=INK, fontsize=7.7, linewidth=0.9, radius=0.055,
                linestyle=(0, (4, 3)))
    orth_arrow(ax, [(5.24, 1.08), (5.24, 1.82), (5.07, 1.82),
                    (5.07, 2.38)], color=BLUE, lw=0.9, mutation=8,
               linestyle=(0, (4, 3)))

    # (c) Zero-initialized local adapter.
    ax.text(9.84, 3.92, "(c)  Zero-Initialized Local Adapter", fontsize=15.0,
            color=TEAL, fontweight="bold", va="top")

    ax.text(10.03, 3.31, "Node feature", ha="left", va="center",
            fontsize=8.9, color=INK)
    ax.text(10.03, 3.08, r"$\mathbf{x}_i\in\mathbb{R}^{64}$", ha="left",
            va="center", fontsize=9.0, color=INK)
    vector_squares(ax, 10.05, 2.77, n=5, size=0.115, gap=0.04)

    ax.text(10.03, 2.07, "Mean of\n12-NN neighbors", ha="left", va="center",
            fontsize=8.5, color=INK, linespacing=1.15)
    ax.text(10.03, 1.70, r"$\bar{\mathbf{x}}_i\in\mathbb{R}^{64}$",
            ha="left", va="center", fontsize=9.0, color=INK)
    vector_squares(ax, 10.05, 1.38, n=5, size=0.115, gap=0.04)

    rounded_box(ax, 12.29, 2.12, 1.20, 0.86, "Concatenate\n(128 dim)",
                face=TEAL_LIGHT, edge=TEAL, fontsize=8.5, linewidth=1.0)
    vector_squares(ax, 12.50, 2.28, n=4, size=0.10, gap=0.035,
                   color="#39B9B2", ellipsis=False)
    residual_x = 11.48
    ax.plot([11.18, residual_x], [2.83, 2.83], color=TEAL,
            linewidth=1.08, zorder=5)
    dot(ax, residual_x, 2.83, color=TEAL_DARK)
    orth_arrow(ax, [(residual_x, 2.83), (11.82, 2.83), (11.82, 2.72),
                    (12.29, 2.72)], color=TEAL, lw=1.08)
    orth_arrow(ax, [(11.18, 1.44), (11.82, 1.44), (11.82, 2.38),
                    (12.29, 2.38)], color=TEAL, lw=1.08)

    rounded_box(ax, 13.86, 2.12, 1.13, 0.86, "LayerNorm\n+\n2-layer MLP",
                face="white", edge=TEAL, fontsize=8.2, linewidth=1.0)
    arrow(ax, (13.49, 2.55), (13.86, 2.55), color=TEAL, lw=1.10)

    gate_x, gate_y = 15.46, 2.55
    sum_node(ax, gate_x, gate_y, r=0.17, edge=ORANGE, symbol=r"$\times$")
    arrow(ax, (14.99, 2.55), (15.29, 2.55), color=TEAL, lw=1.10)
    rounded_box(ax, 14.86, 3.03, 1.20, 0.52,
                "Gate: " + r"$\tanh(\alpha_l)$" + "\n" +
                r"$\alpha_l=0$ initially",
                face="#FFF7F2", edge=ORANGE, color=ORANGE,
                fontsize=7.6, linewidth=0.85, radius=0.055,
                weight="semibold")
    arrow(ax, (gate_x, 3.03), (gate_x, 2.72), color=ORANGE, lw=1.0,
          mutation=8)

    plus_x = 16.32
    sum_node(ax, plus_x, gate_y, r=0.15)
    arrow(ax, (15.63, gate_y), (plus_x - 0.15, gate_y), color=TEAL, lw=1.10)

    # Shift the residual trunk away from the feature labels. A small gap at
    # the lower local branch keeps the two paths visually distinct.
    ax.plot([residual_x, residual_x], [2.83, 1.54], color=INK,
            linewidth=1.05, zorder=4)
    ax.plot([residual_x, residual_x], [1.34, 0.48], color=INK,
            linewidth=1.05, zorder=4)
    ax.plot([residual_x, plus_x], [0.48, 0.48], color=INK,
            linewidth=1.05, zorder=4)
    arrow(ax, (plus_x, 0.48), (plus_x, gate_y - 0.15), lw=1.05)
    ax.text(13.10, 0.68, "backbone output (residual path)", ha="center",
            va="bottom", fontsize=8.2, color=INK)

    rounded_box(ax, 16.75, 2.17, 1.10, 0.76,
                "Updated node\nfeature\n" + r"$\mathbf{Z}^{(l+1)}$",
                face=TEAL_LIGHT, edge=TEAL, color=INK,
                fontsize=7.8, linewidth=0.95, radius=0.055)
    arrow(ax, (plus_x + 0.15, gate_y), (16.75, gate_y), lw=1.12)

    fig.savefig(OUT / "Fig1.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT / "Fig1.png", dpi=360, bbox_inches="tight",
                pad_inches=0.035, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
