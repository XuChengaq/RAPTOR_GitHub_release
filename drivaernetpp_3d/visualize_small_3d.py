from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def equal_axes(ax, xyz: np.ndarray) -> None:
    mins, maxs = xyz.min(0), xyz.max(0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 0.62))
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    prediction = np.load(args.trial / "test_predictions.npz")
    x = prediction["x"]
    y_true = prediction["y_true"][..., 0]
    y_pred = prediction["y_pred"][..., 0]
    error = np.abs(y_pred - y_true)
    rows = list(csv.DictReader((args.trial / "history.csv").open(encoding="utf-8")))
    epoch = np.array([int(r["epoch"]) for r in rows])
    train = np.array([float(r["train_normalized_mse"]) for r in rows])
    val = np.array([float(r["validation_normalized_mse"]) for r in rows])

    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(5, 3, height_ratios=[0.72, 1, 1, 1, 1])
    ax_curve = fig.add_subplot(grid[0, :])
    ax_curve.plot(epoch, train, marker="o", ms=3, lw=2, label="Training MSE")
    ax_curve.plot(epoch, val, marker="s", ms=3, lw=2, label="Validation MSE")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Normalized MSE")
    ax_curve.set_title("RAPTOR 3D small-scale trial: 40 vehicles (32/4/4 split)", weight="bold")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False, ncol=2)

    for i in range(len(x)):
        field_min = min(y_true[i].min(), y_pred[i].min())
        field_max = max(y_true[i].max(), y_pred[i].max())
        field_norm = Normalize(field_min, field_max)
        err_norm = Normalize(0, np.quantile(error[i], 0.98))
        panels = [(y_true[i], field_norm, "Ground truth", "coolwarm"),
                  (y_pred[i], field_norm, "RAPTOR prediction", "coolwarm"),
                  (error[i], err_norm, "Absolute error", "magma")]
        for j, (values, norm, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(grid[i + 1, j], projection="3d")
            mark = ax.scatter(x[i, :, 0], x[i, :, 1], x[i, :, 2], c=values,
                              cmap=cmap, norm=norm, s=3.2, linewidths=0, rasterized=True)
            equal_axes(ax, x[i])
            mse = np.mean((y_pred[i] - y_true[i]) ** 2)
            suffix = f" | MSE={mse:.3f}" if j == 2 else ""
            ax.set_title(f"Test vehicle {i + 1}: {title}{suffix}", fontsize=10)
            fig.colorbar(mark, ax=ax, shrink=0.60, pad=0.01)

    png = args.trial / "Figure_RAPTOR3D_40car_overview.png"
    pdf = args.trial / "Figure_RAPTOR3D_40car_overview.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
