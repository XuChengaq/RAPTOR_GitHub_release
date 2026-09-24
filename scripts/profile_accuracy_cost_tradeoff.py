"""Matched inference profiling and accuracy--cost visualization.

This script is intentionally run only after all capacity-control and KNN
sensitivity training jobs have finished.  It reloads every best checkpoint
and profiles it under one common CUDA protocol, instead of mixing timing and
memory values collected by historical training runs.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from raptor.model import RAPTOR  # noqa: E402
from transolver_tiny.model import TransolverTiny  # noqa: E402


OUT = PROJECT / "outputs" / "capacity_knn_sensitivity"
FIG_DIR = PROJECT / "figs"
WARMUP = 20
REPEATS = 100
SAMPLE_INDEX = 0

TASK_LABELS = {
    "reynolds": "Reynolds-number extrapolation",
    "aoa": "AoA extrapolation",
}


def run_specs() -> list[dict]:
    """Return all checkpoints in the requested accuracy--cost comparison."""
    specs: list[dict] = []
    for task in ("reynolds", "aoa"):
        if task == "reynolds":
            trans_root = PROJECT / "outputs" / "baselines" / "transolver_tiny" / "reynolds-transolver_tiny-s42"
            k12_root = PROJECT / "outputs" / "formal" / "reynolds-raptor-s42"
        else:
            trans_root = PROJECT / "outputs" / "aoa_formal" / "transolver_tiny" / "aoa-transolver_tiny-s42"
            k12_root = PROJECT / "outputs" / "aoa_formal" / "raptor" / "aoa-raptor-s42"

        specs.append(
            {
                "task": task,
                "variant": "transolver_tiny",
                "label": "Transolver-Tiny",
                "kind": "transolver",
                "knn_k": None,
                "neighbor_mode": "none",
                "root": trans_root,
                "cache": PROJECT / "cache" / f"{task}_tr200_te200_n4096.npz",
            }
        )
        specs.append(
            {
                "task": task,
                "variant": "nodewise",
                "label": "Node-wise control",
                "kind": "raptor",
                "knn_k": 12,
                "neighbor_mode": "self",
                "root": OUT / "nodewise" / f"{task}-raptor-s42",
                "cache": PROJECT / "cache" / f"{task}_tr200_te200_n4096.npz",
            }
        )
        for k in (4, 8, 12, 24):
            root = (
                k12_root
                if k == 12
                else OUT / f"knn_k{k}" / f"{task}-raptor-s42"
            )
            cache = (
                PROJECT / "cache" / f"{task}_tr200_te200_n4096.npz"
                if k == 12
                else PROJECT
                / "cache"
                / "knn_sensitivity"
                / f"k{k}"
                / f"{task}_tr200_te200_n4096.npz"
            )
            specs.append(
                {
                    "task": task,
                    "variant": f"raptor_k{k}",
                    "label": f"RAPTOR ($K={k}$)",
                    "kind": "raptor",
                    "knn_k": k,
                    "neighbor_mode": "knn_mean",
                    "root": root,
                    "cache": cache,
                }
            )
    return specs


def load_input(cache_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(cache_path) as data:
        x = data["test_x"][SAMPLE_INDEX]
        x = ((x - data["xmean"]) / data["xstd"]).astype("float32")
        knn = data["test_knn"][SAMPLE_INDEX].astype("int64")
    return (
        torch.from_numpy(x)[None].to(device),
        torch.from_numpy(knn)[None].to(device),
    )


def load_model(spec: dict, device: torch.device) -> torch.nn.Module:
    if spec["kind"] == "transolver":
        model = TransolverTiny(dim=64, depth=4, heads=4, slices=64)
    else:
        model = RAPTOR(
            dim=64,
            depth=4,
            heads=4,
            slices=64,
            knn_k=int(spec["knn_k"]),
            adapter_enabled=True,
            adapter_alpha_init=0.0,
            adapter_neighbor_mode=spec["neighbor_mode"],
        )
    state = torch.load(spec["root"] / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def profile_one(spec: dict, device: torch.device) -> dict:
    result_path = spec["root"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = load_model(spec, device)
    x, knn = load_input(spec["cache"], device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            with torch.amp.autocast("cuda", enabled=True):
                _ = model(x, knn)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        elapsed_ms: list[float] = []
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.amp.autocast("cuda", enabled=True):
                output = model(x, knn)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))

    row = {
        "task": spec["task"],
        "variant": spec["variant"],
        "label": spec["label"],
        "knn_k": spec["knn_k"],
        "adapter_neighbor_mode": spec["neighbor_mode"],
        "macro_rel_l2": float(result["macro_rel_l2"]),
        "parameters": int(result["parameters"]),
        "latency_mean_ms": float(np.mean(elapsed_ms)),
        "latency_std_ms": float(np.std(elapsed_ms, ddof=1)),
        "latency_median_ms": float(np.median(elapsed_ms)),
        "latency_p95_ms": float(np.percentile(elapsed_ms, 95)),
        "inference_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "inference_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        "checkpoint": str((spec["root"] / "best.pt").relative_to(PROJECT)),
        "cache": str(spec["cache"].relative_to(PROJECT)),
    }
    print(
        "PROFILE_DONE"
        f" task={row['task']} variant={row['variant']}"
        f" latency={row['latency_mean_ms']:.3f}+/-{row['latency_std_ms']:.3f} ms"
        f" memory={row['inference_peak_allocated_mib']:.2f} MiB",
        flush=True,
    )

    del output, x, knn, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def draw_panel(ax: plt.Axes, rows: list[dict], cost_key: str, x_label: str) -> None:
    baseline = next(row for row in rows if row["variant"] == "transolver_tiny")
    nodewise = next(row for row in rows if row["variant"] == "nodewise")
    knn_rows = sorted(
        (row for row in rows if row["variant"].startswith("raptor_k")),
        key=lambda row: int(row["knn_k"]),
    )

    xs = [row[cost_key] for row in knn_rows]
    ys = [row["macro_rel_l2"] for row in knn_rows]
    ax.plot(xs, ys, color="#167D9A", linewidth=1.5, alpha=0.8, zorder=2)
    for row in knn_rows:
        xerr = row["latency_std_ms"] if cost_key == "latency_mean_ms" else None
        label_offset_y = -12 if int(row["knn_k"]) == 8 else 6
        ax.errorbar(
            row[cost_key],
            row["macro_rel_l2"],
            xerr=xerr,
            fmt="o",
            markersize=6.5,
            color="#008F87",
            ecolor="#71B8B3",
            elinewidth=1.0,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            f"$K={row['knn_k']}$",
            (row[cost_key], row["macro_rel_l2"]),
            xytext=(5, label_offset_y),
            textcoords="offset points",
            fontsize=7.8,
            color="#006D68",
        )

    ax.scatter(
        baseline[cost_key],
        baseline["macro_rel_l2"],
        marker="D",
        s=48,
        color="#56616F",
        edgecolors="white",
        linewidths=0.6,
        label="Transolver-Tiny",
        zorder=5,
    )
    ax.scatter(
        nodewise[cost_key],
        nodewise["macro_rel_l2"],
        marker="s",
        s=52,
        color="#E68632",
        edgecolors="white",
        linewidths=0.6,
        label="Node-wise capacity control",
        zorder=5,
    )
    ax.annotate(
        "Transolver-Tiny",
        (baseline[cost_key], baseline["macro_rel_l2"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.6,
        color="#3D4652",
    )
    ax.annotate(
        "Node-wise control",
        (nodewise[cost_key], nodewise["macro_rel_l2"]),
        xytext=(6, -13),
        textcoords="offset points",
        fontsize=7.6,
        color="#B45C14",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Macro relative $L_2$ error (lower is better)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, color="#D8DEE6", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def plot(rows: list[dict]) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)
    for row_index, task in enumerate(("reynolds", "aoa")):
        task_rows = [row for row in rows if row["task"] == task]
        draw_panel(
            axes[row_index, 0],
            task_rows,
            "latency_mean_ms",
            "Single-sample inference latency (ms)",
        )
        draw_panel(
            axes[row_index, 1],
            task_rows,
            "inference_peak_allocated_mib",
            "Peak allocated inference memory (MiB)",
        )
        axes[row_index, 0].set_title(f"{TASK_LABELS[task]}: accuracy vs. latency", loc="left", fontweight="bold")
        axes[row_index, 1].set_title(f"{TASK_LABELS[task]}: accuracy vs. memory", loc="left", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.025))
    fig.suptitle(
        "Accuracy--cost trade-off under a matched RTX 4050 inference protocol",
        fontsize=12,
        fontweight="bold",
        y=1.065,
    )
    fig.text(
        0.5,
        -0.025,
        "Fixed $N=4096$, $S=64$, $d=64$, $L=4$; mixed precision; "
        f"{WARMUP} warm-up and {REPEATS} measured forwards. KNN indices are precomputed and construction time is excluded.",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#4A5563",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        OUT / "accuracy_cost_tradeoff.png",
        FIG_DIR / "accuracy_cost_tradeoff.png",
    ):
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    for path in (
        OUT / "accuracy_cost_tradeoff.pdf",
        FIG_DIR / "accuracy_cost_tradeoff.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_outputs(rows: list[dict], device: torch.device) -> None:
    fields = [
        "task",
        "variant",
        "label",
        "knn_k",
        "adapter_neighbor_mode",
        "macro_rel_l2",
        "parameters",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "inference_peak_allocated_mib",
        "inference_peak_reserved_mib",
        "checkpoint",
        "cache",
    ]
    with (OUT / "accuracy_cost_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nodes": 4096,
            "slices": 64,
            "hidden_dim": 64,
            "depth": 4,
            "amp": True,
            "batch_size": 1,
            "sample_index": SAMPLE_INDEX,
            "warmup_forwards": WARMUP,
            "measured_forwards": REPEATS,
            "latency_statistic": "mean +/- one sample standard deviation",
            "memory_statistic": "torch.cuda.max_memory_allocated after warm-up",
            "knn_construction": "precomputed; excluded from forward latency and memory",
        },
        "rows": rows,
    }
    (OUT / "accuracy_cost_profile.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for matched accuracy--cost profiling")
    specs = run_specs()
    missing = [
        str(path.relative_to(PROJECT))
        for spec in specs
        for path in (spec["root"] / "best.pt", spec["root"] / "result.json", spec["cache"])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Profiling inputs are incomplete:\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rows = [profile_one(spec, device) for spec in specs]
    write_outputs(rows, device)
    plot(rows)
    print(f"ACCURACY_COST_PROFILE_COMPLETE rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
