"""Empirical node-scaling benchmark for the RAPTOR paper.

The benchmark intentionally separates model execution from KNN construction:

* Transolver-Tiny and RAPTOR reuse the repository implementations.
* RAPTOR receives a deterministic, precomputed 12-NN index tensor.
* The dense baseline materializes the per-head N x N attention matrix.
* The repository's exact CPU ``torch.cdist + topk`` KNN procedure is timed in
  independent processes and reported separately.

Every model/size/phase combination runs in a fresh subprocess so that an OOM
does not invalidate the remainder of the sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raptor.model import RAPTOR  # noqa: E402
from transolver_tiny.model import TransolverTiny  # noqa: E402


MODEL_LABELS = {
    "transolver_tiny": "Transolver-Tiny",
    "raptor": "RAPTOR",
    "dense_attention": "Dense attention",
}


class DenseNodeAttention(nn.Module):
    """Conventional multi-head node attention with explicit N x N scores."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, dim = x.shape
        qkv = self.qkv(x).reshape(batch, nodes, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch, nodes, dim)
        return self.proj(out)


class DenseBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = DenseNodeAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class DenseAttentionBaseline(nn.Module):
    def __init__(
        self,
        cin: int = 3,
        cout: int = 1,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.input_projection = nn.Linear(cin, dim)
        self.blocks = nn.ModuleList([DenseBlock(dim, heads) for _ in range(depth)])
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, cout)
        )

    def forward(self, x: torch.Tensor, knn=None, groups=None) -> torch.Tensor:
        z = self.input_projection(x)
        for block in self.blocks:
            z = block(z)
        return self.output_projection(z)


def build_model(name: str) -> nn.Module:
    common = dict(cin=3, cout=1, dim=64, depth=4, heads=4)
    if name == "transolver_tiny":
        return TransolverTiny(**common, slices=64)
    if name == "raptor":
        return RAPTOR(**common, slices=64, knn_k=12)
    if name == "dense_attention":
        return DenseAttentionBaseline(**common)
    raise ValueError(f"Unknown model: {name}")


def ring_knn(nodes: int, k: int, device: torch.device) -> torch.Tensor:
    """Deterministic precomputed indices with the required [1, N, K] shape."""
    center = torch.arange(nodes, device=device, dtype=torch.long)[:, None]
    offsets = torch.arange(1, k + 1, device=device, dtype=torch.long)[None, :]
    return ((center + offsets) % nodes).unsqueeze(0)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def cuda_timer(callable_obj) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_obj()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def worker_result_base(model_name: str, nodes: int, phase: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "nodes": nodes,
        "phase": phase,
        "status": "error",
        "latency_ms_mean": None,
        "latency_ms_std": None,
        "latency_ms_p50": None,
        "latency_ms_p95": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "parameters": None,
        "error": None,
    }


def profile_worker(args: argparse.Namespace) -> int:
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = worker_result_base(args.worker_model, args.worker_nodes, args.worker_phase)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device("cuda")
        model = build_model(args.worker_model).to(device)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, args.worker_nodes, 3, device=device)
        target = torch.randn(1, args.worker_nodes, 1, device=device)
        knn = ring_knn(args.worker_nodes, 12, device) if args.worker_model == "raptor" else None

        if args.worker_phase == "forward":
            model.eval()

            def operation() -> None:
                with torch.inference_mode(), amp_context():
                    model(x, knn)

        elif args.worker_phase == "train_step":
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = make_scaler()

            def operation() -> None:
                optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    prediction = model(x, knn)
                    loss = torch.mean((prediction - target) ** 2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            raise ValueError(f"Unknown phase: {args.worker_phase}")

        for _ in range(args.warmup):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = [cuda_timer(operation) for _ in range(args.repeats)]
        result.update(
            status="ok",
            latency_ms_mean=float(statistics.mean(timings)),
            latency_ms_std=float(statistics.stdev(timings)) if len(timings) > 1 else 0.0,
            latency_ms_p50=percentile(timings, 50),
            latency_ms_p95=percentile(timings, 95),
            peak_allocated_mib=float(torch.cuda.max_memory_allocated() / 2**20),
            peak_reserved_mib=float(torch.cuda.max_memory_reserved() / 2**20),
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
    except Exception as exc:  # subprocess result must survive all failures
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    finally:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] in {"ok", "oom"} else 1


def knn_worker(args: argparse.Namespace) -> int:
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.knn_threads)
    result: dict[str, Any] = {
        "nodes": args.worker_nodes,
        "k": 12,
        "implementation": "CPU torch.cdist + torch.topk (self excluded)",
        "pairwise_distance_matrix_mib": args.worker_nodes * args.worker_nodes * 4 / 2**20,
        "torch_num_threads": torch.get_num_threads(),
        "warmup_iterations": args.knn_warmup,
        "measured_iterations": args.knn_repeats,
        "status": "error",
        "latency_ms_mean": None,
        "latency_ms_std": None,
        "latency_ms_p50": None,
        "latency_ms_p95": None,
        "error": None,
    }
    try:
        torch.manual_seed(args.seed)
        points = torch.randn(args.worker_nodes, 3, dtype=torch.float32)
        for _ in range(args.knn_warmup):
            warmup_distance = torch.cdist(points, points)
            warmup_indices = torch.topk(warmup_distance, k=13, largest=False).indices[:, 1:]
            _ = int(warmup_indices[0, 0])
            del warmup_distance, warmup_indices
        timings: list[float] = []
        for _ in range(args.knn_repeats):
            start = time.perf_counter()
            distance = torch.cdist(points, points)
            indices = torch.topk(distance, k=13, largest=False).indices[:, 1:]
            _ = int(indices[0, 0])
            timings.append((time.perf_counter() - start) * 1000.0)
            del distance, indices
        result.update(
            status="ok",
            latency_ms_mean=float(statistics.mean(timings)),
            latency_ms_std=float(statistics.stdev(timings)) if len(timings) > 1 else 0.0,
            latency_ms_p50=percentile(timings, 50),
            latency_ms_p95=percentile(timings, 95),
        )
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "memory" in message.lower() else "error"
        result.update(status=status, error=message)
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    finally:
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] in {"ok", "oom"} else 1


def run_isolated(command: list[str], result_path: Path, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "error": f"Timed out after {timeout} s",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"status": "error", "error": "Worker produced no result file"}
    if completed.returncode and result.get("status") == "ok":
        result["status"] = "error"
        result["error"] = f"Worker exited with code {completed.returncode}"
    if completed.stdout.strip():
        result["stdout"] = completed.stdout.strip()
    if completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()
    return result


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "protocol": {
            "nodes": None,
            "slices": 64,
            "neighbors": 12,
            "hidden_dim": 64,
            "depth": 4,
            "heads": 4,
            "batch_size": 1,
            "amp_dtype": "float16",
            "knn_in_model_timing": "precomputed deterministic indices",
        },
    }
    try:
        import scipy

        metadata["scipy"] = scipy.__version__
    except ImportError:
        metadata["scipy"] = None
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        metadata["gpu_total_memory_mib"] = props.total_memory / 2**20
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        metadata["nvidia_smi"] = query
    except Exception as exc:
        metadata["nvidia_smi"] = f"unavailable: {exc}"
    return metadata


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_status(row: dict[str, Any], key: str, digits: int = 1) -> str:
    if row.get("status") == "ok" and row.get(key) is not None:
        return f"{row[key]:.{digits}f}"
    return str(row.get("status", "--")).upper()


def write_latex_tables(output_dir: Path, model_rows: list[dict[str, Any]], knn_rows: list[dict[str, Any]]) -> None:
    by_key = {(r["model"], int(r["nodes"]), r["phase"]): r for r in model_rows}
    nodes = sorted({int(r["nodes"]) for r in model_rows})
    models = [m for m in MODEL_LABELS if any(r["model"] == m for r in model_rows)]
    display_nodes = [n for n in [1024, 2048, 4096, 8192, 16384] if n in nodes]
    lines = [
        "% Auto-generated by scripts/profile_node_scaling.py",
        "\\begin{tabular}{ll" + "r" * len(display_nodes) + "}",
        "\\toprule",
        "Model & Phase & " + " & ".join(f"$N={n}$" for n in display_nodes) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        for phase, phase_label in [("forward", "Forward (ms)"), ("train_step", "Train step (ms)")]:
            values = [latex_status(by_key[(model, n, phase)], "latency_ms_mean") for n in display_nodes]
            lines.append(f"{MODEL_LABELS[model]} & {phase_label} & " + " & ".join(values) + " \\\\ ")
        values = [latex_status(by_key[(model, n, "train_step")], "peak_allocated_mib") for n in display_nodes]
        lines.append(f"{MODEL_LABELS[model]} & Peak memory (MiB) & " + " & ".join(values) + " \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "node_scaling_table.tex").write_text("\n".join(lines), encoding="utf-8")

    knn_lines = [
        "% Auto-generated by scripts/profile_node_scaling.py",
        "\\begin{tabular}{rrr}",
        "\\toprule",
        "$N$ & KNN construction (ms) & Status \\\\ ",
        "\\midrule",
    ]
    for row in sorted(knn_rows, key=lambda item: int(item["nodes"])):
        latency = latex_status(row, "latency_ms_mean")
        knn_lines.append(f"{int(row['nodes'])} & {latency} & {row['status']} \\\\ ")
    knn_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "knn_scaling_table.tex").write_text("\n".join(knn_lines), encoding="utf-8")


def plot_results(output_dir: Path, model_rows: list[dict[str, Any]], knn_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    colors = {
        "transolver_tiny": "#1f4e8c",
        "raptor": "#008f87",
        "dense_attention": "#d55e00",
    }
    markers = {"transolver_tiny": "o", "raptor": "s", "dense_attention": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)

    for model in MODEL_LABELS:
        train = sorted(
            [r for r in model_rows if r["model"] == model and r["phase"] == "train_step"],
            key=lambda r: int(r["nodes"]),
        )
        ok = [r for r in train if r["status"] == "ok"]
        if ok:
            x = [r["nodes"] for r in ok]
            y = [r["latency_ms_mean"] for r in ok]
            yerr = [r.get("latency_ms_std") or 0.0 for r in ok]
            axes[0].errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers[model],
                color=colors[model],
                linewidth=2,
                elinewidth=1.1,
                capsize=3,
                capthick=1.1,
                label=MODEL_LABELS[model],
            )
            memory = [r["peak_allocated_mib"] for r in ok]
            axes[1].plot(x, memory, marker=markers[model], color=colors[model], linewidth=2, label=MODEL_LABELS[model])
            spill = [r for r in ok if r.get("memory_regime") == "shared_memory_spill"]
            for row in spill:
                axes[0].annotate(
                    "shared-memory spill",
                    (row["nodes"], row["latency_ms_mean"]),
                    xytext=(-18, -18),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    color=colors[model],
                    fontsize=8,
                )
        # Rows that were not measured are intentionally omitted.  In
        # particular, do not assign an artificial y-coordinate to the dense
        # N=16384 unattempted run: that would make an unobserved point look like a
        # measured latency or memory value.

    knn_ok = sorted([r for r in knn_rows if r["status"] == "ok"], key=lambda r: int(r["nodes"]))
    if knn_ok:
        axes[2].errorbar(
            [r["nodes"] for r in knn_ok],
            [r["latency_ms_mean"] for r in knn_ok],
            yerr=[r.get("latency_ms_std") or 0.0 for r in knn_ok],
            marker="D",
            color="#6a3d9a",
            linewidth=2,
            elinewidth=1.1,
            capsize=3,
            capthick=1.1,
            label="Exact CPU KNN",
        )

    metadata = load_json_if_valid(output_dir / "environment.json")
    gpu_total_memory_mib = metadata.get("gpu_total_memory_mib") if metadata else None
    if gpu_total_memory_mib:
        axes[1].axhline(
            gpu_total_memory_mib,
            color="#b22222",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        axes[1].text(
            0.02,
            gpu_total_memory_mib,
            "6141 MiB physical VRAM",
            transform=axes[1].get_yaxis_transform(),
            color="#b22222",
            fontsize=8,
            va="bottom",
        )

    axes[0].set_title("Full AMP training step")
    axes[0].set_ylabel("Latency (ms)")
    axes[1].set_title("Training-step peak allocation")
    axes[1].set_ylabel("Peak allocated GPU memory (MiB)")
    axes[2].set_title("Exact CPU KNN construction (measured separately)")
    axes[2].set_ylabel("CPU time (ms)")
    for ax in axes:
        ax.set_xlabel("Number of nodes, $N$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Empirical node-scaling profile (S=64, K=12, d=64, L=4, batch=1)\n"
        r"Latency error bars show mean $\pm$ 1 SD (models: n=30; KNN: n=10)",
        fontsize=12.5,
    )
    fig.savefig(output_dir / "Figure_node_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "Figure_node_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def load_json_if_valid(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def add_memory_regime(rows: list[dict[str, Any]], gpu_total_memory_mib: float | None) -> None:
    for row in rows:
        peak = row.get("peak_allocated_mib")
        if row.get("status") != "ok" or peak is None or gpu_total_memory_mib is None:
            row["memory_regime"] = None
        elif float(peak) > float(gpu_total_memory_mib):
            row["memory_regime"] = "shared_memory_spill"
        else:
            row["memory_regime"] = "within_physical_vram"


def collect_existing_results(
    output_dir: Path,
    nodes: list[int],
    models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workers_dir = output_dir / "workers"
    model_rows: list[dict[str, Any]] = []
    for model_name in models:
        for node_count in nodes:
            for phase in ("forward", "train_step"):
                path = workers_dir / f"{model_name}_n{node_count}_{phase}.json"
                row = load_json_if_valid(path)
                if row is None:
                    not_attempted = model_name == "dense_attention" and node_count >= 16384
                    row = worker_result_base(model_name, node_count, phase)
                    row["status"] = "not_attempted" if not_attempted else "missing"
                    row["error"] = (
                        "Not attempted to avoid further memory oversubscription after the "
                        "dense N=8192 training step exceeded physical VRAM."
                        if not_attempted
                        else "No valid worker result file was found."
                    )
                row["source"] = "recovered_worker" if path.exists() else "recovery_placeholder"
                model_rows.append(row)

    knn_rows: list[dict[str, Any]] = []
    for node_count in nodes:
        path = workers_dir / f"knn_n{node_count}.json"
        row = load_json_if_valid(path)
        if row is None:
            row = {
                "nodes": node_count,
                "k": 12,
                "implementation": "CPU torch.cdist + torch.topk (self excluded)",
                "status": "missing",
                "latency_ms_mean": None,
                "latency_ms_std": None,
                "latency_ms_p50": None,
                "latency_ms_p95": None,
                "pairwise_distance_matrix_mib": node_count * node_count * 4 / 2**20,
                "torch_num_threads": None,
                "error": "No valid KNN worker result file was found.",
            }
        knn_rows.append(row)
    return model_rows, knn_rows


def write_summary_outputs(
    output_dir: Path,
    model_rows: list[dict[str, Any]],
    knn_rows: list[dict[str, Any]],
) -> None:
    metadata_path = output_dir / "environment.json"
    metadata = load_json_if_valid(metadata_path) or environment_metadata()
    add_memory_regime(model_rows, metadata.get("gpu_total_memory_mib"))
    model_fields = [
        "model",
        "model_label",
        "nodes",
        "phase",
        "status",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_p50",
        "latency_ms_p95",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "memory_regime",
        "parameters",
        "source",
        "error",
    ]
    knn_fields = [
        "nodes",
        "k",
        "implementation",
        "status",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_p50",
        "latency_ms_p95",
        "pairwise_distance_matrix_mib",
        "torch_num_threads",
        "error",
    ]
    write_csv(output_dir / "node_scaling_results.csv", model_rows, model_fields)
    write_csv(output_dir / "knn_scaling_results.csv", knn_rows, knn_fields)
    (output_dir / "node_scaling_results.json").write_text(json.dumps(model_rows, indent=2), encoding="utf-8")
    (output_dir / "knn_scaling_results.json").write_text(json.dumps(knn_rows, indent=2), encoding="utf-8")
    recovery_notes = {
        "recovered_after_host_restart": True,
        "completed_model_workers": sum(row["status"] == "ok" for row in model_rows),
        "expected_model_workers": len(model_rows),
        "dense_n8192_training_memory_note": (
            "The run completed, but peak CUDA allocation exceeded physical VRAM and is "
            "classified as shared-memory spill rather than an in-VRAM result."
        ),
        "dense_n16384_policy": "Not attempted to avoid further memory oversubscription.",
        "knn_is_excluded_from_model_latency": True,
    }
    (output_dir / "recovery_notes.json").write_text(json.dumps(recovery_notes, indent=2), encoding="utf-8")
    write_latex_tables(output_dir, model_rows, knn_rows)
    plot_results(output_dir, model_rows, knn_rows)


def run_knn_stage(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "environment.json"
    metadata = load_json_if_valid(metadata_path) or environment_metadata()
    protocol = metadata.setdefault("protocol", {})
    protocol["knn_repeats"] = args.knn_repeats
    protocol["knn_warmup"] = args.knn_warmup
    protocol["knn_threads"] = args.knn_threads
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for index, node_count in enumerate(args.nodes, start=1):
        result_path = workers_dir / f"knn_n{node_count}.json"
        existing = load_json_if_valid(result_path)
        if existing is not None and existing.get("status") == "ok" and not args.force:
            print(f"[KNN {index}/{len(args.nodes)}] N={node_count}: reusing completed result", flush=True)
            continue
        print(f"[KNN {index}/{len(args.nodes)}] N={node_count}: starting isolated worker", flush=True)
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--knn-worker",
            "--worker-nodes",
            str(node_count),
            "--worker-output",
            str(result_path),
            "--knn-repeats",
            str(args.knn_repeats),
            "--knn-warmup",
            str(args.knn_warmup),
            "--knn-threads",
            str(args.knn_threads),
            "--seed",
            str(args.seed),
        ]
        result = run_isolated(command, result_path, args.knn_timeout)
        print(
            f"    status={result.get('status')} mean_ms={result.get('latency_ms_mean')} "
            f"matrix_mib={result.get('pairwise_distance_matrix_mib')}",
            flush=True,
        )
        if result.get("status") != "ok":
            print("Stopping this KNN stage after a non-success status.", flush=True)
            return 1
    return 0


def consolidate_existing(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    model_rows, knn_rows = collect_existing_results(output_dir, args.nodes, args.models)
    write_summary_outputs(output_dir, model_rows, knn_rows)
    print(
        f"Recovered {sum(row['status'] == 'ok' for row in model_rows)}/{len(model_rows)} "
        f"model workers and {sum(row['status'] == 'ok' for row in knn_rows)}/{len(knn_rows)} KNN workers.",
        flush=True,
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(exist_ok=True)
    metadata = environment_metadata()
    metadata["protocol"]["nodes"] = args.nodes
    metadata["protocol"]["warmup"] = args.warmup
    metadata["protocol"]["measured_iterations"] = args.repeats
    metadata["protocol"]["knn_repeats"] = args.knn_repeats
    (output_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model_rows: list[dict[str, Any]] = []
    total = len(args.models) * len(args.nodes) * 2
    current = 0
    for model_name in args.models:
        for nodes in args.nodes:
            for phase in ("forward", "train_step"):
                current += 1
                print(f"[{current}/{total}] {MODEL_LABELS[model_name]} N={nodes} {phase}", flush=True)
                result_path = workers_dir / f"{model_name}_n{nodes}_{phase}.json"
                result_path.unlink(missing_ok=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-model",
                    model_name,
                    "--worker-nodes",
                    str(nodes),
                    "--worker-phase",
                    phase,
                    "--worker-output",
                    str(result_path),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed),
                ]
                result = run_isolated(command, result_path, args.timeout)
                result.setdefault("model", model_name)
                result.setdefault("model_label", MODEL_LABELS[model_name])
                result.setdefault("nodes", nodes)
                result.setdefault("phase", phase)
                model_rows.append(result)
                print(f"    status={result['status']} mean_ms={result.get('latency_ms_mean')}", flush=True)

    knn_rows: list[dict[str, Any]] = []
    for index, nodes in enumerate(args.nodes, start=1):
        print(f"[KNN {index}/{len(args.nodes)}] N={nodes}", flush=True)
        result_path = workers_dir / f"knn_n{nodes}.json"
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--knn-worker",
            "--worker-nodes",
            str(nodes),
            "--worker-output",
            str(result_path),
            "--knn-repeats",
            str(args.knn_repeats),
            "--knn-warmup",
            str(args.knn_warmup),
            "--knn-threads",
            str(args.knn_threads),
            "--seed",
            str(args.seed),
        ]
        result = run_isolated(command, result_path, args.knn_timeout)
        result.setdefault("nodes", nodes)
        result.setdefault("k", 12)
        knn_rows.append(result)
        print(f"    status={result['status']} mean_ms={result.get('latency_ms_mean')}", flush=True)

    write_summary_outputs(output_dir, model_rows, knn_rows)
    print(f"Results written to {output_dir}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=list(MODEL_LABELS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--knn-repeats", type=int, default=10)
    parser.add_argument("--knn-warmup", type=int, default=1)
    parser.add_argument("--knn-threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900, help="Per model/phase worker timeout in seconds")
    parser.add_argument("--knn-timeout", type=int, default=900, help="Per KNN worker timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "node_scaling_profile"))
    parser.add_argument("--worker-model", choices=list(MODEL_LABELS))
    parser.add_argument("--worker-nodes", type=int)
    parser.add_argument("--worker-phase", choices=["forward", "train_step"])
    parser.add_argument("--worker-output")
    parser.add_argument("--knn-worker", action="store_true")
    parser.add_argument("--knn-only", action="store_true", help="Run only the requested staged KNN workers")
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Recover aggregate files from existing worker JSON without rerunning models",
    )
    parser.add_argument("--force", action="store_true", help="Replace completed KNN worker results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.knn_worker:
        if args.worker_nodes is None or not args.worker_output:
            raise SystemExit("KNN worker requires --worker-nodes and --worker-output")
        return knn_worker(args)
    if args.worker_model:
        if args.worker_nodes is None or not args.worker_phase or not args.worker_output:
            raise SystemExit("Model worker requires model, nodes, phase, and output")
        return profile_worker(args)
    if args.knn_only:
        return run_knn_stage(args)
    if args.consolidate_only:
        return consolidate_existing(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
