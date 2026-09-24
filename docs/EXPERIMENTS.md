# Experiment and artefact guide

This document maps the implementation to the experiments and retained evidence.
Seed 42 is the reference run; the AirfRANS model comparison additionally uses
training seeds 41 and 43 on the same fixed split. Values below are read from
saved JSON/CSV files rather than reconstructed from plot labels.

## 1. Method implementation

- `raptor/model.py`: Physics-Attention blocks, 12-NN residual adapters, and
  zero-initialized bounded gates.
- `raptor/run.py`: CUDA-only AirfRANS training, validation-based early stopping,
  mixed precision, gradient clipping, and evaluation.
- `data.py`: deterministic AirfRANS selection, 4,096-node sampling,
  training-only normalization, token grouping, and KNN construction.
- `transolver_tiny/`, `transolver_plus_tiny/`, `linearno_tiny/`, and
  `mono_tiny/`: matched Tiny baseline implementations.

## 2. AirfRANS experiments

### Protocol

- Tasks: Reynolds-number and angle-of-attack extrapolation.
- Split: 180 training and 20 validation simulations after selecting 200
  official training simulations; 200 Reynolds and 196 AoA test cases.
- Nodes: 4,096 points per simulation with a 0.20 surface fraction.
- Architecture: width 64, depth 4, 64 physics tokens, four heads, and 12-NN
  local neighborhoods.
- Optimization: AdamW, CUDA AMP, validation early stopping, maximum 100 epochs.

### RAPTOR results

| Task | ux | uy | p/rho | turbulent viscosity | Macro relative L2 |
|---|---:|---:|---:|---:|---:|
| Reynolds | 0.1395 | 0.2292 | 0.2251 | 0.2939 | **0.2219** |
| AoA | 0.1510 | 0.2189 | 0.3350 | 0.4828 | **0.2969** |

Primary evidence:

- `outputs/formal/reynolds-raptor-s42/result.json`
- `outputs/aoa_formal/raptor/aoa-raptor-s42/result.json`
- `outputs/linearno_seed42_comparison.csv`

Across seeds 41/42/43, RAPTOR's macro relative L2 is
`0.220968 +/- 0.001926` on Reynolds and `0.308298 +/- 0.010764` on AoA
(mean +/- sample standard deviation). The [three-seed evidence guide](../outputs/airfrans_multiseed_41_43/README.md)
links the full-precision per-run JSON files, epoch histories, console logs,
traditional-baseline runs, paired-improvement CSV, and aggregation files.

### Core ablation

| Variant | Reynolds macro error | AoA macro error |
|---|---:|---:|
| Without local adapter | 0.2280 | 0.3138 |
| Nonzero gate initialization, alpha=0.1 | 0.2456 | 0.3222 |
| RAPTOR, zero initialization | **0.2219** | **0.2969** |

The Reynolds no-adapter value is the independently trained Transolver-Tiny
backbone reported in the main comparison and is not treated as a seed-matched
paired ablation. The AoA no-adapter checkpoint is matched. The zero-versus-
nonzero initialization comparison supports the baseline-preserving design
under the evaluated protocol, but it is not a universal optimization guarantee.

## 3. Three-dimensional automobile experiment

The formal benchmark selects 100 automobile surfaces from Mixed-3D, uses a
deterministic 80/10/10 split, samples 4,096 nodes per vehicle, and predicts the
standardized surface pressure coefficient $C_p$. The automobile geometries are
derived from DrivAerNet++; Mixed-3D provides the panel-method target at
$\mathbf{v}_\infty=[1,0,0]$. This is a surface reconstruction experiment rather
than a complete volumetric-flow reconstruction.

| Model | Mean test sMSE | Parameters | Peak memory (MiB) |
|---|---:|---:|---:|
| PointNet | 0.4863 | 21,057 | 23.95 |
| Transolver-Tiny | 0.145534 | 195,345 | 103.76 |
| RAPTOR without adapter | 0.145534 | 195,345 | 103.76 |
| RAPTOR | **0.135684** | 245,525 | 117.77 |

RAPTOR reduces mean test sMSE by 6.77% relative to Transolver-Tiny. Regional
analysis shows spatially nonuniform gains, motivating adaptive local gating
rather than a claim of universal regional superiority.

Primary evidence and scripts:

- `drivaernetpp_3d/run_formal_100car.py`
- `drivaernetpp_3d/analyze_formal_100car.py`
- `drivaernetpp_3d/analyze_regions_and_cases.py`
- `drivaernetpp_3d/consolidate_100car_analysis.py`
- `drivaernetpp_3d/outputs/formal_100car/analysis_consolidated_20260906/`

## 4. Three-dimensional aircraft geometry evaluation

The experiment uses a stratified set of 100 OpenVSP-derived aircraft from the
Mixed-3D J-20, P180, Boeing 737, and ERJ families. The split is 80/10/10; both
models use the same 4,096 sampled points, seed, and optimization protocol. The
target is the standardized panel-method surface pressure coefficient $C_p$.

| Model | Mean test sMSE | Parameters | Peak memory (MiB) |
|---|---:|---:|---:|
| Transolver-Tiny | 0.843596 | 195,345 | 103.76 |
| RAPTOR | **0.771331** | 245,525 | 117.77 |

The overall reduction is 8.57%. RAPTOR wins on 6/10 test aircraft. Family mean
changes are +3.76% for J-20, -8.46% for P180, +10.67% for Boeing 737, and
+7.00% for ERJ. The best paired case is Boeing 737 sample 0483 (+17.54%); the
worst is P180 sample 1118 (-9.94%).

Primary evidence and scripts:

- `aircraft_3d/run_formal_100aircraft.py`
- `aircraft_3d/analyze_aircraft_results.py`
- `aircraft_3d/make_four_aircraft_high_quality.py`
- `aircraft_3d/make_aircraft_best_worst_2x6.py`
- `aircraft_3d/outputs/formal_100aircraft/analysis/summary.json`
- `aircraft_3d/outputs/formal_100aircraft/analysis/per_aircraft_metrics.csv`

## 5. Empirical node-scaling profile

The isolated profile fixes $S=64$, $K=12$, $d=64$, $L=4$, and batch size one.
Model execution uses 10 warm-up and 30 measured CUDA iterations. Exact CPU
`torch.cdist + topk` KNN construction is timed separately with one warm-up and
10 measured runs.

At $N=16{,}384$, RAPTOR completes a mixed-precision training step in
$66.99\pm1.07$ ms with 410.31 MiB peak allocated GPU memory. Explicit dense
attention reaches an 8,274.59 MiB CUDA allocation and
$12.51\pm0.35$ s per training step at $N=8{,}192$ through WDDM shared-memory
spillover; $N=16{,}384$ was not attempted. The exact KNN distance matrix grows
from 4 MiB at $N=1{,}024$ to 1,024 MiB at $N=16{,}384$.

Primary evidence and script:

- `scripts/profile_node_scaling.py`
- `outputs/node_scaling_profile/node_scaling_results.csv`
- `outputs/node_scaling_profile/knn_scaling_results.csv`
- `outputs/node_scaling_profile/environment.json`

## 6. Capacity control and K sensitivity

The seed-42 node-wise control retains RAPTOR's gate, MLP, and parameter count
but removes cross-node aggregation. Additional RAPTOR runs use `K=4`, `8`,
and `24`; the original `K=12` checkpoints are reused. The
[study evidence guide](../outputs/capacity_knn_sensitivity/README.md) gives
the raw result JSON, training histories, summary CSV/JSON, and separate
100-repeat inference latency and memory profile. These runs should not be
confused with a multi-seed accuracy comparison.

## 7. Tests

- `tests/test_model.py`: model shapes, gradients, gates, and equivalence checks.
- `tests/test_modern_baselines.py`: matched modern-baseline checks.
- `drivaernetpp_3d/test_formal_100car.py`: formal automobile runner checks.
- `aircraft_3d/test_aircraft_runner.py`: aircraft data and runner checks.

```powershell
python -m pytest -q tests drivaernetpp_3d/test_formal_100car.py aircraft_3d/test_aircraft_runner.py
```

Formal training additionally requires CUDA and the original datasets.

## 8. Reproducibility boundaries

- Raw datasets are excluded because they are large and governed by their source
  distribution terms.
- Checkpoints, caches, and prediction tensors are excluded; compact JSON/CSV
  results, epoch histories, selected console logs, and publication PDFs are
  retained.
- Full-mesh 3D colors are interpolation-based visualization products. Numerical
  errors are computed on the matched sampled nodes.
- AirfRANS model comparisons use three training seeds on one fixed split.
  The 3D and capacity/K experiments use one training seed; broader seed,
  split, and full-dataset studies remain future work.
- The node-scaling results characterize one hardware/software configuration
  and do not establish hardware-independent speed superiority.

