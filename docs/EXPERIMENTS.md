# Experiment and artefact guide

This document maps the implementation to the experiments and retained evidence.
All formal comparisons use seed 42 and serial CUDA execution. Values below are
read from saved JSON/CSV files rather than reconstructed from plot labels.

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
- Split: 180 training, 20 validation, and 200 test simulations after selecting
  200 official training simulations and reserving 10% for validation.
- Nodes: 4,096 points per simulation with a 0.20 surface fraction.
- Architecture: width 64, depth 4, 64 physics tokens, four heads, and 12-NN
  local neighborhoods.
- Optimization: AdamW, CUDA AMP, validation early stopping, maximum 100 epochs.

### RAPTOR results

| Task | ux | uy | p | turbulent viscosity | Macro relative L2 |
|---|---:|---:|---:|---:|---:|
| Reynolds | 0.1395 | 0.2292 | 0.2251 | 0.2939 | **0.2219** |
| AoA | 0.1510 | 0.2189 | 0.3350 | 0.4828 | **0.2969** |

Primary evidence:

- `outputs/formal/reynolds-raptor-s42/result.json`
- `outputs/aoa_formal/raptor/aoa-raptor-s42/result.json`
- `outputs/linearno_seed42_comparison.csv`

### Core ablation

| Variant | Reynolds macro error | AoA macro error |
|---|---:|---:|
| Without local adapter | 0.2432 | 0.3138 |
| Nonzero gate initialization, alpha=0.1 | 0.2456 | 0.3222 |
| RAPTOR, zero initialization | **0.2219** | **0.2969** |

The ablation supports the empirical contribution of the local adapter and zero
initialization under the evaluated protocol. It is not a universal optimization
guarantee.

## 3. Three-dimensional automobile experiment

The formal benchmark selects 100 automobile surfaces, uses a deterministic
80/10/10 split, samples 4,096 nodes per vehicle, and predicts one normalized
surface-field channel. It is a surface reconstruction experiment rather than a
complete volumetric-flow reconstruction.

| Model | Mean test MSE | Parameters | Peak memory (MiB) |
|---|---:|---:|---:|
| PointNet | 0.4863 | 21,057 | 23.95 |
| Transolver-Tiny | 0.1455 | 195,345 | 103.76 |
| RAPTOR without adapter | 0.1455 | 195,345 | 103.76 |
| RAPTOR | **0.1357** | 245,525 | 117.77 |

RAPTOR reduces mean test MSE by 6.77% relative to Transolver-Tiny. Regional
analysis shows spatially nonuniform gains, motivating adaptive local gating
rather than a claim of universal regional superiority.

Primary evidence and scripts:

- `drivaernetpp_3d/run_formal_100car.py`
- `drivaernetpp_3d/analyze_formal_100car.py`
- `drivaernetpp_3d/analyze_regions_and_cases.py`
- `drivaernetpp_3d/consolidate_100car_analysis.py`
- `drivaernetpp_3d/outputs/formal_100car/analysis_consolidated_20260906/`

## 4. Three-dimensional aircraft transfer experiment

The experiment uses a stratified set of 100 aircraft drawn from J-20, P180,
Boeing 737, and ERJ families. The split is 80/10/10; both models use the same
4,096 points, 12-NN graph, seed, and optimization protocol.

| Model | Mean test normalized MSE | Parameters | Peak memory (MiB) |
|---|---:|---:|---:|
| Transolver-Tiny | 0.8436 | 195,345 | 103.76 |
| RAPTOR | **0.7713** | 245,525 | 117.77 |

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

## 5. Tests

- `tests/test_model.py`: model shapes, gradients, gates, and equivalence checks.
- `tests/test_modern_baselines.py`: matched modern-baseline checks.
- `drivaernetpp_3d/test_formal_100car.py`: formal automobile runner checks.
- `aircraft_3d/test_aircraft_runner.py`: aircraft data and runner checks.

```powershell
python -m pytest -q tests drivaernetpp_3d/test_formal_100car.py aircraft_3d/test_aircraft_runner.py
```

Formal training additionally requires CUDA and the original datasets.

## 6. Reproducibility boundaries

- Raw datasets are excluded because they are large and governed by their source
  distribution terms.
- Checkpoints, caches, and prediction tensors are excluded; compact JSON/CSV
  results and publication PDFs are retained.
- Full-mesh 3D colors are interpolation-based visualization products. Numerical
  errors are computed on the matched sampled nodes.
- Evidence uses one random seed and controlled 3D subsets; broader multi-seed
  and full-dataset studies remain future work.

