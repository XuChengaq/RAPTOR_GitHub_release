# RAPTOR code structure

## Core operator and shared AirfRANS utilities

| Path | Responsibility |
|---|---|
| `raptor/model.py` | RAPTOR operator blocks: Physics Attention, 12-NN residual local adapter, and zero-initialized bounded gate |
| `raptor/run.py` | CUDA-required AirfRANS training, AMP, gradient clipping, early stopping, evaluation, and result serialization |
| `data.py` | official AirfRANS loading, deterministic simulation selection, 4,096-node sampling, normalization, kNN, and token groups |
| `baseline_common.py` | shared training and evaluation utilities for matched modern baselines |
| `configs/formal.yaml` | Reynolds-number seed-42 RAPTOR configuration |
| `configs/aoa_raptor_formal.yaml` | angle-of-attack seed-42 RAPTOR configuration |

## AirfRANS baseline and ablation experiments

| Path | Responsibility |
|---|---|
| `transolver_tiny/model.py` | matched Physics-Attention backbone used as the principal global-interaction baseline |
| `transolver_tiny/run.py` | matched Transolver-Tiny training and evaluation utilities |
| `transolver_plus_tiny/model.py` | compact Transolver++ adaptation under the common protocol |
| `linearno_tiny/model.py` | compact LinearNO adaptation under the common protocol |
| `mono_tiny/model.py` | compact MoNo adaptation using the recorded upstream implementation |
| `run_aoa_pipeline.py` | restart-safe serial dispatch of the formal AoA model comparison |
| `run_modern_baselines.py` | matched Transolver++-Tiny and MoNo-Tiny experiment runner |
| `run_linearno_after_aoa.ps1` | CUDA-safe sequential LinearNO-Tiny launcher for Reynolds and AoA |
| `run_core_ablation_pipeline.py` | four formal ablations: Reynolds/AoA × no-adapter/nonzero-initialization |
| `configs/` | formal, smoke, modern-baseline, and core-ablation YAML configurations |
| `scripts/summarize_linearno.py` | aggregation of matched seed-42 AirfRANS results and resource measurements |
| `scripts/plot_airfrans_*.py` | AirfRANS four-variable, pressure, truth--prediction, and error visualizations |

## Three-dimensional automobile experiments

| Path | Responsibility |
|---|---|
| `drivaernetpp_3d/audit_dataset.py` | nested-archive integrity, field, shape, and sample-count audit |
| `drivaernetpp_3d/run_small_3d.py` | deterministic 3D cache construction and small CUDA trial |
| `drivaernetpp_3d/run_formal_100car.py` | restart-safe 100-car PointNet, Transolver-Tiny, RAPTOR, and no-adapter benchmark |
| `drivaernetpp_3d/analyze_formal_100car.py` | per-vehicle error statistics and paired comparisons |
| `drivaernetpp_3d/analyze_regions_and_cases.py` | front, roof, side, lower/wheel-zone, and rear regional analysis |
| `drivaernetpp_3d/consolidate_100car_analysis.py` | consolidated CSV, JSON, LaTeX tables, and statistical figure generation |
| `drivaernetpp_3d/plot_car_2x3_field_error_improvement.py` | ground-truth, prediction, error, and local-improvement 2×3 visualization |
| `drivaernetpp_3d/plot_*paper*.py` | full-mesh publication visualizations interpolated from matched sampled nodes |
| `drivaernetpp_3d/test_formal_100car.py` | automobile runner and model-construction tests |

## Three-dimensional aircraft experiments

| Path | Responsibility |
|---|---|
| `aircraft_3d/run_formal_100aircraft.py` | stratified four-family 100-aircraft Transolver-Tiny/RAPTOR benchmark |
| `aircraft_3d/analyze_aircraft_results.py` | per-aircraft paired statistics, family summaries, and base visualizations |
| `aircraft_3d/make_four_aircraft_high_quality.py` | one representative full-surface comparison for each aircraft family |
| `aircraft_3d/make_boeing737_high_quality.py` | reusable six-panel high-quality single-aircraft visualization |
| `aircraft_3d/make_aircraft_best_worst_2x6.py` | automatic best/worst paired-case selection and combined 2×6 figure |
| `aircraft_3d/test_aircraft_runner.py` | aircraft sampling and formal-runner tests |

## Tests, documentation, and publication assets

| Path | Responsibility |
|---|---|
| `tests/test_model.py` | output shape, gradient flow, connection audit, zero-gate equivalence, and optional checkpoint compatibility |
| `tests/test_modern_baselines.py` | matched modern-baseline construction and forward/backward tests |
| `docs/EXPERIMENTS.md` | protocols, quantitative results, evidence paths, and reproducibility limitations |
| `docs/RAPTOR_experiment_code_structure.*` | visual overview of the code hierarchy and end-to-end experimental workflow |
| `figs/Fig1.png` | RAPTOR method architecture used in the paper |
| `figs/RAPTOR_framework_editable.pptx` | editable source of the method architecture |
| `paper_assets/cas-refs.bib` | paper bibliography with `ref01`-style citation keys |
| `paper_assets/RAPTOR_complete_references.docx` | formatted complete reference list |
| `vendor/SOURCES.md` | exact upstream revisions and licenses for adapted third-party baselines |

## Result hierarchy

Per-run `result.json` files are the numerical source of truth. The associated
`history.csv` files record epoch-level training and validation trajectories.
Aggregated CSV, JSON, and LaTeX files under `outputs/`,
`drivaernetpp_3d/outputs/formal_100car/analysis*`, and
`aircraft_3d/outputs/formal_100aircraft/analysis/` are derived paper artifacts.
PDF files in these directories are publication visualizations.

Raw AirfRANS and DrivAerNet++-derived data, deterministic caches, checkpoints,
prediction tensors, logs, and smoke-test outputs are not included in the GitHub
release. Full-mesh 3D colors are visualization-only interpolations from the
matched 4,096 sampled nodes; all reported numerical errors are evaluated on the
original sampled nodes.

