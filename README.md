# RAPTOR

**Resource-Aware Physics-Token Operator with Residual Local Adaptation for
Aerodynamic Field Reconstruction**

RAPTOR combines a Physics-Attention global backbone with a sparse local
`K`-nearest-neighbour residual adapter. Each adapter is controlled by a bounded
gate, `tanh(alpha)`, initialized at zero. The initialization therefore preserves
the unadapted backbone function, while training can progressively activate
node-level geometric correction.

This repository contains the implementation, matched baselines, configurations,
tests, analysis scripts, compact numerical evidence, and paper figures for:

- AirfRANS Reynolds-number and angle-of-attack extrapolation;
- core ablations of the local adapter and gate initialization;
- 100-vehicle three-dimensional surface-field reconstruction;
- 100-aircraft transfer across J-20, P180, Boeing 737, and ERJ families.

All formal experiments use seed 42, 4,096 sampled nodes, latent width 64, four
operator blocks, 64 physics tokens, and 12 nearest neighbours where the local
adapter is enabled. Training was performed with CUDA mixed precision on an
NVIDIA RTX 4050 Laptop GPU with approximately 6 GB of memory.

## Main results

| Experiment | Reference | RAPTOR | Relative change |
|---|---:|---:|---:|
| AirfRANS Reynolds, macro relative L2 | Transolver-Tiny: 0.2280 | **0.2219** | -2.65% error |
| AirfRANS AoA, macro relative L2 | Transolver-Tiny: 0.3138 | **0.2969** | -5.37% error |
| 100-car surface field, normalized MSE | Transolver-Tiny: 0.1455 | **0.1357** | -6.77% error |
| 100-aircraft surface field, normalized MSE | Transolver-Tiny: 0.8436 | **0.7713** | -8.57% error |

The aircraft gain is not uniform: RAPTOR wins on 6 of 10 held-out aircraft and
improves the family mean for J-20, Boeing 737, and ERJ, while degrading P180.
See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for protocols, ablations,
resource measurements, evidence locations, and limitations.
For a path-by-path implementation index, see
[CODE_STRUCTURE.md](CODE_STRUCTURE.md).

## Experimental code structure

![RAPTOR experimental code structure and workflow](docs/RAPTOR_experiment_code_structure.png)

The diagram combines the repository structure with the end-to-end data,
training, evaluation, statistical-analysis, and visualization workflow. A PDF
version and its generation script are provided in `docs/`.

## Repository layout

```text
raptor/                    RAPTOR model and AirfRANS runner
transolver_tiny/           matched Physics-Attention baseline
transolver_plus_tiny/      matched Transolver++-Tiny adaptation
linearno_tiny/             matched LinearNO-Tiny adaptation
mono_tiny/                 matched MoNo-Tiny adaptation
configs/                   smoke, formal, and ablation configurations
tests/                     shape, gradient, equivalence, and runner tests
scripts/                   AirfRANS diagnostics, summaries, and figures
drivaernetpp_3d/           100-car training and analysis
aircraft_3d/               100-aircraft training and analysis
outputs/                   compact AirfRANS numerical evidence
figs/                      publication figures and editable framework source
paper_assets/              bibliography and reference-document utilities
docs/                      experiment guide and implementation notes
vendor/SOURCES.md          exact upstream baseline provenance
```

Large raw datasets, generated caches, checkpoints, prediction tensors, and logs
are intentionally excluded from version control.

## Data preparation

The datasets are not redistributed by this repository.

1. Download AirfRANS from its official source and update `data_root` in the
   YAML files if it is not at
   `../AirfRANS_engineering_validation/data/AirfRANS/Dataset`.
2. Obtain the DrivAerNet++-derived archive used by the 3D experiments and pass
   its path through `--archive`. The 3D target is a single-channel surface
   field, not a complete volumetric flow field.
3. Adapted third-party baseline snapshots are included under `vendor/` together
   with their original license files. Their exact upstream revisions are listed
   in `vendor/SOURCES.md`.

## Environment

Python 3.10 or later and a CUDA-enabled PyTorch installation are required.
Install PyTorch for the local CUDA version, then run:

```powershell
python -m pip install -r requirements.txt
```

Formal runners raise an error when CUDA is unavailable and do not silently fall
back to CPU.

## Reproduction

Run commands from the repository root.

```powershell
# Tests
python -m pytest -q tests drivaernetpp_3d/test_formal_100car.py aircraft_3d/test_aircraft_runner.py

# AirfRANS RAPTOR
python -m raptor.run --config configs/formal.yaml
python -m raptor.run --config configs/aoa_raptor_formal.yaml

# Core AirfRANS ablations
python -s run_core_ablation_pipeline.py

# 100-car benchmark
python -s drivaernetpp_3d/run_formal_100car.py --archive PATH_TO_ARCHIVE \
  --output drivaernetpp_3d/outputs/formal_100car --points 4096 --knn 12 \
  --epochs 100 --patience 15 --seed 42

# 100-aircraft transfer benchmark
python -s aircraft_3d/run_formal_100aircraft.py --archive PATH_TO_ARCHIVE \
  --output aircraft_3d/outputs/formal_100aircraft --points 4096 --knn 12 \
  --epochs 100 --patience 15 --seed 42 --models transolver_tiny raptor
```

Completed `result.json` files act as completion markers, allowing serial
pipelines to resume without overwriting finished results.

## Analysis and figures

```powershell
python drivaernetpp_3d/consolidate_100car_analysis.py
python aircraft_3d/analyze_aircraft_results.py
python aircraft_3d/make_four_aircraft_high_quality.py
python aircraft_3d/make_aircraft_best_worst_2x6.py
```

Full-mesh 3D colors are visualization-only interpolations from matched 4,096
sampled nodes. Reported MSE values are evaluated on the sampled nodes.

## Scope and limitations

- AirfRANS and 3D tasks use different targets; their errors are not directly
  comparable across datasets.
- The 3D studies use controlled subsets of 100 objects with 80/10/10 splits.
- Current evidence uses one training seed and does not establish universal
  per-sample improvement or hardware-independent speed superiority.
- Lower asymptotic connectivity does not guarantee lower wall-clock time or
  peak memory on every platform.

## Third-party code

Baseline names and upstream revisions are documented in
[vendor/SOURCES.md](vendor/SOURCES.md). Tiny models are matched-protocol
adaptations, not reproductions of the authors' full published configurations.
