# Parameter-matched control and neighbourhood-size study

This directory contains the fixed-split, seed-42 AirfRANS runs used to compare
RAPTOR with a parameter-matched node-wise residual adapter and to vary
`K` over 4, 8, 12, and 24. The node-wise control retains the same gate,
LayerNorm, MLP, and parameter count, but replaces the neighbour mean with
the centre-node feature. Its implementation is in `raptor/model.py`, with
`adapter_neighbor_mode: self` in
`configs/capacity_control_nodewise.yaml`.

Each of the eight newly trained model/task combinations under `nodewise/`,
`knn_k4/`, `knn_k8/`, and `knn_k24/` includes a full-precision `result.json`
and epoch-level `history.csv`. The original `K=12` seed-42 results were reused
from `outputs/formal/reynolds-raptor-s42/` and
`outputs/aoa_formal/raptor/aoa-raptor-s42/`; they were not retrained for this
study. `summary.csv` and `.json` combine the ten rows. The original training
console trace is `pipeline_stdout_run2.log`. The configuration files are
under `configs/knn_sensitivity_k{4,8,24}.yaml`.

`accuracy_cost_profile.csv` and `.json` contain the separate inference
latency and peak allocated GPU-memory measurements behind the paper's
accuracy--cost figure (`paper/figs/accuracy_cost_tradeoff.pdf`). Each latency
is based on 100 measured mixed-precision single-sample forward passes after
20 warm-up passes, with precomputed neighbour indices. Those repetitions
characterize timing variability, **not** training-seed accuracy variability.
The profile source is `scripts/profile_accuracy_cost_tradeoff.py`.

`train_seconds` in each `result.json` is a wall-clock duration of that run,
including final evaluation. It is not a controlled per-step architecture
benchmark. The study uses one training seed, so accuracy comparisons remain
descriptive. Source code and configs are included for inspection and optional
retraining; the JSON/CSV results can be checked without running a training
job. The script `scripts/summarize_capacity_knn_sensitivity.py` also launches
profiling after writing its summary, so it should **not** be used as a
read-only verification command.

Dataset caches, `best.pt` checkpoints, and per-sample prediction tensors are
not included in this compact Git release. Consequently, the published files
allow recalculation of the reported aggregate metrics and inspection of the
training curves, but do not permit independent inference from the original
weights without separately obtaining them.
