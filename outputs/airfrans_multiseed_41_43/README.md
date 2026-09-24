# AirfRANS three-seed operator comparison

This directory contains the numerical evidence for the additional training
seeds 41 and 43 on the fixed AirfRANS Reynolds-number and angle-of-attack
(AoA) splits. Seed 42 is the original reference run. The data-selection seed
is 2026; each case uses 4,096 sampled nodes. The additional runs change the
training seed, not the data cache, sampling, normalization, model width, or
early-stopping rule.

The five newly run operator models are RAPTOR, Transolver-Tiny,
Transolver++-Tiny, LinearNO-Tiny, and MoNo-Tiny. The 20 added run directories
live at the paths named by the `result` column of `per_seed_results.csv`.
Each contains `result.json` (full-precision test metrics and run metadata)
and `history.csv` (epoch-level training and validation losses). The
corresponding console traces are in `logs/`. The ten seed-42 `result.json`
files are in their original output directories elsewhere in this repository.
The independently trained seed-42 Reynolds Transolver-Tiny reference has a
recovered evaluation record; its original epoch history is unavailable.

The other five models in the manuscript's three-seed table—MLP, GraphSAGE,
PointNet, Graph U-Net (`gunet`), and Sparse Token—have their original
`result.json` and `history.csv` files under `traditional_baselines/`, with
one directory per model, task, and seed. These were copied from the separate
AirfRANS baseline experiment directory after verifying that their means and
sample standard deviations match the manuscript table. Their `result.json`
files do not contain a precomputed macro metric: compute it as the arithmetic
mean of `rel_l2_ux`, `rel_l2_uy`, `rel_l2_p`, and `rel_l2_nut`.

`per_seed_results.csv` lists the four variable-wise relative L2 errors,
their unweighted macro average, best validation loss, elapsed run time, and
result path for every model/task/seed combination. `three_seed_summary.csv`
and `.json` report the mean and **sample** standard deviation across the
three training seeds. `paired_improvements.csv` computes, within each common
seed, `100 * (baseline_macro - RAPTOR_macro) / baseline_macro`, then reports
the mean and sample standard deviation of those three percentages. The
underlying JSON values have more precision than the manuscript tables.

The run supervisor is `run_airfrans_multiseed_pipeline.py`; the aggregation
source is `scripts/summarize_airfrans_multiseed.py`. Both are provided to
document the protocol, not to imply that rerunning the training is needed to
inspect these results. The runner expects separately downloaded AirfRANS data
and local CUDA infrastructure. Its original checkpoint validation also
expects `best.pt`; model weights are not part of this compact evidence
release. The published CSV/JSON files can be audited directly without GPU
or checkpoint files.

This bundle therefore contains the per-run numerical evidence behind all ten
rows of the manuscript's three-seed macro table. Epoch histories are present
for all traditional baselines, all 20 newly run operator jobs, and nine of
the ten seed-42 operator jobs; the seed-42 Reynolds Transolver-Tiny history
is the known exception noted above. Separate
per-job console logs are available for the 20 newly run operator jobs, not for
the earlier traditional-baseline runs.
