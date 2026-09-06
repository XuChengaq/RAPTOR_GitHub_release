# Modern AirfRANS baseline screening design

## Objective

Evaluate RAPTOR against two recent general-geometry PDE solvers under the
existing matched AirfRANS protocol. The first screening uses only the Reynolds
task and seed 42. It must not overwrite the preserved RAPTOR or
Transolver-Tiny evidence.

## Baselines

- `transolver_plus_tiny`: a resource-matched adaptation of the official 2025
  Transolver++ architecture.
- `mono_tiny`: a resource-matched adaptation of the official 2026 MoNo
  architecture and AirfRANS data path.

Official source snapshots are kept under `vendor/` for provenance. Project
adapters are kept separate from those snapshots so upstream code remains
unchanged.

## Fair protocol

All models use the same cached AirfRANS split, 4096 sampled nodes, seven input
channels, four output fields, training-only normalization, Reynolds task,
seed 42, 100 epochs, patience 15, AdamW, AMP, gradient clipping, and CUDA-only
execution. Tiny variants use hidden width 64 and four blocks where the official
architecture permits. Deviations required by an architecture are recorded in
each `result.json`.

## Execution gates

1. Verify source provenance and licenses.
2. Test output shape, finite gradients, absence of dense 4096-by-4096 node
   attention, and parameter count.
3. Run a one-simulation CUDA smoke test for each baseline.
4. Run formal Reynolds seed-42 training only after smoke tests pass.
5. Compare four relative L2 errors, macro error, surface pressure MSE,
   parameters, peak allocated memory, inference latency, and training time.

The screening does not claim stable superiority from one seed. Only promising
baselines are expanded later to seeds 41/42/43 and the Scarce task.

## Failure and recovery

Each runner skips an existing valid `result.json`, writes to a unique output
directory, forbids CPU fallback, and saves the best checkpoint before final
evaluation. An incompatible official dependency or CUDA out-of-memory error is
reported rather than silently changing the protocol.
