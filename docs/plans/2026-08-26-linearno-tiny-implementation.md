# LinearNO-Tiny Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Implement and run a faithful Tiny LinearNO baseline on matched Reynolds and AoA seed-42 AirfRANS tasks.

**Architecture:** Add an isolated LinearNO-Tiny package whose core computes asymmetric canonical linear attention without slice self-attention. Reuse the existing matched training runner, cache builder, and metrics so only the model mechanism changes.

**Tech Stack:** Python 3.12, PyTorch 2.5, CUDA AMP, pytest, YAML.

---

### Task 1: Specify the LinearNO core with failing tests

**Files:**
- Modify: `tests/test_modern_baselines.py`
- Create: `linearno_tiny/model.py`

1. Add tests for output shape, finite gradients, asymmetric query/key weights,
   normalization sums, and connection audit.
2. Run `pytest -q tests/test_modern_baselines.py -k linearno` and verify failure
   because `linearno_tiny` does not yet exist.
3. Implement only the model code needed by the tests.
4. Re-run the focused tests and expect PASS.

### Task 2: Integrate the model into the matched runner

**Files:**
- Create: `linearno_tiny/__init__.py`
- Modify: `run_modern_baselines.py`

1. Add `linearno_tiny` to imports, model choices, construction, and metadata.
2. Verify CLI help lists `linearno_tiny`.
3. Run the complete modern-baseline unit-test file and expect PASS.

### Task 3: Add isolated formal configuration

**Files:**
- Create: `configs/linearno_formal.yaml`

1. Copy the matched AirfRANS protocol exactly.
2. Set tasks to Reynolds and AoA, seeds to 42, model to LinearNO-Tiny, and
   output directory to `outputs/linearno_formal`.
3. Parse the YAML and assert both tasks and the isolated output directory.

### Task 4: Smoke-test and formally run without GPU overlap

**Files:**
- Create: `run_linearno_pipeline.py`

1. Add a guard that rejects any concurrent project training process.
2. Run a CPU shape/gradient smoke test.
3. After the active AoA pipeline exits, verify CUDA availability and launch
   Reynolds followed by AoA through the matched runner.
4. Verify two CUDA `result.json` files and no failure logs.

### Task 5: Compare and report

**Files:**
- Create: `scripts/summarize_linearno.py`

1. Read LinearNO-Tiny plus existing RAPTOR, Transolver-Tiny,
   Transolver++-Tiny, and MoNo-Tiny results.
2. Export comparable four-variable errors, macro error, parameters, training
   time, peak memory, and inference time without inventing missing values.
3. Run relevant tests and report all result paths.

