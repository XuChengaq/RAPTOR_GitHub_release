# Implementation plan

1. Import official Transolver++ and MoNo source snapshots into isolated vendor
   directories and record revisions.
2. Identify the smallest faithful model components and their required inputs.
3. Implement matched Tiny wrappers without editing upstream files.
4. Add shared runner plumbing and architecture-specific metadata.
5. Add shape, gradient, connection-audit, and CUDA smoke tests.
6. Run Reynolds seed 42 and produce a machine-readable comparison report.
