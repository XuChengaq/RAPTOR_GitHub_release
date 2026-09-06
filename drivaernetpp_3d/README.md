# RAPTOR 3D vehicle surface-field reconstruction

This directory isolates all DrivAerNet++ follow-up work from the 2D AirfRANS experiments.

## Data boundary

- Source archive: pass the local DrivAerNet++-derived archive with `--archive`;
  the development layout used `../DrivAerNetPP_100/mixed_3d.zip`.
- The source archive is read-only and must not be overwritten.
- The nested archive contains DrivAerNet vehicle meshes and aircraft meshes. Only paths under `DrivAerNet/` are in scope.
- Available targets are single-channel node/element surface fields. They must not be described as full volumetric 3D flow fields.

## Staged workflow

1. Audit archive integrity, fields, shapes, and vehicle-family counts.
2. Build deterministic train/validation/test manifests.
3. Test one-sample loading and mesh indexing.
4. Run a CUDA-only forward/backward smoke test.
5. Run one small RAPTOR 3D experiment before any formal benchmark expansion.
