# RAPTOR manuscript

This directory contains the current manuscript and the figure assets required
to compile it.

- `RAPTOR.tex`: LaTeX source.
- `cas-refs.bib`: bibliography database.
- `figs/`: figures referenced by the source.
- `RAPTOR.pdf`: compiled 22-page manuscript.

The source uses Elsevier's `cas-sc` class. With the Elsevier CAS templates and
the standard dependencies installed, compile from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error RAPTOR.tex
```

The accuracy results use fixed splits and one training seed. The separate
node-scaling profile is an implementation-level resource measurement on the
specified RTX 4050 Laptop GPU; its raw aggregate and worker results are stored
under `../outputs/node_scaling_profile/`.
