# LinearNO-Tiny Design

## Objective

Add a resource-matched LinearNO-Tiny baseline for AirfRANS Reynolds and AoA,
both with seed 42, without changing the shared data, optimization, or evaluation
protocol and without overwriting existing results.

## Method fidelity

The implementation follows Hu et al. (AAAI 2026): independent learnable
projections produce asymmetric query and key feature maps; the query map is
softmax-normalized over the latent-state dimension and the key map over the
node dimension. The output is computed in canonical linear-attention order,
`phi(Q) @ (psi(K)^T @ V)`. Slice self-attention is intentionally absent. This
preserves the paper's generalization and simplification steps while scaling the
network to hidden dimension 64, depth 4, four heads, and 64 latent states.

## Fair protocol

- Dataset, cache, deterministic 4096-node sampling, normalization, split, and
  targets are shared with the existing RAPTOR modern baselines.
- Reynolds and AoA use seed 42, 100 maximum epochs, patience 15, AdamW,
  learning rate 1e-3, weight decay 1e-6, gradient clipping, and CUDA AMP.
- Results are written below `outputs/linearno_formal`; existing baseline and
  RAPTOR outputs are read-only.
- Formal runs are sequential and may start only after the active AoA pipeline
  has released the GPU.

## Verification

Unit tests cover shape, finite non-zero gradients, asymmetric projections,
normalization axes, absence of dense node attention, and the expected
`N*M` logical score count. A CPU smoke test precedes CUDA formal training.

