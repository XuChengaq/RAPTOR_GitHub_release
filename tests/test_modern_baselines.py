import torch
import pytest

from mono_tiny import MoNoTiny
from linearno_tiny import LinearNOTiny
from transolver_plus_tiny import TransolverPlusTiny


def _check(model, nodes=37):
    x = torch.randn(2, nodes, 7)
    out = model(x)
    assert out.shape == (2, nodes, 4)
    out.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert model.connection_audit(nodes)["uses_dense_node_attention"] is False


def test_transolver_plus_tiny_shape_gradient_and_connections():
    _check(TransolverPlusTiny(dim=32, depth=2, heads=4, slices=8))


def test_mono_tiny_shape_gradient_and_connections():
    _check(MoNoTiny(modes=8, dim=32, layers=1), nodes=24)


def test_linearno_tiny_shape_gradient_and_connections():
    model = LinearNOTiny(dim=32, depth=2, heads=4, states=8)
    _check(model)
    audit = model.connection_audit(37)
    assert audit["linear_attention_scores_per_layer"] == 37 * 8
    assert audit["uses_slice_self_attention"] is False


def test_linearno_uses_asymmetric_maps_and_expected_normalization_axes():
    model = LinearNOTiny(dim=32, depth=1, heads=4, states=8)
    attention = model.blocks[0].attention
    assert attention.to_query.weight.data_ptr() != attention.to_key.weight.data_ptr()
    x = torch.randn(2, 37, 32)
    attention(x)
    assert attention.last_query_map_shape == (2, 4, 37, 8)
    assert attention.last_key_map_shape == (2, 4, 37, 8)
    assert torch.allclose(attention.last_query_sums, torch.ones_like(attention.last_query_sums), atol=1e-5)
    assert torch.allclose(attention.last_key_sums, torch.ones_like(attention.last_key_sums), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is required")
def test_transolver_plus_tiny_amp_gradients_are_finite_and_update_parameters():
    model = TransolverPlusTiny(dim=32, depth=2, heads=4, slices=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x = torch.randn(1, 64, 7, device="cuda")
    target = torch.randn(1, 64, 4, device="cuda")
    before = model.output_projection[-1].weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss = (model(x) - target).square().mean()
    scaler.scale(loss).backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()

    delta = (model.output_projection[-1].weight.detach() - before).abs().max()
    assert float(delta) > 0.0
