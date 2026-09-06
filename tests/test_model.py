import torch
import pytest
from pathlib import Path

from raptor.model import RAPTOR
from transolver_tiny.model import TransolverTiny


def sample():
    x = torch.randn(2, 37, 7)
    knn = torch.randint(0, 37, (2, 37, 6))
    return x, knn


def copy_transolver_backbone(source, target):
    target.input_projection.load_state_dict(source.input_projection.state_dict())
    target.output_projection.load_state_dict(source.output_projection.state_dict())
    for src, dst in zip(source.blocks, target.blocks):
        dst.norm1.load_state_dict(src.norm1.state_dict())
        dst.physics_attention.load_state_dict(src.physics_attention.state_dict())
        dst.norm2.load_state_dict(src.norm2.state_dict())
        dst.ffn.load_state_dict(src.ffn.state_dict())


def test_zero_adapter_is_exactly_transolver_equivalent():
    torch.manual_seed(7)
    base = TransolverTiny(dim=32, depth=2, heads=4, slices=8).eval()
    model = RAPTOR(dim=32, depth=2, heads=4, slices=8, knn_k=6).eval()
    copy_transolver_backbone(base, model)
    x, knn = sample()
    with torch.no_grad():
        expected = base(x)
        actual = model(x, knn)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.adapter_scales() == [0.0, 0.0]


def test_adapter_gate_and_physics_slices_receive_gradient():
    x, knn = sample()
    model = RAPTOR(dim=32, depth=2, heads=4, slices=8, knn_k=6)
    model(x, knn).square().mean().backward()
    grads = [
        model.blocks[0].local_adapter.alpha.grad,
        model.blocks[0].physics_attention.project_slice.weight.grad,
    ]
    for grad in grads:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_shapes_and_connection_audit():
    x, knn = sample()
    model = RAPTOR(dim=32, depth=2, heads=4, slices=8, knn_k=6)
    assert model(x, knn).shape == (2, 37, 4)
    audit = model.connection_audit(37)
    assert audit["knn_connections_per_layer"] == 37 * 6
    assert audit["point_to_slice_scores_per_layer"] == 37 * 8
    assert audit["uses_dense_node_attention"] is False


def test_preserved_formal_checkpoint_loads_into_raptor():
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "formal"
        / "reynolds-raptor-s42"
        / "best.pt"
    )
    if not checkpoint.exists():
        pytest.skip("formal checkpoint is intentionally excluded from release archives")
    model = RAPTOR(dim=64, depth=4, heads=4, slices=64, knn_k=12)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_adapter_ablation_variants_have_valid_shapes_and_scales():
    x, knn = sample()
    disabled = RAPTOR(dim=32, depth=1, heads=4, slices=8, knn_k=6,
                      adapter_enabled=False)
    nonzero = RAPTOR(dim=32, depth=1, heads=4, slices=8, knn_k=6,
                     adapter_alpha_init=0.1)
    assert disabled(x, knn).shape == (2, 37, 4)
    assert disabled.adapter_scales() == []
    assert nonzero(x, knn).shape == (2, 37, 4)
    assert nonzero.adapter_scales()[0] != 0.0
