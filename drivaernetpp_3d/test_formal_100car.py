import torch

from run_formal_100car import make_model


def test_all_model_shapes():
    x = torch.randn(1, 32, 3)
    knn = torch.randint(0, 32, (1, 32, 12))
    for name in ("pointnet", "transolver_tiny", "raptor", "no_adapter"):
        y = make_model(name, 12)(x, knn)
        assert y.shape == (1, 32, 1), (name, y.shape)


def test_adapter_contract():
    assert make_model("raptor", 12).blocks[0].local_adapter is not None
    assert make_model("no_adapter", 12).blocks[0].local_adapter is None
