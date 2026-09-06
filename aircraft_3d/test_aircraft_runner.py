import importlib.util
from pathlib import Path


def load_runner():
    path = Path(__file__).with_name("run_formal_100aircraft.py")
    spec = importlib.util.spec_from_file_location("aircraft_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_definition_and_models():
    runner = load_runner()
    assert runner.FAMILIES == ("J20", "P180", "boeing737", "erj")
    transolver = runner.make_model("transolver_tiny", 12)
    raptor = runner.make_model("raptor", 12)
    assert sum(p.numel() for p in transolver.parameters()) > 0
    assert sum(p.numel() for p in raptor.parameters()) > 0
