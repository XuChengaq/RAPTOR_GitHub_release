"""Run matched Transolver++-Tiny and MoNo-Tiny AirfRANS experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from baseline_common import train_and_evaluate
from data import build_cache
from linearno_tiny import LinearNOTiny
from mono_tiny import MoNoTiny
from transolver_plus_tiny import TransolverPlusTiny

PROJECT = Path(__file__).resolve().parent


def build_model(name, cfg):
    if name == "linearno_tiny":
        return LinearNOTiny(dim=cfg["hidden_dim"], depth=cfg["depth"],
                            heads=cfg["heads"], states=cfg["linearno_states"])
    if name == "transolver_plus_tiny":
        return TransolverPlusTiny(dim=cfg["hidden_dim"], depth=cfg["depth"],
                                  heads=cfg["heads"], slices=cfg["slice_count"])
    if name == "mono_tiny":
        return MoNoTiny(modes=cfg["mono_modes"], dim=cfg["hidden_dim"],
                        layers=cfg["mono_layers"])
    raise ValueError(name)


def metadata(name, cfg):
    common = {"hidden_dim": cfg["hidden_dim"], "tiny_adaptation": True}
    if name == "linearno_tiny":
        return {**common, "depth": cfg["depth"], "heads": cfg["heads"],
                "linearno_states": cfg["linearno_states"],
                "asymmetric_query_key_maps": True,
                "slice_self_attention": False,
                "method_source": "Hu et al., AAAI 2026",
                "doi": "10.1609/aaai.v40i1.37003"}
    if name == "transolver_plus_tiny":
        return {**common, "depth": cfg["depth"], "heads": cfg["heads"],
                "slice_count": cfg["slice_count"],
                "amp_safe_fp32_eidetic_slicing": True,
                "upstream_commit": "d5a23bc734a0ebac56384cf72049a26af9673452"}
    return {**common, "mono_modes": cfg["mono_modes"], "mono_layers": cfg["mono_layers"],
            "amp_core": False, "amp_disabled_reason": "official mixed-feature scatter requires FP32",
            "upstream_commit": "60d086a45d4929769796fb3b2dfacc6680528f3b"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/modern_smoke.yaml")
    parser.add_argument("--model", choices=["transolver_plus_tiny", "mono_tiny", "linearno_tiny"], action="append")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load((PROJECT / args.config).read_text(encoding="utf-8"))
    names = args.model or cfg["models"]
    tasks = [args.task] if args.task else cfg["tasks"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    results = []
    for task in tasks:
        cache = build_cache(PROJECT / cfg["data_root"], task,
                            cfg["train_simulations"], cfg["test_simulations"],
                            cfg["points_per_simulation"], cfg["surface_fraction"],
                            cfg["token_count"], cfg["knn_k"],
                            PROJECT / cfg["cache_dir"], seed=cfg["data_seed"],
                            val_fraction=cfg["validation_fraction"])
        data = np.load(cache)
        for name in names:
            for seed in seeds:
                output = PROJECT / cfg["output_dir"] / name / f"{task}-{name}-s{seed}"
                output.mkdir(parents=True, exist_ok=True)
                result = output / "result.json"
                if result.exists():
                    print("SKIP", output.name, flush=True)
                    results.append(json.loads(result.read_text(encoding="utf-8")))
                    continue
                factory = lambda name=name: build_model(name, cfg)
                results.append(train_and_evaluate(cfg, task, seed, data, output,
                                                  name, factory, metadata(name, cfg)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
