"""确定性二次校验：同种子两次 benchmark 核心指标逐位一致。作者：晨星"""

from __future__ import annotations

import json

from core.config import Config
from pipeline.pipeline import run_benchmark


def _core(bench):
    out = {}
    for env_name, agg in bench["envs"].items():
        out[env_name] = {
            m: {
                "sr_mean": agg[m]["success_rate"]["mean"],
                "sr_std": agg[m]["success_rate"]["std"],
                "ret_mean": agg[m]["return"]["mean"],
            }
            for m in agg
        }
    return out


def test_deterministic_two_runs():
    cfg = Config()
    cfg.seeds = (7,)
    cfg.n_eval = 8
    b1 = run_benchmark(cfg, env_names=["cartpole"], methods=["bc", "imitafuse"])
    b2 = run_benchmark(cfg, env_names=["cartpole"], methods=["bc", "imitafuse"])
    assert json.dumps(_core(b1), sort_keys=True) == json.dumps(
        _core(b2), sort_keys=True
    )
