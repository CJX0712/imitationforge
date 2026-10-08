"""方法：BC/DAGGER/ImiteFuse 训练+评估可运行、产出有限指标；离线兜底路径。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.config import Config
from envs.common import evaluate, make_demos
from envs.registry import make_env
from methods.registry import METHOD_ORDER, REGISTRY


def _data(env, cfg, seed=7, n=20):
    from core.seed import new_rng, set_all

    set_all(seed)
    rng = new_rng(seed * 1000 + 11)
    cand = env.sample_starts(rng, n)
    S, A, kept = make_demos(env, cand, env.max_steps)
    rng3 = new_rng(seed * 3000 + 11)
    # 复用 eval 起点：专家能成功的
    from envs.common import expert_is_successful

    ev = []
    while len(ev) < 8:
        s = env.sample_starts(rng3, 1)[0]
        if expert_is_successful(env, s):
            ev.append(s)
    return S, A, kept, ev


def test_all_methods_run_and_finite():
    env = make_env("cartpole")
    cfg = Config()
    cfg.seeds = (7,)
    S, A, kept, ev = _data(env, cfg)
    for name in ["bc", "nn", "dagger", "imitafuse"]:
        policy = REGISTRY[name](env, S, A, kept, cfg)
        sr, mr, sd = evaluate(env, policy, ev)
        assert 0.0 <= sr <= 1.0, f"{name} 成功率越界 {sr}"
        assert np.isfinite(mr) and np.isfinite(sd)


def test_imitafuse_gate_runs():
    env = make_env("cartpole")
    cfg = Config()
    cfg.seeds = (7,)
    S, A, kept, ev = _data(env, cfg)
    p = REGISTRY["imitafuse"](env, S, A, kept, cfg)
    sr, _, _ = evaluate(env, p, ev)
    assert np.isfinite(sr)


def test_offline_fallback_path():
    """use_sklearn=False 时，方法仍应产出可用策略（Tier-1 纯 numpy 兜底）。"""
    env = make_env("cartpole")
    cfg = Config()
    cfg.use_sklearn = False
    cfg.seeds = (7,)
    S, A, kept, ev = _data(env, cfg)
    p = REGISTRY["bc"](env, S, A, kept, cfg)
    sr, mr, _sd = evaluate(env, p, ev)
    assert 0.0 <= sr <= 1.0 and np.isfinite(mr)


def test_registry_complete():
    for m in METHOD_ORDER:
        assert m in REGISTRY, f"方法 {m} 未在 REGISTRY 注册"
