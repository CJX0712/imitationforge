"""流水线编排。作者：晨星"""

from __future__ import annotations

import time

import numpy as np

from core.config import Config
from core.seed import new_rng, set_all
from envs.common import evaluate, expert_is_successful, make_demos
from envs.registry import ENV_SEED_OFFSET, make_env
from methods.registry import METHOD_ORDER, REGISTRY


def _eval_starts(env, rng, n):
    out, tries = [], 0
    while len(out) < n and tries < n * 20:
        s = env.sample_starts(rng, 1)[0]
        tries += 1
        if expert_is_successful(env, s):
            out.append(s)
    return out


def run_env(env_name: str, cfg: Config, methods):
    env = make_env(env_name)
    off = ENV_SEED_OFFSET[env_name]
    per_method = {m: [] for m in methods}
    for seed in cfg.seeds:
        set_all(int(seed))
        rng = new_rng(int(seed) * 1000 + off)
        cand = env.sample_starts(rng, max(cfg.n_demo, 12))
        demo_S, demo_A, kept = make_demos(env, cand, env.max_steps)
        if len(kept) < max(5, cfg.n_demo // 2):
            rng2 = new_rng(int(seed) * 2000 + off)
            cand2 = env.sample_starts(rng2, cfg.n_demo * 8)
            demo_S, demo_A, kept = make_demos(env, cand2, env.max_steps)
        demo_starts = kept
        rng3 = new_rng(int(seed) * 3000 + off)
        eval_starts = _eval_starts(env, rng3, cfg.n_eval)
        for m in methods:
            t0 = time.time()
            policy = REGISTRY[m](env, demo_S, demo_A, demo_starts, cfg)
            sr, mr, sd = evaluate(env, policy, eval_starts)
            dt = time.time() - t0
            per_method[m].append(
                (int(seed), float(sr), float(mr), float(sd), float(dt))
            )
    return per_method


def run_benchmark(cfg: Config | None = None, env_names=None, methods=None) -> dict:
    cfg = cfg or Config().validate()
    env_names = env_names or ["gridworld", "pendulum", "cartpole"]
    methods = methods or METHOD_ORDER
    out = {"config": _config_dict(cfg), "envs": {}}
    for env_name in env_names:
        per = run_env(env_name, cfg, methods)
        agg = {}
        for m in methods:
            rows = per[m]
            sr = np.array([r[1] for r in rows])
            mr = np.array([r[2] for r in rows])
            sd = np.array([r[3] for r in rows])
            dt = np.array([r[4] for r in rows])
            agg[m] = {
                "success_rate": {"mean": float(sr.mean()), "std": float(sr.std())},
                "return": {"mean": float(mr.mean()), "std": float(sd.mean())},
                "n_episodes": int(cfg.n_eval),
                "elapsed_sec": float(dt.sum()),
                "per_seed": [
                    {
                        "seed": r[0],
                        "success_rate": r[1],
                        "return": r[2],
                        "return_std": r[3],
                    }
                    for r in rows
                ],
            }
        out["envs"][env_name] = agg
    return out


def _config_dict(cfg: Config) -> dict:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
    return d


def format_table(bench: dict) -> str:
    lines = []
    methods = list(next(iter(bench["envs"].values())).keys())
    for env_name, agg in bench["envs"].items():
        lines.append(f"\n=== {env_name} ===")
        lines.append(f"{'method':<20}{'success_rate':>16}{'return':>12}")
        for m in methods:
            e = agg[m]
            sr = e["success_rate"]
            lines.append(
                f"{m:<20}{sr['mean']:>8.4f}±{sr['std']:>5.3f}{e['return']['mean']:>12.2f}"
            )
        if "bc" in agg:
            bc = agg["bc"]["success_rate"]["mean"]
            fl = agg["imitafuse"]["success_rate"]["mean"]
            lines.append(f"  -> ImiteFuse vs BC: Δsuccess = {fl - bc:+.4f}")
    return "\n".join(lines)


def summarize(bench: dict) -> dict:
    """旗舰相对 BC 的聚合增益（跨环境均值）。"""
    res = {}
    for env_name, agg in bench["envs"].items():
        if "bc" in agg and "imitafuse" in agg:
            bc = agg["bc"]["success_rate"]["mean"]
            fl = agg["imitafuse"]["success_rate"]["mean"]
            res[env_name] = {
                "bc": bc,
                "imitafuse": fl,
                "delta_success": fl - bc,
            }
    return res
