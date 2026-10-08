"""临时调试：验证三环境的专家成功率与收敛。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.seed import set_all
from envs.common import rollout
from envs.pendulum import Pendulum
from envs.registry import make_env


def closest(env, res):
    """最近逼近度（越小越好）：gridworld/pointnav→到目标距离；pendulum/cartpole→到竖直角差。"""
    if isinstance(env, Pendulum) or env.__class__.__name__ == "CartPole":
        out = []
        for s in res.states:
            if env.__class__.__name__ == "CartPole":
                th = np.arctan2(s[2], s[3])
                d = (th + np.pi) % (2 * np.pi) - np.pi
            else:
                th = np.arctan2(s[1], s[0])
                d = (th - np.pi + np.pi) % (2 * np.pi) - np.pi
            out.append(abs(d))
        return out
    return [float(np.linalg.norm(s[:2] - env.target)) for s in res.states]


def expert_success(env, rng, n=200):
    starts = env.sample_starts(rng, n)
    ok = 0
    min_ds = []
    for s in starts:
        res = rollout(env, env.expert, s, env.max_steps)
        if res.success:
            ok += 1
        ds = closest(env, res)
        min_ds.append(min(ds))
    return ok, n, min_ds


def main():
    set_all(20240601)
    for name in ["gridworld", "pendulum", "cartpole"]:
        env = make_env(name)
        rng = np.random.default_rng(12345)
        ok, n, min_ds = expert_success(env, rng, 200)
        arr = np.array(min_ds)
        thr = 0.05 if not isinstance(env, Pendulum) else 0.25
        print(
            f"{name}: expert success {ok}/{n}={ok / n:.3f} | "
            f"min_dist mean={arr.mean():.3f} min={arr.min():.3f} max={arr.max():.3f} "
            f"pct<thr={float((arr < thr).mean()):.3f}"
        )


if __name__ == "__main__":
    main()
