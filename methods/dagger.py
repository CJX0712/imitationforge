"""DAGGER（Ross et al. 2011, ICML）：数据集聚合。

核心：用当前学习器回放收集其「真实分布」下的状态，再向专家查询这些状态的动作，
并入训练集重新拟合。结构性缓解行为克隆的分布偏移问题。

ensemble=True 时旗舰的集成变体（DAGGER + K 策略集成）。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from envs.common import bootstrap, collect_visited_states
from policy.learners import EnsemblePolicy, make_policy


def train_dagger(env, demo_S, demo_A, demo_starts, cfg, ensemble: bool = False):
    if ensemble:
        members = [make_policy(cfg, env.continuous) for _ in range(cfg.ensemble_size)]
        for i, m in enumerate(members):
            bs, ba = bootstrap(demo_S, demo_A, cfg.seed * 100 + i)
            m.fit(bs, ba)
        policy = EnsemblePolicy(members)
    else:
        policy = make_policy(cfg, env.continuous)
        policy.fit(demo_S, demo_A)

    S = demo_S.copy()
    A = demo_A.copy()
    for _ in range(cfg.dagger_rounds):
        if not demo_starts:
            break
        visited = collect_visited_states(env, policy, demo_starts)
        if len(visited) == 0:
            break
        ex = np.array([env.expert(s) for s in visited])
        S = np.vstack([S, visited])
        A = np.concatenate([A, ex])
        if ensemble:
            for i, m in enumerate(members):
                bs, ba = bootstrap(S, A, cfg.seed * 100 + i)
                m.fit(bs, ba)
            policy = EnsemblePolicy(members)
        else:
            policy.fit(S, A)
    return policy
