"""ImiteFuse 旗舰：DAGGER 数据集聚合 + K 策略集成 + 不确定性门控专家查询。作者：晨星

相比朴素 DAGGER 的三点增益（均有消融对照）：
  1) 集成（ensemble）：多 learner 融合，预测更稳，并给出不确定性估计；
  2) DAGGER 聚合：在「学习器真实分布」补标，修复分布偏移；
  3) 不确定性门控查询（gate）：同等专家查询预算下，只向最不确定的状态查询，
     把专家算力花在信息量最大的地方（消融 imitafuse_nogate 关掉门控→均匀采样）。

gate=True 为发布旗舰；gate=False 用于消融。
作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.seed import new_rng
from envs.common import bootstrap, collect_visited_states
from policy.learners import EnsemblePolicy, make_policy


def train_imitafuse(env, demo_S, demo_A, demo_starts, cfg, gate: bool = True):
    members = [make_policy(cfg, env.continuous) for _ in range(cfg.ensemble_size)]
    for i, m in enumerate(members):
        bs, ba = bootstrap(demo_S, demo_A, cfg.seed * 100 + i)
        m.fit(bs, ba)
    policy = EnsemblePolicy(members)

    S = demo_S.copy()
    A = demo_A.copy()
    for r in range(cfg.dagger_rounds):
        if not demo_starts:
            break
        visited = collect_visited_states(env, policy, demo_starts)
        if len(visited) == 0:
            break
        unc = policy.uncertainty(visited)
        K = min(len(visited), cfg.query_budget_per_round)
        if K == 0:
            break
        if gate:
            order = np.argsort(-unc)[:K]  # 最不确定优先
        else:
            rng = new_rng(int(cfg.seed) * 7 + r)
            order = np.sort(rng.choice(len(visited), size=K, replace=False))
        to_q = visited[order]
        ex = np.array([env.expert(s) for s in to_q])
        S = np.vstack([S, to_q])
        A = np.concatenate([A, ex])
        for i, m in enumerate(members):
            bs, ba = bootstrap(S, A, cfg.seed * 100 + i)
            m.fit(bs, ba)
        policy = EnsemblePolicy(members)
    return policy


def train_ensemble_bc(env, demo_S, demo_A, demo_starts, cfg):
    """消融 imitafuse_nodagger：仅在演示上训练集成（无 DAGGER 聚合）。"""
    members = [make_policy(cfg, env.continuous) for _ in range(cfg.ensemble_size)]
    for i, m in enumerate(members):
        bs, ba = bootstrap(demo_S, demo_A, cfg.seed * 100 + i)
        m.fit(bs, ba)
    return EnsemblePolicy(members)
