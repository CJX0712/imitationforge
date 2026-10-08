"""行为克隆 BC：仅在专家演示上监督学习。强基线。作者：晨星"""

from __future__ import annotations

from policy.learners import make_policy


def train_bc(env, demo_S, demo_A, demo_starts, cfg):
    policy = make_policy(cfg, env.continuous)
    policy.fit(demo_S, demo_A)
    return policy
