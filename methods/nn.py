"""最近邻基线：记忆演示，最近状态查表。强非参数基线。作者：晨星"""

from __future__ import annotations

from policy.learners import NearestNeighborPolicy


def train_nn(env, demo_S, demo_A, demo_starts, cfg):
    return NearestNeighborPolicy(demo_S, demo_A, env.continuous)
