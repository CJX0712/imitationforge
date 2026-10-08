"""环境与策略交互的公共原语（rollout / 数据收集 / 评估）。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.types import RolloutResult


def _ctrl(policy):
    """把策略的预测封装成单状态控制器：连续→动作向量，离散→整数。"""

    def f(state):
        p = np.asarray(policy.predict(state.reshape(1, -1)))
        if policy.continuous:
            return p.ravel().astype(float)
        return int(p.ravel()[0])

    return f


def rollout(env, controller, start, max_steps):
    """回放单条轨迹。controller(state) -> action。

    注意：初始态与「触发 done 的终点态」都会被写入 states，
    否则 is_success 会漏掉刚好抵达成功半径的那一帧，系统性低估成功率。
    """
    state = env.reset(start)
    states, actions, rewards = [state.copy()], [], []
    done = False
    for _ in range(max_steps):
        a = controller(state)
        actions.append(a)
        state, r, done, _info = env.step(env.clip_action(a))
        rewards.append(r)
        states.append(state.copy())
        if done:
            break
    res = RolloutResult(
        states=states, actions=actions, rewards=rewards, steps=len(states)
    )
    res.return_ = float(sum(rewards))
    res.success = env.is_success(res)
    return res


def expert_is_successful(env, start):
    return rollout(env, env.expert, start, env.max_steps).success


def make_demos(env, starts, max_steps):
    """跑专家生成演示；只保留专家成功的轨迹。返回 (S, A, kept_starts)。"""
    S, A, kept = [], [], []
    for start in starts:
        res = rollout(env, env.expert, start, max_steps)
        if res.success:
            for s, a in zip(res.states, res.actions):
                S.append(s)
                A.append(a)
            kept.append(start)
    return np.asarray(S, float), np.asarray(A), kept


def evaluate(env, policy, starts):
    """在独立评测起点上评估策略。返回 (成功率, 平均回报, 回报std)。"""
    ctrl = _ctrl(policy)
    succ, rets = [], []
    for start in starts:
        res = rollout(env, ctrl, start, env.max_steps)
        succ.append(1.0 if res.success else 0.0)
        rets.append(res.return_)
    return (
        float(np.mean(succ)),
        float(np.mean(rets)),
        float(np.std(rets)) if len(rets) > 1 else 0.0,
    )


def collect_visited_states(env, policy, starts, max_states=600):
    """DAGGER：用当前策略回放，收集其真实分布下访问到的状态。"""
    states = []
    ctrl = _ctrl(policy)
    for start in starts:
        s = env.reset(start)
        for _ in range(env.max_steps):
            a = ctrl(s)
            states.append(s.copy())
            s, _r, done, _ = env.step(env.clip_action(a))
            if done or len(states) >= max_states:
                break
        if len(states) >= max_states:
            break
    return np.asarray(states, float)


def bootstrap(S, A, seed, size=None):
    """确定性 bootstrap 重采样（供集成成员独立拟合）。"""
    rng = np.random.default_rng(seed)
    n = len(S)
    if size is None:
        size = n
    idx = rng.integers(0, n, size)
    return S[idx], A[idx]
