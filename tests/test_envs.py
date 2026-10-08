"""环境：专家成功率、rollout 含终点态、障碍阻挡。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.seed import set_all
from core.types import RolloutResult
from envs.common import make_demos, rollout
from envs.registry import ENV_NAMES, make_env


def _expert_rate(env, n=120, seed=12345):
    set_all(seed)
    rng = np.random.default_rng(seed)
    starts = env.sample_starts(rng, n)
    ok = sum(rollout(env, env.expert, s, env.max_steps).success for s in starts)
    return ok / n


def test_expert_success_high():
    # 专家必须高成功，否则无演示数据、无天花板
    for name in ENV_NAMES:
        env = make_env(name)
        rate = _expert_rate(env)
        assert rate >= 0.9, f"{name} 专家成功率 {rate:.3f} < 0.9"


def test_rollout_includes_terminal_state():
    # 核心修复点：触发 done 的「终点态」必须写入 states，
    # 否则 env.is_success 会漏算末帧成功 → 系统性低估成功率。
    env = make_env("cartpole")
    set_all(1)
    rng = np.random.default_rng(1)
    start = env.sample_starts(rng, 1)[0]
    res = rollout(env, env.expert, start, env.max_steps)
    # 初始态 + 至少一帧 => 长度 >= 2
    assert len(res.states) >= 2
    # 末态被真实记录（与逐步 step 复跑的终点一致），而非漏掉
    s = env.reset(start)
    last = s.copy()
    for _ in range(env.max_steps):
        a = np.asarray(env.expert(s))
        s, _r, done, _ = env.step(env.clip_action(a))
        last = s.copy()
        if done:
            break
    assert np.allclose(res.states[-1], last), "末态未被写入 states"
    # 若整体成功，则末态本身应落入成功域（用环境自身定义，跨环境通用）
    if res.success:
        assert env.is_success(RolloutResult(states=[res.states[-1]]))


def test_gridworld_blocks_obstacle():
    env = make_env("gridworld")
    set_all(2)
    # 朝墙中心（缝隙外）撞，速度应被清零、不穿墙
    env.reset([0.40, 0.50])  # 墙左侧，y=0.5 在缝隙内 → 应可通行
    # 取墙外一点：y=0.10 在缝隙外
    env.reset([0.44, 0.10])
    obs, r, done, _ = env.step(np.array([env.accel, 0.0]))  # 向右撞墙
    del obs, r, done
    # 撞墙后位置不应进入障碍区域
    assert not env._in_obstacle(env.pos[0], env.pos[1]), "穿墙！"


def test_make_demos_keeps_only_successful():
    env = make_env("cartpole")
    set_all(3)
    rng = np.random.default_rng(3)
    starts = env.sample_starts(rng, 12)
    S, A, kept = make_demos(env, starts, env.max_steps)
    # 保留的起点数 ≤ 原始；演示动作与状态同数
    assert len(S) == len(A)
    assert len(kept) <= len(starts)
    assert len(kept) >= 1
