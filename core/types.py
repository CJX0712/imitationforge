"""共享数据类型。作者：晨星"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RolloutResult:
    """单条轨迹的回放结果。"""

    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    success: bool = False
    return_: float = 0.0
    steps: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class Demo:
    """一条专家演示（状态序列 + 动作序列，等长）。"""

    states: list
    actions: list

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class MethodResult:
    """某个方法在某个环境上的评测结果。"""

    method: str
    env: str
    seed: int
    success_rate: float
    mean_return: float
    std_return: float
    n_episodes: int
    elapsed_sec: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkEntry:
    """benchmark 聚合的一行。"""

    method: str
    env: str
    success_rate_mean: float
    success_rate_std: float
    return_mean: float
    return_std: float
    n_episodes: int
    detail: dict = field(default_factory=dict)
