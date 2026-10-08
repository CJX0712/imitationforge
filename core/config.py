"""配置：ENV_IMITATIONFORGE_* 覆盖 + schema 校验。作者：晨星"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError


@dataclass
class Config:
    # 演示 / 评测规模（默认已右调至 demo ≤60s 预算；CI 可经参数/环境变量放大）
    n_demo: int = 12
    n_eval: int = 12
    # DAGGER / 旗舰
    dagger_rounds: int = 1
    ensemble_size: int = 2
    query_budget_per_round: int = 80  # DAGGER 每轮最多向专家查询的状态数
    uncertainty_gate: bool = True  # 旗舰是否启用不确定性门控查询（False=均匀采样）
    # 交互
    max_steps: int = 200
    # 复现
    seeds: tuple[int, ...] = (7,)
    seed: int = 0
    # 后端
    use_sklearn: bool = True
    policy_kind: str = "mlp"  # mlp | ridge | tree
    # 杂项
    verbose: bool = False

    def validate(self) -> Config:
        if self.n_demo < 1:
            raise ConfigError("n_demo >= 1")
        if self.n_eval < 1:
            raise ConfigError("n_eval >= 1")
        if self.dagger_rounds < 0:
            raise ConfigError("dagger_rounds >= 0")
        if self.ensemble_size < 1:
            raise ConfigError("ensemble_size >= 1")
        if self.max_steps < 1:
            raise ConfigError("max_steps >= 1")
        if self.policy_kind not in ("mlp", "ridge", "tree"):
            raise ConfigError("policy_kind in {mlp,ridge,tree}")
        if len(self.seeds) < 1:
            raise ConfigError("seeds non-empty")
        return self

    @classmethod
    def from_env(cls) -> Config:
        c = cls()
        for f in cls.__dataclass_fields__:
            ev = os.environ.get(f"ENV_IMITATIONFORGE_{f.upper()}")
            if ev is None:
                continue
            cur = getattr(c, f)
            if isinstance(cur, bool):
                setattr(c, f, ev.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(c, f, int(ev))
            elif isinstance(cur, float):
                setattr(c, f, float(ev))
            elif isinstance(cur, tuple):
                setattr(c, f, tuple(int(x) for x in ev.split(",") if x.strip()))
            else:
                setattr(c, f, ev)
        return c.validate()


def default_config() -> Config:
    return Config().validate()
