"""接口契约（Protocol）。各模块只依赖抽象，可独立验证。作者：晨星"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    continuous: bool

    def fit(self, states: np.ndarray, actions: np.ndarray) -> Policy: ...

    def predict(self, states: np.ndarray) -> np.ndarray: ...

    def uncertainty(self, states: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Environment(Protocol):
    obs_dim: int
    act_dim: int
    continuous: bool
    action_names: list

    def reset(self, start: Any) -> np.ndarray: ...

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict]: ...

    def expert(self, state: np.ndarray) -> Any: ...

    def sample_starts(self, rng: np.random.Generator, n: int) -> list: ...
