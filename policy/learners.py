"""策略 learner 实现。作者：晨星

- SklearnPolicy：Tier-0 复用世界顶级 scikit-learn（MLP/Ridge/Tree）。
- NumpyPolicy：Tier-1 离线兜底（线性回归 / 最近质心 1-NN），零第三方运行时依赖。
- EnsemblePolicy：K 成员集成，predict=均值/投票，uncertainty=方差/分歧。
- NearestNeighborPolicy：强非参数基线（记忆演示，最近邻查表）。
"""

from __future__ import annotations

import numpy as np

from core.errors import PolicyError


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


class BasePolicy:
    continuous: bool = True

    def fit(self, states: np.ndarray, actions: np.ndarray) -> BasePolicy:
        raise NotImplementedError

    def predict(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def uncertainty(self, states: np.ndarray) -> np.ndarray:
        return np.zeros(len(states))


class SklearnPolicy(BasePolicy):
    def __init__(self, kind: str, continuous: bool):
        self.kind = kind
        self.continuous = continuous
        self._model = None

    def _build(self):
        from sklearn.linear_model import Ridge, RidgeClassifier
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if self.continuous:
            if self.kind == "mlp":
                return MLPRegressor(
                    hidden_layer_sizes=(24, 24),
                    max_iter=2000,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=0,
                )
            if self.kind == "ridge":
                return Ridge(alpha=1.0)
            if self.kind == "tree":
                return DecisionTreeRegressor(max_depth=8)
        else:
            if self.kind == "mlp":
                return MLPClassifier(
                    hidden_layer_sizes=(24, 24),
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=0,
                )
            if self.kind == "ridge":
                return RidgeClassifier(alpha=1.0)
            if self.kind == "tree":
                return DecisionTreeClassifier(max_depth=8)
        raise PolicyError(f"unknown policy_kind={self.kind}")

    def fit(self, states, actions):
        states = np.asarray(states, float)
        self._model = self._build()
        if self.continuous:
            y = np.asarray(actions, float)
            # sklearn MLPRegressor 单输出期望 1D y；多输出保持 (n, k)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            self._model.fit(states, y)
        else:
            self._model.fit(states, np.asarray(actions, int).ravel())
        return self

    def predict(self, states):
        states = np.asarray(states, float)
        p = self._model.predict(states)
        return np.asarray(p, float) if self.continuous else np.asarray(p, int)


class NumpyPolicy(BasePolicy):
    """Tier-1 离线兜底：连续=最小二乘线性回归；离散=最近质心(1-NN 查表)。"""

    def __init__(self, continuous: bool):
        self.continuous = continuous
        self._S = None
        self._A = None
        self._W = None

    def fit(self, states, actions):
        states = np.asarray(states, float)
        actions = np.asarray(actions)
        if self.continuous:
            actions = actions.astype(float)
            X = np.hstack([states, np.ones((len(states), 1))])
            self._W, *_ = np.linalg.lstsq(X, actions, rcond=None)
        else:
            self._S = states
            self._A = actions.astype(int).ravel()
        return self

    def predict(self, states):
        states = np.asarray(states, float)
        if self.continuous:
            X = np.hstack([states, np.ones((len(states), 1))])
            return X @ self._W
        d = np.linalg.norm(self._S[None] - states[:, None], axis=2)
        return self._A[np.argmin(d, axis=1)]

    def uncertainty(self, states):
        return np.zeros(len(states))


class EnsemblePolicy(BasePolicy):
    def __init__(self, members):
        self.members = list(members)
        self.continuous = self.members[0].continuous

    def fit(self, states, actions):
        # 成员各自已在外部 bootstrap 拟合；此处仅重新包装。
        return self

    def predict(self, states):
        preds = np.stack([np.asarray(m.predict(states)) for m in self.members], axis=0)
        if self.continuous:
            return preds.mean(axis=0)
        preds = preds.astype(int)
        out = np.empty(preds.shape[1], dtype=int)
        for j in range(preds.shape[1]):
            vals, counts = np.unique(preds[:, j], return_counts=True)
            out[j] = vals[int(np.argmax(counts))]
        return out

    def uncertainty(self, states):
        preds = np.stack([np.asarray(m.predict(states)) for m in self.members], axis=0)
        if self.continuous:
            return preds.var(axis=0).mean(axis=-1)
        preds = preds.astype(int)
        frac = np.array(
            [
                float((preds[:, j] != np.bincount(preds[:, j]).argmax()).mean())
                for j in range(preds.shape[1])
            ]
        )
        return frac


class NearestNeighborPolicy(BasePolicy):
    """强非参数基线：记忆演示 (s,a)，预测=最近演示状态对应的动作。"""

    def __init__(self, states, actions, continuous: bool):
        self.continuous = continuous
        self._S = np.asarray(states, float)
        self._A = np.asarray(actions)
        if not continuous:
            self._A = self._A.astype(int)

    def fit(self, states=None, actions=None):
        return self

    def predict(self, states):
        states = np.asarray(states, float)
        d = np.linalg.norm(self._S[None] - states[:, None], axis=2)
        return self._A[np.argmin(d, axis=1)]

    def uncertainty(self, states):
        states = np.asarray(states, float)
        d = np.linalg.norm(self._S[None] - states[:, None], axis=2)
        return d.min(axis=1)


def make_policy(cfg, continuous: bool) -> BasePolicy:
    """按配置选择 learner；sklearn 不可用时自动降级纯 numpy。"""
    if cfg.use_sklearn and available_sklearn():
        return SklearnPolicy(cfg.policy_kind, continuous)
    return NumpyPolicy(continuous)
