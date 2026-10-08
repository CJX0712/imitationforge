"""策略 learner：sklearn 路径、纯 numpy 离线兜底、集成不确定性。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.config import Config
from policy.learners import (
    EnsemblePolicy,
    available_sklearn,
    make_policy,
)


def _toy(continuous, n=60, dim=6, adim=2):
    rng = np.random.default_rng(0)
    S = rng.uniform(-1, 1, (n, dim))
    if continuous:
        A = S[:, :adim] * 0.5 + rng.normal(0, 0.01, (n, adim))
    else:
        A = (S[:, 0] > 0).astype(int)
    return S, A


def test_sklearn_policy_shape():
    if not available_sklearn():
        return
    cfg = Config()
    S, A = _toy(continuous=True)
    p = make_policy(cfg, continuous=True)
    p.fit(S, A)
    out = p.predict(S[:5])
    assert out.shape == (5, 2)
    # 不确定性 ≥ 0
    assert np.all(p.uncertainty(S[:5]) >= 0)


def test_numpy_fallback_shape():
    cfg = Config()
    cfg.use_sklearn = False
    S, A = _toy(continuous=True)
    p = make_policy(cfg, continuous=True)
    p.fit(S, A)
    out = p.predict(S[:7])
    assert out.shape == (7, 2)
    # 纯 numpy 无不确定性估计（返回 0）
    assert np.allclose(p.uncertainty(S[:7]), 0.0)


def test_ensemble_uncertainty_disagreement():
    if not available_sklearn():
        return
    cfg = Config()
    cfg.ensemble_size = 3
    S, A = _toy(continuous=True)
    members = [make_policy(cfg, True) for _ in range(3)]
    for m in members:
        m.fit(S, A)
    ens = EnsemblePolicy(members)
    unc = ens.uncertainty(S[:10])
    assert unc.shape == (10,)
    assert np.all(unc >= 0)
    # 预测为成员均值
    pred = ens.predict(S[:3])
    stacked = np.stack([np.asarray(m.predict(S[:3])) for m in members], 0)
    assert np.allclose(pred, stacked.mean(0), atol=1e-9)


def test_discrete_policy_runs():
    if not available_sklearn():
        return
    cfg = Config()
    S, A = _toy(continuous=False)
    p = make_policy(cfg, continuous=False)
    p.fit(S, A)
    out = p.predict(S[:5])
    assert out.shape == (5,)
    assert set(np.unique(out)).issubset({0, 1})
