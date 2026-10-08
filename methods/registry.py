"""方法注册表。作者：晨星"""

from __future__ import annotations

from .bc import train_bc
from .dagger import train_dagger
from .imitafuse import train_ensemble_bc, train_imitafuse
from .nn import train_nn

REGISTRY = {
    "bc": train_bc,
    "nn": train_nn,
    "dagger": lambda e, ds, da, st, c: train_dagger(e, ds, da, st, c, ensemble=False),
    "imitafuse_node": lambda e, ds, da, st, c: train_dagger(
        e, ds, da, st, c, ensemble=True
    ),
    "imitafuse_nodagger": train_ensemble_bc,
    "imitafuse_nogate": lambda e, ds, da, st, c: train_imitafuse(
        e, ds, da, st, c, gate=False
    ),
    "imitafuse": lambda e, ds, da, st, c: train_imitafuse(e, ds, da, st, c, gate=True),
}

METHOD_ORDER = [
    "bc",
    "nn",
    "dagger",
    "imitafuse_node",
    "imitafuse_nodagger",
    "imitafuse_nogate",
    "imitafuse",
]
