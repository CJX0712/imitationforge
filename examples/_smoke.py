"""临时 smoke：全方法 × 三环境（精简默认规模），看对比度 + 计时。作者：晨星"""

from __future__ import annotations

import time

from core.config import Config
from pipeline.pipeline import format_table, run_benchmark

cfg = Config().validate()
print(
    "config:",
    {
        k: getattr(cfg, k)
        for k in ["n_demo", "n_eval", "dagger_rounds", "ensemble_size", "seeds"]
    },
)
t0 = time.time()
bench = run_benchmark(
    cfg,
    env_names=["gridworld", "pendulum", "cartpole"],
    methods=[
        "bc",
        "nn",
        "dagger",
        "imitafuse_node",
        "imitafuse_nodagger",
        "imitafuse_nogate",
        "imitafuse",
    ],
)
print(format_table(bench))
print(f"elapsed={time.time() - t0:.1f}s")
