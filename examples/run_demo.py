"""端到端演示：生成数据→多算法 benchmark→落盘 benchmark.json。作者：晨星

确定性二次校验：同 seed 两次运行，核心指标（不含耗时）逐位一致。
为守 60s 预算，主基准跑一遍；确定性校验在「轻量子集」（单环境、两方法、小 n_eval）
上跑两遍比对，成本可忽略。
"""

from __future__ import annotations

import json
import time

from core.config import Config
from pipeline.pipeline import format_table, run_benchmark, summarize


def _core(bench: dict) -> dict:
    """抽取确定性比对用的核心指标（排除耗时）。"""
    out = {}
    for env_name, agg in bench["envs"].items():
        out[env_name] = {
            m: {
                "sr_mean": agg[m]["success_rate"]["mean"],
                "sr_std": agg[m]["success_rate"]["std"],
            }
            for m in agg
        }
    return out


def main() -> int:
    cfg = Config().validate()
    # demo ≤60s：核心主线 bc → nn → dagger → imitafuse（覆盖 BC→DAGGER→旗舰），
    # 含全部 3 环境；完整 7 方法消融表交由 CI（重配置）产出。
    methods = ["bc", "nn", "dagger", "imitafuse"]
    t0 = time.time()
    bench = run_benchmark(cfg, methods=methods)
    dt = time.time() - t0
    print(format_table(bench))

    print("\n=== ImiteFuse vs BC 增益（跨环境）===")
    for env_name, s in summarize(bench).items():
        flag = "PASS" if s["delta_success"] >= 0.10 else "低于门槛(0.10)"
        print(
            f"  {env_name:<12} BC={s['bc']:.4f}  ImiteFuse={s['imitafuse']:.4f}  "
            f"Δ={s['delta_success']:+.4f}  [{flag}]"
        )

    # 确定性二次校验（轻量子集，避免主基准跑两遍翻倍超时）
    dcfg = Config().validate()
    dcfg.n_eval = 6
    t1 = time.time()
    b1 = run_benchmark(dcfg, env_names=["cartpole"], methods=["bc", "imitafuse"])
    b2 = run_benchmark(dcfg, env_names=["cartpole"], methods=["bc", "imitafuse"])
    dt2 = time.time() - t1
    deterministic = json.dumps(_core(b1), sort_keys=True) == json.dumps(
        _core(b2), sort_keys=True
    )
    print("\n=== 确定性二次校验（轻量子集）===")
    print(f"  2x subset={dt2:.2f}s  bit_identical={deterministic}")

    bench["_deterministic"] = deterministic
    bench["_runtime_sec"] = [dt, dt2]
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[OK] 已写入 benchmark.json  (主基准 {dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
