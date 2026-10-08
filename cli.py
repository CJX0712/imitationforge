"""命令行入口。作者：晨星

用法:
  python cli.py --env gridworld pendulum --method bc imitafuse
  python cli.py --out benchmark.json
  python cli.py --seeds 7 11 23 --n-demo 30 --n-eval 40
"""

from __future__ import annotations

import argparse
import json

from core.config import Config
from pipeline.pipeline import format_table, run_benchmark, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ImitaForge · 模仿学习基准")
    ap.add_argument("--env", nargs="*", default=None, help="环境名")
    ap.add_argument("--method", nargs="*", default=None, help="方法名")
    ap.add_argument("--out", default="benchmark.json", help="输出 JSON 路径")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n-demo", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--dagger-rounds", type=int, default=None)
    ap.add_argument("--ensemble-size", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config().validate()
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    if args.n_demo:
        cfg.n_demo = args.n_demo
    if args.n_eval:
        cfg.n_eval = args.n_eval
    if args.dagger_rounds:
        cfg.dagger_rounds = args.dagger_rounds
    if args.ensemble_size:
        cfg.ensemble_size = args.ensemble_size

    bench = run_benchmark(cfg, env_names=args.env, methods=args.method)
    print(format_table(bench))
    print("\n=== ImiteFuse vs BC 增益 ===")
    for env_name, s in summarize(bench).items():
        print(
            f"  {env_name:<12} BC={s['bc']:.4f}  ImiteFuse={s['imitafuse']:.4f}  "
            f"Δ={s['delta_success']:+.4f}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n[OK] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
