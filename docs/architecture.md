# ImitationForge · 系统架构

> 作者：晨星 · 域：模仿学习（Imitation Learning）· 旗舰：ImiteFuse
> 一句话：用 **BC → DAGGER → 集成 + 不确定性门控专家查询** 的结构化流水线，在确定性合成控制任务上修复分布偏移。

---

## 1. 设计原则

| 层 | 世界级选型（复用，不重复造轮子） |
|---|---|
| 数学 / 数值 | `numpy` 2.5.3 数值稳定运算、SVD/Cholesky、`logsumexp` |
| 算法核心 | `scikit-learn` MLPRegressor / Ridge / DecisionTree（世界级监督学习实现） |
| 优化 | `scipy` 求解器、离散代数 Riccati（DARE）求 LQR 最优增益 |
| 工程 | 类型注解、`pytest`、`ruff` 硬门禁、`github actions` CI |
| 可复现 | 锁定依赖版本 + 全局确定性 seed 入口 + 一键脚本 |

**自研部分**（附理由，见 `docs/model_card.md` §5）：ImiteFuse 的「DAGGER 数据集聚合 + K 策略集成 + 不确定性门控专家查询」组合，以及五类确定性不变量门禁框架。三者均不被任一对照基线包含，增益可归因。

---

## 2. 目录结构

```
imitationforge/
├── core/            # 确定性 seed 入口、配置、错误、类型、接口
│   ├── seed.py          # set_all / new_rng：全局 + 每环境偏移，跨进程稳定
│   ├── config.py        # Config dataclass + ENV_IMITATIONFORGE_* 覆盖 + 校验
│   ├── errors.py        # EnvError / PolicyError / ConfigError
│   ├── types.py         # RolloutResult / Demo 等
│   └── interfaces.py    # 环境 / 策略 / 方法抽象接口
├── envs/            # 确定性合成控制任务 + 最优专家
│   ├── registry.py      # 环境注册表（gridworld/pendulum/cartpole）
│   ├── common.py        # rollout / make_demos / evaluate / collect_visited_states / bootstrap
│   ├── gridworld.py     # 航点跟随 + 临界阻尼 PD 专家，撞墙终局
│   ├── pendulum.py      # 能量整形 + 顶端 PD 专家（稀疏演示→BC 失效）
│   └── cartpole.py      # 连续倒立摆，离散 LQR 专家（世界级最优控制）
├── policy/          # 策略 learner（Tier-0 sklearn + Tier-1 纯 numpy 兜底）
│   └── learners.py      # SklearnPolicy / NumpyPolicy / EnsemblePolicy / NearestNeighborPolicy
├── methods/         # 模仿学习方法层（7 方法注册）
│   ├── bc.py            # 行为克隆（Ross 2010）
│   ├── nn.py            # 最近邻强基线
│   ├── dagger.py        # 数据集聚合（Ross 2011）
│   ├── imitafuse.py     # 旗舰：DAGGER + 集成 + 不确定性门控查询
│   └── registry.py      # REGISTRY / METHOD_ORDER（含消融变体）
├── pipeline/        # 编排：单环境评测 + 跨种子聚合 + benchmark.json
│   └── pipeline.py      # run_env / run_benchmark / format_table / summarize
├── tests/           # pytest 单测（17 项）
├── examples/
│   └── run_demo.py      # 端到端基准 → benchmark.json + 确定性二次校验
├── cli.py           # 命令行入口
├── gh_push.py       # 三级降级推送（git → Git Data API → Contents API）
├── preflight.py     # 密钥自查
├── docs/            # architecture.md / model_card.md
├── README.md · LICENSE · CHANGELOG.md
├── requirements.txt · requirements.lock.txt · Dockerfile · Makefile · .gitignore
└── .github/workflows/ci.yml
```

---

## 3. 主调用链（单向无环）

```
run_demo.py
  └─> pipeline.run_benchmark(cfg, env_names, methods)
        └─> for env_name: pipeline.run_env(env_name, cfg, methods)
              ├─> envs.registry.make_env(name)          # 构造环境
              ├─> envs.common.make_demos(env, cand)     # 专家成功轨迹 → (S, A, kept)
              └─> for method m:
                    ├─> methods.registry.REGISTRY[m](env, S, A, kept, cfg)
                    │     └─> policy.learners.make_policy(cfg, continuous)  # sklearn / numpy 兜底
                    └─> envs.common.evaluate(env, policy, eval_starts)
                          └─> rollout(env, ctrl, start, max_steps) → 成功率 / 回报
```

调用方向恒为：`examples/cli → pipeline → methods → policy → envs`，无反向依赖。

---

## 4. 关键不变量（确定性门禁，见 `docs/model_card.md` §4）

1. **全局确定性**：相同 seed → 两次运行逐位一致（`bit_identical=True`）。
2. **rollout 含终点态**：触发 `done` 的那一帧必写入 `states`，否则 `is_success` 漏算末帧（早期 bug 已修）。
3. **专家高成功**：三环境专家成功率 ≥ 0.9（gridworld 0.99 / pendulum 1.00 / cartpole 1.00）。
4. **CartPole LQR 增益形状**：`K` 为 `(1,4)`，增益经 DARE 迭代收敛（残差 < 1e-10）。
5. **离线兜底**：`use_sklearn=False` 时纯 numpy 线性回归 / 最近质心仍可产出有限策略。

---

## 5. 复现

```bash
pip install -r requirements.lock.txt
make lint        # ruff check + format --check
make test        # pytest（17 项）
make demo        # 端到端基准 → benchmark.json（+ 确定性二次校验）
make cov         # 覆盖率报告（当前 88%）
```

CI（`.github/workflows/ci.yml`）在 py3.11–3.13 矩阵上跑：密钥自查 → ruff → pytest+cov → demo 冒烟。
