# Changelog

All notable changes to **ImitationForge** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) · Versioning: [SemVer](https://semver.org/).

## [0.1.0] — 2026-10-09

### Added
- 旗舰方法 **ImiteFuse**：DAGGER 数据集聚合 + K 策略集成 + 不确定性门控专家查询。
- 模仿学习方法层（7 方法注册）：`bc` / `nn` / `dagger` / `imitafuse` 及消融变体 `imitafuse_node` / `imitafuse_nodagger` / `imitafuse_nogate`。
- 三个确定性合成控制环境 + 最优专家：`gridworld`（航点跟随 + 临界阻尼 PD）、`pendulum`（能量整形，稀疏演示）、`cartpole`（离散 LQR 专家）。
- 策略 learner：`SklearnPolicy`（MLP/Ridge/Tree，Tier-0）+ `NumpyPolicy`（纯 numpy 离线兜底，Tier-1）+ `EnsemblePolicy` + `NearestNeighborPolicy`。
- 流水线编排：单环境评测 + 跨种子聚合 + `benchmark.json` 产出。
- 全局确定性 seed 入口（跨进程稳定偏移），同 seed 两次运行逐位一致。
- 17 项 pytest 单测、`docs/architecture.md`、`docs/model_card.md`、CI 工作流、Dockerfile、Makefile、密钥自查 `preflight.py`、三级降级推送 `gh_push.py`。

### Fixed
- `rollout` 漏算触发 `done` 的终点态 → `is_success` 系统性低估成功率（已修正为终点态写入 `states`）。
- CartPole 专家经离散 LQR（DARE）替代手写 PID，专家成功率 0% → 100%。
- 多维连续动作被 `ravel` 压成 1D 导致 fit 维度不一致 → 保持 2D。
- `cli.py` 相对导入与绝对导入混用 → 统一为绝对导入。
- MLP 迭代 800 → 2000，确保强 MLP 收敛以公平对比。

### Known Limitations
- 平稳专家任务（gridworld / cartpole）BC 已满分，旗舰与之持平（Δ=0），相对优势集中于难任务（pendulum 稀疏演示）。
- 结论限于合成确定性任务，不可直接外推真实高维场景。

[0.1.0]: https://github.com/CJX0712/imitationforge/releases/tag/v0.1.0
