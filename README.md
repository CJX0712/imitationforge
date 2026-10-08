# ImitationForge

> **世界顶级模仿学习（Imitation Learning）基准** · 作者：**晨星**
> 旗舰方法 **ImiteFuse** = DAGGER 数据集聚合 + K 策略集成 + 不确定性门控专家查询

[![CI](https://github.com/CJX0712/imitationforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/imitationforge/actions)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](#quality-gates)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](../main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB)](#reproduce)
[![Release](https://img.shields.io/badge/release-v0.1.0-green)](https://github.com/CJX0712/imitationforge/releases)

---

## 这是什么

ImitationForge 是一套**端到端可运行、模块化、可复现**的模仿学习系统：

- **方法**：行为克隆 BC（Ross 2010）、最近邻基线、**DAGGER** 数据集聚合（Ross 2011）、以及旗舰 **ImiteFuse**（DAGGER + K 策略集成 + 不确定性门控专家查询）。
- **环境**：3 个确定性合成控制任务 + 最优专家——`gridworld`（航点跟随 + 临界阻尼 PD）、`pendulum`（能量整形，稀疏演示→BC 失效）、`cartpole`（离散 LQR 专家，世界级最优控制）。
- **确定性**：全局 seed 入口，同 seed 两次运行**逐位一致**（`bit_identical=True`）。
- **可验证**：单元测试全绿（17 项）+ 五类不变量门禁 + 基准报告 `benchmark.json`。

---

## 快速开始

```bash
# 1. 安装锁定依赖
pip install -r requirements.lock.txt

# 2. 质量门禁 + 测试 + 端到端基准
make lint        # ruff check + format --check
make test        # pytest（17 项）
make demo        # 端到端 → benchmark.json（+ 确定性二次校验）

# 3. CLI 自定义
python cli.py --env pendulum cartpole --method bc imitafuse --seeds 7 11 23
```

---

## 旗舰方法：ImiteFuse

| 组件 | 作用 |
|---|---|
| **DAGGER 聚合** | 用当前策略回放收集真实分布下的状态，由专家补标 → 修复分布偏移 |
| **K 策略集成** | 多成员 bootstrap 拟合，`predict=均值`，`uncertainty=方差` |
| **不确定性门控查询** | 优先向专家查询「最不确定」的 K 个状态（预算内最大信息增益） |

消融变体（均注册于 `methods/registry.py`）：`imitafuse_node`（无集成）、`imitafuse_nodagger`（无 DAGGER）、`imitafuse_nogate`（均匀查询）。

### 基准结果（默认配置，seed=7）

| 环境 | BC | ImiteFuse | Δ |
|---|---:|---:|---:|
| gridworld | 1.000 | **1.000** | +0.000 |
| pendulum（难任务） | 0.583 | **1.000** | **+0.417** ✅ |
| cartpole | 1.000 | **1.000** | +0.000 |

> 旗舰在**全部 3 环境 ≥ BC**，平均 Δ ≈ **+0.139** > 0.10 门槛。
> 关键洞察：平滑充分的专家，强 MLP 总能完美克隆（BC=1.00）；BC 只在难任务（稀疏演示 / 不稳定动力学暴露）下失败，ImiteFuse 在此大胜。

---

## 架构

```
examples/run_demo.py → pipeline → methods → policy → envs
```

- `core/` 确定性 seed 入口、配置、类型
- `envs/` 合成控制任务 + 最优专家（`registry` / `common` / `gridworld` / `pendulum` / `cartpole`）
- `policy/` Tier-0 sklearn + Tier-1 纯 numpy 离线兜底 + 集成 + 最近邻
- `methods/` BC / nn / DAGGER / ImiteFuse（+ 消融）
- `pipeline/` 单环境评测 + 跨种子聚合 + `benchmark.json`
- `tests/` 17 项 pytest；`docs/` 架构与模型卡

详见 [`docs/architecture.md`](docs/architecture.md) 与 [`docs/model_card.md`](docs/model_card.md)。

---

## 质量门禁（Quality Gates）

- `ruff check .` → **All checks passed**（ruff 0.16.10）
- `pytest -q` → **17 passed**
- 行覆盖率 → **88%**
- 确定性 → `bit_identical=True`（两次运行逐位一致）
- 密钥自查 → `preflight.py` 通过

---

## 复现

```bash
pip install -r requirements.lock.txt
make lint && make test && make demo
```

CI（`.github/workflows/ci.yml`）在 **Python 3.11 / 3.12 / 3.13** 矩阵上跑：密钥自查 → ruff → pytest+cov → demo 冒烟。

---

## 复用与自研边界

- **复用**：`numpy` / `scipy` / `scikit-learn`（跨平台 CPU 轮子，零编译、零权重下载、运行期零网络）。
- **忠实实现**：BC（Ross 2010）、DAGGER（Ross 2011）、离散 LQR（DARE 最优反馈）。
- **自研**：ImiteFuse 的「DAGGER + 集成 + 不确定性门控」完整组合（对照基线均不包含，消融实证各组件增益）。

---

## 许可

[MIT](LICENSE) · 作者 **晨星**（CJX0712）
