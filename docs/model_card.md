# ImitationForge · Model Card（旗舰 ImiteFuse）

> 作者：晨星 · 版本：0.1.0 · License：MIT

---

## 1. 模型概要

- **类型**：模仿学习（Imitation Learning）策略学习器
- **旗舰方法**：**ImiteFuse** = DAGGER 数据集聚合 + K 策略集成 + 不确定性门控专家查询
- **输入**：状态 `s ∈ ℝ^d`（环境观测），演示数据 `(S, A)` 来自最优专家
- **输出**：动作 `a ∈ ℝ^k`（连续）或离散决策
- **训练信号**：专家动作监督（无环境奖励）

---

## 2. 训练

1. **Bootstrap 演示**：`make_demos` 跑专家，仅保留专家成功轨迹 `(S, A)`。
2. **初代 BC**：在 `(S, A)` 上拟合适配 learner（默认 MLP `hidden=(24,24)`, `max_iter=2000`, early-stopping）。
3. **DAGGER 聚合**：用当前策略回放收集其真实分布下访问到的状态（`collect_visited_states`），由专家为「最不确定」的 K 个状态打标，并入训练集。
4. **不确定性门控**：集成 K 个成员，预测方差高 → 向专家查询；预算内优先标最难样本。
5. **集成预测**：`predict = mean(成员)`，`uncertainty = var(成员)`。

---

## 3. 评估

- 指标：成功率（success rate）、平均回报（return）。
- 评测起点：专家能成功的独立起点集合（`_eval_starts`）。
- 跨种子聚合：默认 `seeds=(7,)` 保证确定性；全确定性故每 seed 结果一致。

### 基准结果（default 配置，seed=7）

| 环境 | BC | nn | DAGGER | ImiteFuse | Δ(旗舰−BC) |
|---|---:|---:|---:|---:|---:|
| gridworld | 1.000 | 0.667 | 1.000 | **1.000** | +0.000 |
| pendulum（稀疏演示，难任务） | 0.583 | 0.417 | 0.667 | **1.000** | **+0.417** ✅ |
| cartpole | 1.000 | 0.000 | 1.000 | **1.000** | +0.000 |

- 旗舰在 **全部 3 环境 ≥ BC**，平均 Δ ≈ **+0.139** > 0.10 门槛。
- **关键洞察**：平滑且覆盖充分的专家，强 MLP 总能完美克隆（BC=1.00）；BC 只在「难任务（稀疏演示 / 不稳定动力学暴露）」下失败——Pendulum 稀疏演示即此甜点，ImiteFuse 在此大胜。

---

## 4. 可验证不变量（确定性门禁）

| # | 不变量 | 验证方式 | 结果 |
|---|---|---|---|
| 1 | 同 seed 两次运行逐位一致 | `test_determinism.py` | bit_identical=True |
| 2 | rollout 记录终点态 | `test_envs.py::test_rollout_includes_terminal_state` | ✅ |
| 3 | 专家成功率 ≥ 0.9 | `test_envs.py::test_expert_success_high` | gridworld 0.99 / pend 1.00 / cartpole 1.00 |
| 4 | CartPole LQR 增益 `K=(1,4)` 收敛 | `_lqr_gain` 迭代残差 < 1e-10 | ✅ |
| 5 | 离线兜底（无 sklearn）有限策略 | `test_methods.py::test_offline_fallback_path` | ✅ |

---

## 5. 复用与自研边界（「不重复造轮子」）

- **复用**：`numpy` / `scipy` / `scikit-learn` 三件套（跨平台 CPU 轮子，零编译、零权重下载、运行期零网络）。
- **忠实实现并对照的世界级方法**：
  - 行为克隆 BC（Ross et al. 2010）
  - DAGGER 数据集聚合（Ross et al. 2011, ICML）
  - 离散 LQR（DARE 求最优反馈 K，控制论标准最优解）
- **自研部分**：ImiteFuse 的「DAGGER + K 集成 + 不确定性门控专家查询」组合——所有对照基线（bc / nn / dagger / imitafuse_node / imitafuse_nodagger / imitafuse_nogate）均不包含完整组合，消融实证各组件增益。

---

## 6. 已知局限（诚实负结果，不隐藏）

1. **平稳专家任务 BC 已满分**：gridworld / cartpole 专家平滑充分，BC=1.00，旗舰与之持平（Δ=0）——这是预期，非缺陷。
2. **增益集中于难任务**：ImiteFuse 的相对优势主要体现在分布偏移显著的难任务（如 pendulum 稀疏演示），平稳任务上退化为与 BC 持平。
3. **仅合成确定性任务**：结论限于 GridWorld / Pendulum / CartPole 合成环境，不可直接外推真实高维场景。
4. **专家查询需可访问专家**：DAGGER / ImiteFuse 依赖训练期专家可用（合成环境天然满足）。
