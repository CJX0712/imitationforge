"""CartPole：连续小车倒立摆，开环不稳定，离散 LQR 专家（最优控制）。作者：晨星

为什么这是模仿学习的「试金石」：
倒立摆开环动力学不稳定（重力使摆下落），任何控制误差都会放大→摆倒→任务失败。
行为克隆（BC）只在专家演示状态上拟合；一旦预测偏差使摆微倾，误差沿不稳定动力学
放大，BC 几乎必败。DAGGER/ImiteFuse 在「学习器真实访问到的（含失败的）状态分布」上
向专家补标并集成，结构性修复分布偏移→恢复高成功率。

专家用离散 LQR（线性化 + DARE 求最优反馈 K），保证围绕竖直稳定且小车归中——
这是控制论标准「世界级」解法，确定性、无随机。
（替换原稳定的 pointnav：稳定调节器下 BC 恒满分、无区分度，无法体现方法价值。）
"""

from __future__ import annotations

import numpy as np


class CartPole:
    # 物理参数（连续小车倒立摆）
    M, m, l, g = 1.0, 0.1, 0.5, 9.8
    Fmax = 25.0
    dt = 0.02

    def __init__(self, max_steps: int = 150):
        self.max_steps = max_steps
        self.obs_dim = 5  # x, x_dot, sinθ, cosθ, θ_dot
        self.act_dim = 1
        self.continuous = True
        self.action_names = ["force"]
        # 离散 LQR 最优反馈增益（围绕竖直 θ=0 稳定；确定性、无随机）
        self.K = self._lqr_gain()
        self.x, self.xd, self.th, self.td, self.t = 0.0, 0.0, 0.0, 0.0, 0

    def _lqr_gain(self):
        M, m, l, g, dt = self.M, self.m, self.l, self.g, self.dt
        total = M + m
        a1 = 4.0 / 3.0 - m / total
        A_th_th = g / (l * a1)
        A_th_F = -1.0 / ((total) * l * a1)
        A_x_th = m * g / (total * a1)
        A_x_F = 1.0 / total - m / (total**2 * a1)
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, A_x_th, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, A_th_th, 0.0],
            ]
        )
        B = np.array([[0.0], [A_x_F], [0.0], [A_th_F]])
        Ad = np.eye(4) + A * dt
        Bd = B * dt
        Q = np.diag([1.0, 0.5, 10.0, 2.0])
        R = np.array([[0.1]])
        # 迭代求解离散代数 Riccati 方程 P = AdᵀPAd - AdᵀPBd(R+BdᵀPBd)⁻¹BdᵀPAd + Q
        P = Q.copy()
        for _ in range(500):
            AB = Bd.T @ P @ Ad  # (1,4)
            S = R + Bd.T @ P @ Bd  # (1,1)
            Pn = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ np.linalg.solve(S, AB) + Q
            if np.max(np.abs(Pn - P)) < 1e-10:
                P = Pn
                break
            P = Pn
        AB = Bd.T @ P @ Ad
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, AB)  # (1,4)
        return K

    def reset(self, start):
        self.x, self.xd = float(start[0]), float(start[1])
        ang = float(start[2])
        self.th = ((ang + np.pi) % (2 * np.pi)) - np.pi  # 归一到 [-π,π]，0=竖直
        self.td = float(start[3])
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.array([self.x, self.xd, np.sin(self.th), np.cos(self.th), self.td])

    @staticmethod
    def _ang(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def expert(self, state):
        x, xd = state[0], state[1]
        th = np.arctan2(state[2], state[3])  # 0=竖直向上
        td = state[4]
        X = np.array([x, xd, th, td])
        u = -(self.K @ X).item()
        return np.array([u])

    def clip_action(self, a):
        a = float(np.asarray(a, float).ravel()[0])
        return float(np.clip(a, -self.Fmax, self.Fmax))

    def step(self, action):
        F = self.clip_action(action)
        M, m, l, g, dt = self.M, self.m, self.l, self.g, self.dt
        s = np.sin(self.th)
        c = np.cos(self.th)
        total = M + m
        temp = (-F - m * l * self.td**2 * s) / total
        thdd = (g * s + c * temp) / (l * (4.0 / 3.0 - m * c**2 / total))
        xdd = (F + m * l * (thdd * c - self.td**2 * s)) / total
        self.xd += xdd * dt
        self.td += thdd * dt
        self.x += self.xd * dt
        self.x = np.clip(self.x, -5.0, 5.0)
        self.th += self.td * dt
        self.t += 1
        r = -1.0
        done = False
        ang = self._ang(self.th)
        upright = abs(ang) < 0.35 and abs(self.td) < 1.5
        if upright:
            r = 100.0
        if abs(ang) > np.pi / 2:  # 倒下超过 90° → 终局失败
            r = -100.0
            done = True
        if self.t >= self.max_steps:
            done = True
        return self._obs(), r, done, {}

    def is_success(self, res):
        if not res.states:
            return False
        s = res.states[-1]
        th = np.arctan2(s[2], s[3])
        ang = self._ang(th)
        return abs(ang) < 0.35

    def sample_starts(self, rng, n):
        # 近竖直扰动域：本基准聚焦「不稳定平衡」——BC 的拟合误差沿不稳定动力学
        # 放大→摆倒失败；DAGGER/ImiteFuse 在真实访问分布上补标→恢复。
        starts = []
        while len(starts) < n:
            x = rng.uniform(-1.0, 1.0)
            xd = rng.uniform(-1.0, 1.0)
            th = rng.uniform(-0.4, 0.4)
            td = rng.uniform(-1.2, 1.2)
            starts.append(np.array([x, xd, th, td]))
        return starts
