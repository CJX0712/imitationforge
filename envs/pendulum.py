"""Pendulum 倒立摆：连续动作，能量整形 + 顶端 LQR 专家。作者：晨星

专家：能量控制摆起的经典控制器（Spong 1995），到顶端附近切换为角度/角速度 PD 稳定。
BC 若在某状态预测偏差，能量不足→摆不到顶；DAGGER 在「学习器真实分布」补标修复。
成功判定：整段末态维持竖直（|θ-π|<0.25 且 |ω|<1.0）。
"""

from __future__ import annotations

import numpy as np


class Pendulum:
    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.obs_dim = 3
        self.act_dim = 1
        self.continuous = True
        self.action_names = ["torque"]
        self.m, self.l, self.g, self.b = 1.0, 1.0, 9.8, 0.1
        self.dt = 0.05
        self.umax = 5.0
        self.theta, self.omega, self.t = 0.0, 0.0, 0

    def reset(self, start):
        self.theta = float(start[0])
        self.omega = float(start[1])
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.omega])

    @staticmethod
    def _angdiff(a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def expert(self, state):
        th = np.arctan2(state[1], state[0])
        om = state[2]
        I = self.m * self.l**2
        E = 0.5 * I * om**2 + self.m * self.g * self.l * (1 - np.cos(th))
        Etarget = 2 * self.m * self.g * self.l
        ang = self._angdiff(th, np.pi)
        if abs(ang) < 0.5 and abs(om) < 1.5:
            # 顶端附近：角度/角速度 PD 稳定
            u = -10.0 * ang - 2.0 * om
        else:
            # 能量整形摆动上升：dE/dt = u·ω，需 u 与 ω 同号 ⇒ 升能当 E<Etarget
            s = 1.0 if om >= 0 else -1.0
            u = 0.8 * (Etarget - E) * s
        return float(np.clip(u, -self.umax, self.umax))

    def clip_action(self, a):
        a = float(np.asarray(a, float).ravel()[0])
        return float(np.clip(a, -self.umax, self.umax))

    def step(self, action):
        u = float(np.clip(action, -self.umax, self.umax))
        I = self.m * self.l**2
        alpha = (
            u - self.b * self.omega - self.m * self.g * self.l * np.sin(self.theta)
        ) / I
        self.omega += alpha * self.dt
        self.theta += self.omega * self.dt
        self.t += 1
        ang = self._angdiff(self.theta, np.pi)
        upright = abs(ang) < 0.25 and abs(self.omega) < 1.0
        r = 100.0 if upright else -1.0
        done = self.t >= self.max_steps
        return self._obs(), r, done, {}

    def is_success(self, res):
        if not res.states:
            return False
        s = res.states[-1]
        th = np.arctan2(s[1], s[0])
        ang = self._angdiff(th, np.pi)
        return abs(ang) < 0.25 and abs(s[2]) < 1.0

    def sample_starts(self, rng, n):
        starts = []
        while len(starts) < n:
            starts.append(
                np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0)])
            )
        return starts
