"""GridWorld：连续坐标 + 惯性导航 + 中央带缝隙的墙，BFS 逐格航点 PD 专家。作者：晨星

要点：专家用 BFS 求出到目标的最短自由路径，逐格把「下一格中心」作为 subgoal，
PD 跟踪；相邻格仅 1/N≈0.048 远，PD 不可能过冲撞墙，保证专家必定收敛到目标。
BC 只在专家轨迹状态上学，当学习器因误差进入未见状态（缝隙边缘 / 惯性过冲撞墙）
会撞墙→成功率崩。DAGGER 用专家在「学习器真实分布」上补标，结构性修复分布偏移。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class GridWorld:
    def __init__(self, N: int = 21, max_steps: int = 200):
        self.N = N
        self.max_steps = max_steps
        self.obs_dim = 6
        self.act_dim = 2
        self.continuous = True
        self.action_names = ["ax", "ay"]
        # 目标在墙的右侧，必须穿过缝隙抵达
        self.target = np.array([0.75, 0.5])
        self.wall_x = (0.47, 0.53)
        self.gap = (0.20, 0.80)
        # 与 PointNav 一致的半隐式欧拉 + 阻尼 + 临界阻尼 PD
        self.dt, self.friction, self.accel = 0.05, 0.9, 3.0
        self.k1, self.k2 = 100.0, 20.0
        self.pos, self.vel, self.t = np.zeros(2), np.zeros(2), 0
        self._sg = self.target.copy()
        self._build_bfs()

    def _in_obstacle(self, x, y):
        return self.wall_x[0] <= x <= self.wall_x[1] and not (
            self.gap[0] <= y <= self.gap[1]
        )

    def _build_bfs(self):
        N = self.N
        INF = 10**9
        dist = np.full((N, N), INF)
        tc = (int(self.target[0] * N), int(self.target[1] * N))
        dist[tc] = 0
        q = deque([tc])
        neigh = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        while q:
            cx, cy = q.popleft()
            for dx, dy in neigh:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N and dist[nx, ny] == INF:
                    wx, wy = (nx + 0.5) / N, (ny + 0.5) / N
                    if self._in_obstacle(wx, wy):
                        continue
                    dist[nx, ny] = dist[cx, cy] + 1
                    q.append((nx, ny))
        self.dist = dist

    def reset(self, start):
        self.pos = np.array(start[:2], float)
        self.vel = np.zeros(2)
        self.t = 0
        self._sg = self.target.copy()
        return self._obs()

    def _obs(self):
        return np.array(
            [
                self.pos[0],
                self.pos[1],
                self.vel[0],
                self.vel[1],
                self.target[0],
                self.target[1],
            ]
        )

    def _next_subgoal(self, p):
        N = self.N
        cx = min(N - 1, max(0, int(p[0] * N)))
        cy = min(N - 1, max(0, int(p[1] * N)))
        if self.dist[cx, cy] == 0:
            return self.target.copy()
        best, bestd = None, 10**9
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < N and 0 <= ny < N and self.dist[nx, ny] < bestd:
                bestd = self.dist[nx, ny]
                best = (nx, ny)
        return np.array([(best[0] + 0.5) / N, (best[1] + 0.5) / N])

    def expert(self, state):
        p, v = state[:2], state[2:4]
        if np.linalg.norm(p - self._sg) < 0.5 / self.N:
            self._sg = self._next_subgoal(p)
        u = -self.k1 * (p - self._sg) - self.k2 * v
        return u.astype(float)

    def clip_action(self, a):
        a = np.asarray(a, float)
        amag = np.linalg.norm(a)
        if amag > self.accel:
            a = a / amag * self.accel
        return a

    def step(self, action):
        a = self.clip_action(action)
        self.vel = (self.vel + a * self.dt) * self.friction
        new = self.pos + self.vel * self.dt
        self.t += 1
        if self._in_obstacle(new[0], new[1]):
            # 撞墙 = 终局致命失败（学习器若误入墙体即任务失败）
            self.pos = new
            return self._obs(), -100.0, True, {"collision": True}
        self.pos = np.clip(new, 0.0, 1.0)
        d = float(np.linalg.norm(self.pos - self.target))
        r = -1.0
        done = False
        if d < 0.05:
            r = 100.0
            done = True
        if self.t >= self.max_steps:
            done = True
        return self._obs(), r, done, {}

    def is_success(self, res):
        for s in res.states:
            if np.linalg.norm(s[:2] - self.target) < 0.05:
                return True
        return False

    def sample_starts(self, rng, n):
        starts = []
        while len(starts) < n:
            p = rng.uniform(0.05, 0.95, 2)
            if self._in_obstacle(p[0], p[1]):
                continue
            if np.linalg.norm(p - self.target) < 0.2:
                continue
            starts.append(p.copy())
        return starts
