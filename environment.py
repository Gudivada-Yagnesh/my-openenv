import random
from enum import Enum
from typing import Dict, Any


class ActionType(str, Enum):
    SCALE_UP = "scale_up_resources"
    SCALE_DOWN = "scale_down_resources"
    COMPRESS = "compress_memory"
    SCHEDULE = "schedule_tasks"
    NO_OP = "no_op"


class CloudResourceEnv:

    TASKS = {
        "easy": {
            "base_workload": 10.0,
            "base_memory": 55.0,
            "variance": 2.0,
            "burst_prob": 0.1,
            "burst_scale": 10.0,
        },
        "medium": {
            "base_workload": 14.0,
            "base_memory": 60.0,
            "variance": 3.5,
            "burst_prob": 0.25,
            "burst_scale": 18.0,
        },
        "hard": {
            "base_workload": 18.0,
            "base_memory": 65.0,
            "variance": 5.0,
            "burst_prob": 0.4,
            "burst_scale": 28.0,
        },
    }

    def __init__(self, task="easy"):
        self.task = task.lower()
        self.max_steps = 50
        self.reset()

    # ---------------- RESET ----------------
    def reset(self):
        cfg = self.TASKS[self.task]

        self.time = 0

        self.cpu_capacity = 100.0
        self.mem_capacity = 100.0

        self.workload = cfg["base_workload"]
        self.memory = cfg["base_memory"]
        self.cpu = self.workload * 2

        self.queue = 0
        self.compression = False

        self.predicted_mem = self.memory
        self.alpha = 0.3

        self.total_reward = 0.0

        self.stability_hist = []
        self.efficiency_hist = []
        self.cost_hist = []

        return self._obs()

    # ---------------- OBS ----------------
    def _obs(self):
        return {
            "cpu": round((self.cpu / self.cpu_capacity) * 100, 2),
            "memory": round((self.memory / self.mem_capacity) * 100, 2),
            "predicted_memory": round((self.predicted_mem / self.mem_capacity) * 100, 2),
            "queue": self.queue,
            "workload": round(self.workload, 2),
            "compression": self.compression,
            "step": self.time,
        }

    # ---------------- STEP ----------------
    def step(self, action: str):

        self.time += 1

        # ---- ACTION EFFECTS ----
        action_cost = 0
        latency_penalty = 0

        if action == ActionType.SCALE_UP.value:
            self.cpu_capacity += 15
            self.mem_capacity += 15
            action_cost = 2

        elif action == ActionType.SCALE_DOWN.value:
            self.cpu_capacity = max(50, self.cpu_capacity - 15)
            self.mem_capacity = max(50, self.mem_capacity - 15)
            action_cost = 1

        elif action == ActionType.COMPRESS.value:
            self.memory *= 0.75
            self.cpu += 8
            self.compression = True
            action_cost = 1.5
            latency_penalty = 1.5

        elif action == ActionType.SCHEDULE.value:
            processed = min(self.queue, 5)
            self.queue -= processed
            latency_penalty = 1

        elif action == ActionType.NO_OP.value:
            pass

        else:
            raise ValueError("Invalid action")

        # ---- WORKLOAD DYNAMICS ----
        cfg = self.TASKS[self.task]

        if random.random() < cfg["burst_prob"]:
            delta = random.uniform(0, cfg["burst_scale"])
        else:
            delta = random.uniform(-cfg["variance"], cfg["variance"])

        self.workload = max(5, self.workload + delta)

        incoming = int(self.workload * random.uniform(0.2, 0.4))
        self.queue += incoming

        # ---- SYSTEM UPDATE ----
        self.cpu = self.workload * 1.5 + self.queue * 0.6
        if self.compression:
            self.cpu += 10

        processed = min(self.queue, int(self.cpu_capacity / 25))
        self.queue -= processed

        self.memory = self.memory * 0.85 + (self.workload * 0.8 + self.queue * 0.3)

        # ---- EMA PREDICTION ----
        self.predicted_mem = (
            self.alpha * self.memory +
            (1 - self.alpha) * self.predicted_mem
        )

        cpu_pct = (self.cpu / self.cpu_capacity) * 100
        mem_pct = (self.memory / self.mem_capacity) * 100

        # ---------------- REWARD ----------------
        stability = max(0, 1 - abs(mem_pct - 70) / 50)
        efficiency = max(0, 1 - cpu_pct / 100)
        cost = max(0, 1 - action_cost / 3)

        reward = (
            0.5 * stability +
            0.3 * efficiency +
            0.2 * cost
            - latency_penalty * 0.1
        )

        done = False
        info = {}

        if mem_pct > 95:
            reward -= 1
            done = True
            info["reason"] = "overflow"

        if self.time >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        self.total_reward += reward

        self.stability_hist.append(stability)
        self.efficiency_hist.append(efficiency)
        self.cost_hist.append(cost)

        if done:
            info["score"] = self._score()

        return self._obs(), round(reward, 4), done, info

    # ---------------- FINAL SCORE ----------------
    def _score(self):
        s = sum(self.stability_hist) / len(self.stability_hist)
        e = sum(self.efficiency_hist) / len(self.efficiency_hist)
        c = sum(self.cost_hist) / len(self.cost_hist)

        score = 0.4 * s + 0.35 * e + 0.25 * c
        return round(max(0, min(1, score)), 4)

    # ---------------- STATE ----------------
    def state(self):
        return {
            "step": self.time,
            "total_reward": self.total_reward,
            "score": self._score(),
        }