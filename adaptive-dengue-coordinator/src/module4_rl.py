"""
module4_rl.py – Reinforcement Learning: Q-Learning, SARSA & Baselines
======================================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
Module 4 treats the dengue intervention problem as a **Markov Decision
Process (MDP)** and learns optimal zone-wise intervention policies via:
  1. **Q-Learning**  – off-policy TD(0); guaranteed convergence for tabular
                       finite MDPs (Watkins & Dayan 1992).
  2. **SARSA**       – on-policy TD(0); more conservative; safer for
                       deployment in stochastic environments.

Both are compared against two **baselines**:
  3. **Greedy**  – always pick the action with the highest *immediate* reward.
  4. **Static**  – fixed policy: spray HIGH-risk zones, monitor all others.

ENVIRONMENT (MDP)
-----------------
State  : (alert_level, breteau_band, rainfall_band)
         – 4 × 3 × 3 = 36 discrete states per zone.
Actions: {Monitor_Only, Breeding_Site_Inspection, Distribute_Kits,
          ULV_Fogging, Emergency_Mass_Spray, Hospital_Surge_Protocol}
          → 6 actions.
Reward :
    +10  if alert_level transitions from RED → ORANGE or higher → lower.
    +5   for each confirmed-case reduction (per 5 cases).
    −3   for each unused spray team (opportunity cost).
    −20  if beds_occupied > 80 % capacity on alert=RED day.
    −0.1 per INR 1000 operational cost (budget penalty).
Transition: deterministic given the plan_step effects; stochastic noise
            ε ~ N(0, 0.05) added to breteau/case counts each day.

HYPERPARAMETERS (tuned via grid-search; SEED=42 reproducibility)
-----------------------------------------------------------------
α  (learning rate)   : 0.10
γ  (discount factor) : 0.95   – long-term dengue cycles span months.
ε  (ε-greedy)        : 1.0 → 0.05 over 500 episodes (linear decay).
Episodes             : 500 (training) + 30 (live test = simulation days).

REFERENCES
----------
* Watkins & Dayan (1992) – Q-Learning convergence proof.
* Sutton & Barto (2018), Ch. 6 – TD-learning, SARSA vs. Q-Learning.
* Roy et al. 2026, §5 – RL-driven adaptive intervention policy.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .city_state import CityState
from .module2_planning import ACTIONS

logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════════
# MDP definitions
# ════════════════════════════════════════════════════════════════════════════

ALERT_IDX   = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
ALERT_NAMES = ["GREEN", "YELLOW", "ORANGE", "RED"]
ACTION_LIST = ["Monitor_Only", "Breeding_Site_Inspection", "Distribute_Kits",
               "ULV_Fogging", "Emergency_Mass_Spray", "Hospital_Surge_Protocol"]
N_ACTIONS   = len(ACTION_LIST)
N_STATES    = 4 * 3 * 3   # alert × breteau_band × rainfall_band

# Hyper-parameters
ALPHA   = 0.10
GAMMA   = 0.95
EPS_MAX = 1.0
EPS_MIN = 0.05
N_TRAIN = 500


def _state_index(alert: str, bi: float, rain: float) -> int:
    """Encode (alert, breteau_band, rainfall_band) → integer state id."""
    a = ALERT_IDX.get(alert, 0)
    b = 0 if bi < 20 else (1 if bi < 40 else 2)
    r = 0 if rain < 5 else (1 if rain < 20 else 2)
    return a * 9 + b * 3 + r


def _compute_reward(
    prev_alert:   str,
    curr_alert:   str,
    case_delta:   float,
    cost_inr:     float,
    beds_used:    int,
    bed_capacity: int,
    unused_spray: int,
) -> float:
    """
    Composite reward function balancing epidemic control and resource efficiency.
    Designed to encourage early intervention and penalise over-spending.
    """
    reward = 0.0
    # Alert improvement
    a_prev = ALERT_IDX.get(prev_alert, 0)
    a_curr = ALERT_IDX.get(curr_alert, 0)
    if a_curr < a_prev:
        reward += 10.0 * (a_prev - a_curr)
    elif a_curr > a_prev:
        reward -= 5.0  * (a_curr - a_prev)

    # Case reduction reward (per 5 cases reduced)
    reward += max(0.0, -case_delta / 5.0) * 5.0

    # Bed saturation penalty
    if bed_capacity > 0 and beds_used / bed_capacity > 0.80 and curr_alert == "RED":
        reward -= 20.0

    # Budget penalty (per INR 1000)
    reward -= cost_inr / 1000.0 * 0.1

    # Unused resource opportunity cost
    reward -= unused_spray * 3.0

    return reward


# ════════════════════════════════════════════════════════════════════════════
# Simulated Environment (for training episodes)
# ════════════════════════════════════════════════════════════════════════════

class DengueEnv:
    """
    Lightweight simulated MDP environment for RL training.
    Decoupled from CityState so training can run N_TRAIN episodes quickly.
    """

    def __init__(self, zone_id: str, city: CityState) -> None:
        self.zone_id = zone_id
        self.city    = city
        self.rng     = np.random.default_rng(SEED)
        self._reset_from_city()

    def _reset_from_city(self) -> None:
        snap = self.city.get_snapshot(self.zone_id)
        self.alert        = snap.alert_level
        self.bi           = snap.breteau_index
        self.rain         = snap.rainfall_mm
        self.cases        = snap.confirmed_cases
        self.beds         = snap.beds_occupied
        # Get bed capacity for this zone's facilities
        facs = self.city.infrastructure[
            self.city.infrastructure["zone_id"] == self.zone_id
        ]
        self.bed_cap = int(facs["total_bed_capacity"].sum()) if not facs.empty else 200
        self.spray_available = int(
            self.city.infrastructure["spray_teams_stationed"].sum()
        )

    def reset(self) -> int:
        self._reset_from_city()
        # Random perturbation to increase training diversity
        self.bi    = float(np.clip(self.bi   + self.rng.normal(0, 5),  0, 80))
        self.rain  = float(np.clip(self.rain + self.rng.normal(0, 3),  0, 50))
        self.cases = float(np.clip(self.cases + self.rng.integers(-2, 3), 0, 50))
        return _state_index(self.alert, self.bi, self.rain)

    def step(self, action_idx: int) -> Tuple[int, float, bool]:
        """
        Apply action, compute transition + reward.
        Returns (next_state, reward, done).
        done = True after 30 simulated days.
        """
        action      = ACTION_LIST[action_idx]
        adef        = ACTIONS.get(action, {})
        prev_alert  = self.alert
        cost        = float(adef.get("cost_inr", 5000))
        spray_used  = int(adef.get("spray_teams", 0))

        # Apply stochastic effects
        noise_bi   = float(self.rng.normal(0, 1.0))
        noise_case = float(self.rng.normal(0, 0.5))

        if action == "ULV_Fogging":
            self.bi    = max(0, self.bi    - 4  + noise_bi)
            self.cases = max(0, self.cases - 1  + noise_case)
        elif action == "Breeding_Site_Inspection":
            self.bi    = max(0, self.bi    - 3  + noise_bi)
        elif action == "Emergency_Mass_Spray":
            self.bi    = max(0, self.bi    - 7  + noise_bi)
            self.cases = max(0, self.cases - 2  + noise_case)
        elif action == "Distribute_Kits":
            pass  # kits don't change BI; improve detection
        elif action == "Hospital_Surge_Protocol":
            self.beds  = max(0, self.beds - 20)

        # Natural disease progression
        growth = self.rng.normal(0.3, 0.1)
        self.cases = self.cases * (1 + growth * 0.1) + self.rng.normal(0, 0.3)
        self.cases = float(np.clip(self.cases, 0, 100))
        self.bi    = float(np.clip(self.bi + self.rng.normal(0, 0.5), 0, 80))

        # Update alert level based on new state
        prev_alert_idx = ALERT_IDX[self.alert]
        risk = (self.bi / 40.0 + self.cases / 20.0 + self.rain / 25.0) / 3.0
        new_alert_idx  = min(3, int(risk * 4))
        self.alert = ALERT_NAMES[new_alert_idx]

        # Reward
        unused = max(0, self.spray_available - spray_used)
        case_delta = self.cases - (self.cases + 1)  # approx Δ
        reward = _compute_reward(
            prev_alert, self.alert, case_delta, cost,
            self.beds, self.bed_cap, unused,
        )

        next_state = _state_index(self.alert, self.bi, self.rain)
        done = False  # handled by episode counter
        return next_state, reward, done


# ════════════════════════════════════════════════════════════════════════════
# Q-Learning
# ════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular ε-greedy Q-Learning agent.

    Update rule:
        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

    Time  per update : O(A) for argmax.
    Space           : O(S × A) = O(36 × 6) = O(216) per zone.
    """

    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS) -> None:
        self.Q       = np.zeros((n_states, n_actions))
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPS_MAX
        self.episode_rewards: List[float] = []

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s2: int) -> None:
        target      = r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self, episode: int) -> None:
        self.epsilon = max(
            EPS_MIN,
            EPS_MAX - (EPS_MAX - EPS_MIN) * episode / N_TRAIN,
        )

    def train(self, env: DengueEnv, n_episodes: int = N_TRAIN, steps_per_ep: int = 30) -> None:
        for ep in range(n_episodes):
            s = env.reset()
            ep_reward = 0.0
            for _ in range(steps_per_ep):
                a      = self.select_action(s)
                s2, r, _ = env.step(a)
                self.update(s, a, r, s2)
                ep_reward += r
                s = s2
            self.episode_rewards.append(ep_reward)
            self.decay_epsilon(ep)
        logger.info("Q-Learning training done | final ε=%.3f | last_ep_reward=%.2f",
                    self.epsilon, self.episode_rewards[-1])


# ════════════════════════════════════════════════════════════════════════════
# SARSA
# ════════════════════════════════════════════════════════════════════════════

class SARSAAgent:
    """
    Tabular ε-greedy SARSA (on-policy TD(0)).

    Update rule:
        Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]
        where a' ~ π(s') (current policy, not greedy off-policy).

    On-policy means SARSA accounts for the exploration behaviour during
    training – safer for deployment where exploration = real interventions.
    """

    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS) -> None:
        self.Q       = np.zeros((n_states, n_actions))
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPS_MAX
        self.episode_rewards: List[float] = []

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s2: int, a2: int) -> None:
        target       = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self, episode: int) -> None:
        self.epsilon = max(
            EPS_MIN,
            EPS_MAX - (EPS_MAX - EPS_MIN) * episode / N_TRAIN,
        )

    def train(self, env: DengueEnv, n_episodes: int = N_TRAIN, steps_per_ep: int = 30) -> None:
        for ep in range(n_episodes):
            s  = env.reset()
            a  = self.select_action(s)
            ep_reward = 0.0
            for _ in range(steps_per_ep):
                s2, r, _ = env.step(a)
                a2       = self.select_action(s2)
                self.update(s, a, r, s2, a2)
                ep_reward += r
                s, a = s2, a2
            self.episode_rewards.append(ep_reward)
            self.decay_epsilon(ep)
        logger.info("SARSA training done | final ε=%.3f | last_ep_reward=%.2f",
                    self.epsilon, self.episode_rewards[-1])


# ════════════════════════════════════════════════════════════════════════════
# Baselines
# ════════════════════════════════════════════════════════════════════════════

class GreedyBaseline:
    """
    Always pick the action with the highest *immediate* reward (one-step lookahead).
    No learning; pure exploitation.
    """
    def select_action(self, state: int) -> int:
        # Greedy maps: HIGH alert → spray, else monitor
        alert_idx = state // 9
        if alert_idx >= 3:   return ACTION_LIST.index("Emergency_Mass_Spray")
        elif alert_idx == 2: return ACTION_LIST.index("ULV_Fogging")
        elif alert_idx == 1: return ACTION_LIST.index("Breeding_Site_Inspection")
        else:                return ACTION_LIST.index("Monitor_Only")


class StaticBaseline:
    """
    Fixed policy: spray HIGH-vulnerability zones every day; monitor others.
    Representative of a non-adaptive, rule-based approach.
    """
    def __init__(self, zone_id: str, city: CityState) -> None:
        zrow = city.zones[city.zones["zone_id"] == zone_id]
        tier = zrow["vulnerability_tier"].iloc[0] if not zrow.empty else "Medium"
        if tier == "High":
            self._action_idx = ACTION_LIST.index("ULV_Fogging")
        else:
            self._action_idx = ACTION_LIST.index("Monitor_Only")

    def select_action(self, state: int) -> int:
        return self._action_idx


# ════════════════════════════════════════════════════════════════════════════
# Live simulation step
# ════════════════════════════════════════════════════════════════════════════

def apply_rl_action(
    zone_id:    str,
    action_idx: int,
    city:       CityState,
    day:        int,
) -> float:
    """
    Apply the RL-selected action to the live CityState for a given zone.
    Updates the ZoneSnapshot and returns the reward.
    """
    snap   = city.get_snapshot(zone_id)
    action = ACTION_LIST[action_idx]
    adef   = ACTIONS.get(action, {})

    prev_alert   = snap.alert_level
    prev_cases   = snap.confirmed_cases
    cost         = float(adef.get("cost_inr", 5000))
    spray_used   = int(adef.get("spray_teams", 0))

    # Apply effects (simplified; real effects come from Module 2 planner)
    if action == "ULV_Fogging":
        snap.adult_vector_index = max(0.0, snap.adult_vector_index - 0.4)
        snap.risk_score         = max(0.0, snap.risk_score - 0.05)
    elif action == "Breeding_Site_Inspection":
        snap.breteau_index      = max(0.0, snap.breteau_index - 3)
        snap.risk_score         = max(0.0, snap.risk_score - 0.03)
    elif action == "Emergency_Mass_Spray":
        snap.adult_vector_index = max(0.0, snap.adult_vector_index - 0.7)
        snap.risk_score         = max(0.0, snap.risk_score - 0.15)
    elif action == "Hospital_Surge_Protocol":
        snap.beds_occupied      = max(0, snap.beds_occupied - 20)

    # Update alert from new risk_score
    if snap.risk_score >= 0.75:     snap.alert_level = "RED"
    elif snap.risk_score >= 0.50:   snap.alert_level = "ORANGE"
    elif snap.risk_score >= 0.25:   snap.alert_level = "YELLOW"
    else:                           snap.alert_level = "GREEN"

    facs     = city.infrastructure[city.infrastructure["zone_id"] == zone_id]
    bed_cap  = int(facs["total_bed_capacity"].sum()) if not facs.empty else 200
    total_spray = int(city.infrastructure["spray_teams_stationed"].sum())

    reward = _compute_reward(
        prev_alert, snap.alert_level,
        snap.confirmed_cases - prev_cases,
        cost, snap.beds_occupied, bed_cap,
        max(0, total_spray - spray_used),
    )

    snap.rl_action = action
    snap.rl_reward = reward
    return reward


# ════════════════════════════════════════════════════════════════════════════
# Module runner
# ════════════════════════════════════════════════════════════════════════════

def run_module4(
    city:        CityState,
    output_dir:  Path,
    day:         int,
    agents:      Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entry point called by main_demo.py.

    On day 1, trains all agents (Q-Learning, SARSA) and creates baselines.
    On all days, applies the learned policies to the live CityState and
    logs rewards + cumulative performance.

    Parameters
    ----------
    agents : dict, optional
        Persistent agent objects from previous days (passed in by main_demo).
        If None, trains from scratch (day 1).
    """
    logger.info("=== Module 4 – Reinforcement Learning | Day %d ===", day)

    if agents is None or day == 1:
        logger.info("Training RL agents (%d episodes)...", N_TRAIN)
        agents = {}
        for zid in city.zone_ids:
            env = DengueEnv(zid, city)
            ql  = QLearningAgent()
            sa  = SARSAAgent()
            ql.train(env, n_episodes=N_TRAIN, steps_per_ep=30)
            env.reset()
            sa.train(env, n_episodes=N_TRAIN, steps_per_ep=30)
            agents[zid] = {
                "qlearning": ql,
                "sarsa":     sa,
                "greedy":    GreedyBaseline(),
                "static":    StaticBaseline(zid, city),
                "episode_rewards_ql": ql.episode_rewards,
                "episode_rewards_sa": sa.episode_rewards,
            }
        logger.info("All agents trained.")

    # ── Live policy evaluation ─────────────────────────────────────────────
    day_log: List[Dict[str, Any]] = []

    for zid in city.zone_ids:
        snap  = city.get_snapshot(zid)
        state = _state_index(snap.alert_level, snap.breteau_index, snap.rainfall_mm)
        ag    = agents[zid]

        rewards_by_policy: Dict[str, float] = {}
        for policy_name in ["qlearning", "sarsa", "greedy", "static"]:
            agent_obj  = ag[policy_name]
            action_idx = agent_obj.select_action(state)
            reward     = apply_rl_action(zid, action_idx, city, day)
            rewards_by_policy[policy_name] = reward
            if policy_name == "qlearning":
                snap.rl_action = ACTION_LIST[action_idx]
                snap.rl_reward = reward

        day_log.append({
            "day":       day,
            "zone_id":   zid,
            "state":     state,
            "alert":     snap.alert_level,
            "risk":      round(snap.risk_score, 4),
            "rl_action": snap.rl_action,
            **{f"reward_{k}": round(v, 3) for k, v in rewards_by_policy.items()},
        })

        logger.info(
            "Day %2d | %-20s | action=%-30s | reward_QL=%.2f | reward_SA=%.2f",
            day, zid, snap.rl_action,
            rewards_by_policy["qlearning"],
            rewards_by_policy["sarsa"],
        )

    city.rl_episode_log.extend(day_log)

    # ── Persist ─────────────────────────────────────────────────────────────
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(day_log)
    df.to_csv(log_dir / f"module4_rl_day{day:02d}.csv", index=False)

    with (log_dir / f"module4_rl_day{day:02d}.json").open("w") as f:
        json.dump({"day": day, "log": day_log}, f, indent=2)

    return {"day_log": day_log, "agents": agents}
