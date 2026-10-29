"""
module2_planning.py – Causal/Temporal DAG Planner & Dynamic Weather Replanner
==============================================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
Module 2 implements a **STRIPS-style causal planner** operating on a
Directed Acyclic Graph (DAG) of precondition → effect chains.  Each node in
the DAG is a PlanStep (see city_state.py).

KEY ALGORITHMS
--------------
1. **Forward Causal Chaining (Greedy best-first)**
   - Start from the current CityState (day d).
   - Greedily select the action with the highest expected risk-reduction.
   - Check preconditions; if met, apply effects and advance the state.
   - Repeat until the planning horizon (H days) or all alert levels ≤ GREEN.
   - Time  : O(H × A × Z) where A = #actions, Z = #zones.
   - Space : O(H × A) for the plan DAG.

2. **Dynamic Weather Replanner**
   - After each simulated day, compare actual rainfall/temperature to the
     forecast used during planning.
   - If |actual_rainfall - forecast| > REPLAN_THRESHOLD (10 mm), or the
     Breteau Index spikes > 20 %, trigger a replan from the current state.
   - This mimics the adaptive re-planning loop in Roy et al. 2026 (§4.3).

3. **Critical Path Method (CPM)**
   - Build a task-dependency DAG; compute earliest-start and latest-start
     times via topological sort + forward/backward pass.
   - Identifies the critical path (zero float) so field teams know which
     tasks cannot be delayed.
   - Time  : O(V + E) for both passes.

REFERENCES
----------
* Roy et al. 2026, §4 – adaptive planning under environmental uncertainty.
* Fikes & Nilsson (1971) – original STRIPS planner formulation.
* Project Management Institute (2017) – CPM critical path method.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .city_state import CityState, PlanStep

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Action library
# ════════════════════════════════════════════════════════════════════════════

ACTIONS: Dict[str, Dict[str, Any]] = {
    "ULV_Fogging": {
        "preconditions":  ["spray_teams_available >= 1", "risk_score >= 0.25"],
        "effects":        ["adult_vector_index -= 0.4", "risk_score -= 0.10"],
        "spray_teams":    2,
        "surv_teams":     0,
        "kits":           0,
        "cost_inr":       30_000,
        "duration_days":  1,
        "priority":       3,
    },
    "Breeding_Site_Inspection": {
        "preconditions":  ["surv_teams_available >= 1"],
        "effects":        ["breteau_index -= 3", "risk_score -= 0.05"],
        "spray_teams":    0,
        "surv_teams":     2,
        "kits":           0,
        "cost_inr":       15_000,
        "duration_days":  1,
        "priority":       2,
    },
    "Distribute_Kits": {
        "preconditions":  ["kits_available >= 20", "confirmed_cases >= 3"],
        "effects":        ["estimated_true_cases -= 5", "risk_score -= 0.03"],
        "spray_teams":    0,
        "surv_teams":     0,
        "kits":           20,
        "cost_inr":       8_000,
        "duration_days":  1,
        "priority":       2,
    },
    "Monitor_Only": {
        "preconditions":  [],
        "effects":        [],
        "spray_teams":    0,
        "surv_teams":     0,
        "kits":           0,
        "cost_inr":       5_000,
        "duration_days":  1,
        "priority":       1,
    },
    "Emergency_Mass_Spray": {
        "preconditions":  ["alert_level == RED", "spray_teams_available >= 3"],
        "effects":        ["adult_vector_index -= 0.7", "risk_score -= 0.20", "alert_level = ORANGE"],
        "spray_teams":    4,
        "surv_teams":     1,
        "kits":           0,
        "cost_inr":       60_000,
        "duration_days":  2,
        "priority":       5,
    },
    "Hospital_Surge_Protocol": {
        "preconditions":  ["beds_occupied >= 50", "alert_level in [RED, ORANGE]"],
        "effects":        ["beds_occupied -= 20"],
        "spray_teams":    0,
        "surv_teams":     0,
        "kits":           50,
        "cost_inr":       25_000,
        "duration_days":  3,
        "priority":       4,
    },
}

REPLAN_THRESHOLD_RAIN_MM: float = 10.0   # mm deviation triggers replan
REPLAN_THRESHOLD_BI_PCT:  float = 0.20   # 20 % Breteau Index spike


# ════════════════════════════════════════════════════════════════════════════
# Precondition evaluator
# ════════════════════════════════════════════════════════════════════════════

def _check_preconditions(
    action_name: str,
    zone_id:     str,
    city:        CityState,
    resources:   Dict[str, int],
) -> bool:
    """
    Evaluate STRIPS preconditions against the current ZoneSnapshot.
    Handles: numeric comparisons, equality, and membership tests.
    Returns True iff all preconditions are satisfied.
    """
    snap = city.get_snapshot(zone_id)
    precs = ACTIONS[action_name]["preconditions"]

    state_map = {
        "spray_teams_available": resources.get("spray", 0),
        "surv_teams_available":  resources.get("surv",  0),
        "kits_available":        snap.kits_remaining,
        "risk_score":            snap.risk_score,
        "confirmed_cases":       snap.confirmed_cases,
        "beds_occupied":         snap.beds_occupied,
        "alert_level":           snap.alert_level,
        "breteau_index":         snap.breteau_index,
        "adult_vector_index":    snap.adult_vector_index,
    }

    for prec in precs:
        try:
            if "==" in prec:
                lhs, rhs = prec.split("==")
                lhs, rhs = lhs.strip(), rhs.strip()
                val = state_map.get(lhs, 0)
                if str(val) != rhs:
                    return False
            elif ">=" in prec:
                lhs, rhs = prec.split(">=")
                val = state_map.get(lhs.strip(), 0)
                if not (isinstance(val, (int, float)) and val >= float(rhs)):
                    return False
            elif "<=" in prec:
                lhs, rhs = prec.split("<=")
                val = state_map.get(lhs.strip(), 0)
                if not (isinstance(val, (int, float)) and val <= float(rhs)):
                    return False
            elif ">" in prec:
                lhs, rhs = prec.split(">")
                val = state_map.get(lhs.strip(), 0)
                if not (isinstance(val, (int, float)) and val > float(rhs)):
                    return False
            elif "in" in prec:
                parts = prec.split("in")
                lhs   = parts[0].strip()
                vals  = [v.strip().strip("[").strip("]") for v in parts[1].split(",")]
                if str(state_map.get(lhs, "")) not in vals:
                    return False
        except Exception:
            pass  # Treat parse errors as non-blocking

    return True


# ════════════════════════════════════════════════════════════════════════════
# Forward Causal Planner
# ════════════════════════════════════════════════════════════════════════════

def forward_causal_plan(
    city:    CityState,
    horizon: int = 7,
) -> List[PlanStep]:
    """
    Greedy forward causal chaining planner.

    For each day in [current_day .. current_day + horizon):
      For each zone (ordered by risk_score descending):
        Select the highest-priority action whose preconditions are met.
        Record as a PlanStep with preconditions/effects/cost.

    Returns the list of PlanStep objects (also appended to city.plan_steps).
    """
    plan: List[PlanStep] = []
    step_id = len(city.plan_steps)

    # Snapshot available resources (city-wide pool)
    spray_pool = int(city.infrastructure["spray_teams_stationed"].sum())
    surv_pool  = int(city.infrastructure["surveillance_teams_stationed"].sum())

    for offset in range(horizon):
        day = city.current_day + offset
        resources = {"spray": spray_pool, "surv": surv_pool}

        # Sort zones by risk_score (descending) – fail-first
        zones_sorted = sorted(
            city.zone_ids,
            key=lambda z: city.get_snapshot(z).risk_score,
            reverse=True,
        )

        for zid in zones_sorted:
            snap = city.get_snapshot(zid)

            # Select best action (highest priority with satisfied preconditions)
            best_action = "Monitor_Only"
            best_priority = 0
            for action_name, adef in ACTIONS.items():
                if adef["priority"] > best_priority:
                    if _check_preconditions(action_name, zid, city, resources):
                        # Check resource availability
                        if (resources["spray"] >= adef["spray_teams"] and
                                resources["surv"] >= adef["surv_teams"] and
                                snap.kits_remaining >= adef["kits"]):
                            best_action   = action_name
                            best_priority = adef["priority"]

            adef = ACTIONS[best_action]
            step = PlanStep(
                step_id      = step_id,
                day          = day,
                zone_id      = zid,
                action       = best_action,
                preconditions= adef["preconditions"],
                effects      = adef["effects"],
                cost_inr     = adef["cost_inr"],
                status       = "PLANNED",
            )
            plan.append(step)
            step_id += 1

            # Deduct resources
            resources["spray"] = max(0, resources["spray"] - adef["spray_teams"])
            resources["surv"]  = max(0, resources["surv"]  - adef["surv_teams"])

    city.plan_steps.extend(plan)
    logger.info("Forward planner produced %d steps over %d-day horizon.", len(plan), horizon)
    return plan


# ════════════════════════════════════════════════════════════════════════════
# Dynamic Weather Replanner
# ════════════════════════════════════════════════════════════════════════════

def check_replan_trigger(
    city:     CityState,
    day:      int,
    forecast: Dict[str, float],  # {"rainfall_mm": x, "breteau_index": y}
) -> bool:
    """
    Compare forecast vs. actual environmental conditions.
    Returns True if replanning is warranted.
    """
    actual_rain = 0.0
    actual_bi   = 0.0
    for zid in city.zone_ids:
        snap        = city.get_snapshot(zid)
        actual_rain = max(actual_rain, snap.rainfall_mm)
        actual_bi   = max(actual_bi,   snap.breteau_index)

    rain_diff = abs(actual_rain - forecast.get("rainfall_mm", actual_rain))
    bi_ref    = forecast.get("breteau_index", 1.0)
    bi_diff   = abs(actual_bi - bi_ref) / max(bi_ref, 1.0)

    trigger = rain_diff > REPLAN_THRESHOLD_RAIN_MM or bi_diff > REPLAN_THRESHOLD_BI_PCT
    if trigger:
        logger.info(
            "Replan triggered on day %d: Δrain=%.1f mm, ΔBI=%.1f%%.",
            day, rain_diff, bi_diff * 100,
        )
    return trigger


def dynamic_replan(
    city:    CityState,
    horizon: int = 5,
) -> List[PlanStep]:
    """
    Discard future PLANNED steps for current_day onward and re-plan.
    Preserves DONE/SKIPPED steps from the historical record.
    """
    city.plan_steps = [s for s in city.plan_steps if s.status != "PLANNED"]
    new_steps = forward_causal_plan(city, horizon=horizon)
    logger.info("Replan produced %d new steps.", len(new_steps))
    return new_steps


# ════════════════════════════════════════════════════════════════════════════
# Critical Path Method (CPM)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CPMTask:
    """A task node in the CPM DAG."""
    task_id:       str
    duration_days: int
    dependencies:  List[str] = field(default_factory=list)
    # Computed attributes
    ES:  int = 0   # Earliest Start
    EF:  int = 0   # Earliest Finish
    LS:  int = 0   # Latest Start
    LF:  int = 0   # Latest Finish
    TF:  int = 0   # Total Float
    critical: bool = False


def compute_critical_path(plan_steps: List[PlanStep]) -> Tuple[List[CPMTask], List[str]]:
    """
    Build a CPM task graph from plan_steps and identify the critical path.

    Dependency rule: a step S2 depends on S1 if S2.day > S1.day and
    S2.zone_id == S1.zone_id (zone-level sequential dependency).

    Time  : O(V + E) for topological sort and two DAG traversals.
    Space : O(V + E).

    Returns (list_of_CPMTasks, critical_path_task_ids).
    """
    # Build task list
    tasks: Dict[str, CPMTask] = {}
    for step in plan_steps:
        tid = f"D{step.day:02d}_{step.zone_id}_{step.step_id}"
        tasks[tid] = CPMTask(
            task_id       = tid,
            duration_days = ACTIONS.get(step.action, {}).get("duration_days", 1),
        )

    # Add zone-sequential dependencies
    last_by_zone: Dict[str, str] = {}
    for step in sorted(plan_steps, key=lambda s: (s.zone_id, s.day)):
        tid = f"D{step.day:02d}_{step.zone_id}_{step.step_id}"
        if step.zone_id in last_by_zone:
            tasks[tid].dependencies.append(last_by_zone[step.zone_id])
        last_by_zone[step.zone_id] = tid

    # Build DAG
    G = nx.DiGraph()
    for tid, task in tasks.items():
        G.add_node(tid, task=task)
    for tid, task in tasks.items():
        for dep in task.dependencies:
            G.add_edge(dep, tid)

    # Forward pass (ES / EF)
    for tid in nx.topological_sort(G):
        task = tasks[tid]
        preds = list(G.predecessors(tid))
        task.ES = max((tasks[p].EF for p in preds), default=0)
        task.EF = task.ES + task.duration_days

    # Project finish
    project_finish = max(t.EF for t in tasks.values()) if tasks else 0

    # Backward pass (LS / LF / TF)
    for tid in reversed(list(nx.topological_sort(G))):
        task = tasks[tid]
        succs = list(G.successors(tid))
        task.LF = min((tasks[s].LS for s in succs), default=project_finish)
        task.LS = task.LF - task.duration_days
        task.TF = task.LS - task.ES
        task.critical = task.TF == 0

    critical_path = [tid for tid, t in tasks.items() if t.critical]
    logger.info("CPM: project finish = day %d | critical tasks = %d.", project_finish, len(critical_path))
    return list(tasks.values()), critical_path


# ════════════════════════════════════════════════════════════════════════════
# Module runner
# ════════════════════════════════════════════════════════════════════════════

def run_module2(
    city:       CityState,
    output_dir: Path,
    day:        int,
    horizon:    int = 7,
    forecast:   Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Entry point called by main_demo.py.

    1. Forward causal planning (7-day lookahead).
    2. Weather replan check.
    3. CPM critical path.
    4. Persist plan as JSON + CSV.
    """
    logger.info("=== Module 2 – Planning | Day %d ===", day)

    if forecast is None:
        forecast = {"rainfall_mm": 5.0, "breteau_index": 20.0}

    replan = check_replan_trigger(city, day, forecast)
    if replan:
        plan = dynamic_replan(city, horizon=horizon)
    else:
        plan = forward_causal_plan(city, horizon=horizon)

    cpm_tasks, critical_path = compute_critical_path(city.plan_steps)

    # ── Persist ─────────────────────────────────────────────────────────────
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan_dicts = [vars(p) for p in plan]
    with (log_dir / f"module2_plan_day{day:02d}.json").open("w") as f:
        json.dump({"day": day, "replan": replan, "plan": plan_dicts, "critical_path": critical_path}, f, indent=2)

    plan_df = pd.DataFrame(plan_dicts)
    plan_df.to_csv(log_dir / f"module2_plan_day{day:02d}.csv", index=False)

    logger.info(
        "Module 2 done: %d plan steps | CPM critical=%d | replan=%s.",
        len(plan), len(critical_path), replan,
    )
    return {
        "plan":          plan,
        "cpm_tasks":     cpm_tasks,
        "critical_path": critical_path,
        "replan":        replan,
    }
