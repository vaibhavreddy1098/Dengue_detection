"""
module1_search_csp.py – Search Algorithms & Constraint Satisfaction
====================================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
Two complementary AI problems are solved here:

1. **Resource Routing (Search)**
   Given a source node (depot/hospital) and a destination zone, find the
   optimal path using:
   - UCS  (Uniform-Cost Search) – guaranteed optimal; O((V+E) log V).
   - A*   – heuristic-guided; optimal when h(n) is admissible.

   The heuristic h(n) = Haversine distance to goal × speed_factor.
   This is admissible because actual road distance ≥ straight-line distance.

2. **Resource Allocation CSP (Constraint Satisfaction)**
   Variables  : allocation of spray-teams, kits, surveillance-teams per zone.
   Domains    : [0 .. max_available] per resource type.
   Constraints:
     - Global supply limits (spray_total ≤ available_spray_teams, etc.)
     - Per-zone minimum (if alert=RED, spray_teams ≥ 1)
     - Kit constraint (allocated_kits ≤ zone_kit_capacity)
   Solved with backtracking + AC-3 arc-consistency pre-processing.
   Complexity: AC-3 is O(e·d²) where e=#arcs, d=max domain size.

3. **Bottleneck Diagnoser**
   After solving the CSP, identifies zones where no feasible allocation
   exists given current resource availability and logs the cause.

REFERENCES
----------
* Russell & Norvig (2021) AIPMA 4th ed., Ch. 3 (UCS/A*), Ch. 6 (CSP/AC-3).
* Roy et al. 2026 (M-SDT) – resource-to-zone assignment formulation.
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .city_state import CityState, RoutingResult

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helper – Haversine distance (admissible A* heuristic)
# ════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine formula for great-circle distance.
    Admissible heuristic: never overestimates road distance.
    O(1) time/space.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ════════════════════════════════════════════════════════════════════════════
# 1 · UCS and A* Search
# ════════════════════════════════════════════════════════════════════════════

@dataclass(order=True)
class _PQItem:
    """Priority-queue item for UCS / A*."""
    priority:  float
    path_cost: float
    node:      str  = field(compare=False)
    path:      List[str] = field(compare=False, default_factory=list)
    time_min:  float = field(compare=False, default=0.0)


def _node_coords(city: CityState, node_id: str) -> Tuple[float, float]:
    """Return (lat, lon) for a zone or facility node."""
    z = city.zones[city.zones["zone_id"] == node_id]
    if not z.empty:
        return float(z["centroid_lat"].iloc[0]), float(z["centroid_lon"].iloc[0])
    f = city.infrastructure[city.infrastructure["facility_id"] == node_id]
    if not f.empty:
        return float(f["lat"].iloc[0]), float(f["lon"].iloc[0])
    return 17.72, 83.30  # fallback centroid


def ucs(
    city: CityState,
    source: str,
    destination: str,
    weight_attr: str = "weight",
) -> Optional[RoutingResult]:
    """
    Uniform-Cost Search on city.infra_graph.

    Time  : O((V + E) log V)  – each edge relaxed at most once via min-heap.
    Space : O(V)               – frontier + visited set.

    Returns RoutingResult or None if unreachable.
    """
    G = city.infra_graph
    if source not in G or destination not in G:
        logger.warning("UCS: node %s or %s not in graph.", source, destination)
        return None

    frontier: List[_PQItem] = []
    heapq.heappush(frontier, _PQItem(0.0, 0.0, source, [source], 0.0))
    visited: Dict[str, float] = {}

    while frontier:
        item = heapq.heappop(frontier)
        if item.node in visited and visited[item.node] <= item.path_cost:
            continue
        visited[item.node] = item.path_cost

        if item.node == destination:
            return RoutingResult(
                algorithm   = "UCS",
                source      = source,
                destination = destination,
                path        = item.path,
                cost        = round(item.path_cost, 3),
                time_min    = round(item.time_min, 1),
                day         = city.current_day,
            )

        for nbr, edge_data in G[item.node].items():
            # multi-edge support: take minimum weight
            if isinstance(edge_data, dict) and "weight" in edge_data:
                w = float(edge_data.get(weight_attr, edge_data.get("distance_km", 1.0)))
                t = float(edge_data.get("travel_time_min", 0.0))
            else:
                # multigraph: pick min-weight parallel edge
                edges = edge_data.values() if hasattr(edge_data, "values") else [edge_data]
                w = min(float(e.get(weight_attr, e.get("distance_km", 1.0))) for e in edges)
                t = min(float(e.get("travel_time_min", 0.0)) for e in edges)

            new_cost = item.path_cost + w
            new_time = item.time_min  + t
            if nbr not in visited or visited.get(nbr, math.inf) > new_cost:
                heapq.heappush(
                    frontier,
                    _PQItem(new_cost, new_cost, nbr,
                            item.path + [nbr], new_time),
                )
    return None


def astar(
    city: CityState,
    source: str,
    destination: str,
    weight_attr: str = "weight",
) -> Optional[RoutingResult]:
    """
    A* Search with Haversine admissible heuristic.

    Time  : O((V + E) log V) – same as UCS in worst case; typically faster.
    Space : O(V).

    The heuristic converts straight-line km to edge-weight units by dividing
    by the mean road-distance/km ratio (≈1.0 since weight=distance_km).
    """
    G = city.infra_graph
    if source not in G or destination not in G:
        logger.warning("A*: node %s or %s not in graph.", source, destination)
        return None

    goal_lat, goal_lon = _node_coords(city, destination)

    def h(node: str) -> float:
        lat, lon = _node_coords(city, node)
        return haversine_km(lat, lon, goal_lat, goal_lon)

    frontier: List[_PQItem] = []
    g_cost = {source: 0.0}
    heapq.heappush(frontier, _PQItem(h(source), 0.0, source, [source], 0.0))
    visited: set = set()

    while frontier:
        item = heapq.heappop(frontier)
        if item.node in visited:
            continue
        visited.add(item.node)

        if item.node == destination:
            return RoutingResult(
                algorithm   = "A*",
                source      = source,
                destination = destination,
                path        = item.path,
                cost        = round(item.path_cost, 3),
                time_min    = round(item.time_min, 1),
                day         = city.current_day,
            )

        for nbr, edge_data in G[item.node].items():
            if isinstance(edge_data, dict) and "weight" in edge_data:
                w = float(edge_data.get(weight_attr, edge_data.get("distance_km", 1.0)))
                t = float(edge_data.get("travel_time_min", 0.0))
            else:
                edges = edge_data.values() if hasattr(edge_data, "values") else [edge_data]
                w = min(float(e.get(weight_attr, e.get("distance_km", 1.0))) for e in edges)
                t = min(float(e.get("travel_time_min", 0.0)) for e in edges)

            tentative_g = item.path_cost + w
            if nbr not in g_cost or tentative_g < g_cost[nbr]:
                g_cost[nbr] = tentative_g
                f = tentative_g + h(nbr)
                heapq.heappush(
                    frontier,
                    _PQItem(f, tentative_g, nbr,
                            item.path + [nbr], item.time_min + t),
                )
    return None


# ════════════════════════════════════════════════════════════════════════════
# 2 · Resource Allocation CSP with Backtracking + AC-3
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CSPResult:
    """Output of the resource-allocation CSP solver."""
    day:         int
    feasible:    bool
    assignment:  Dict[str, Dict[str, int]]   # zone → {spray, surv, kits}
    violations:  List[str]
    bottlenecks: List[str]


def _build_domains(city: CityState) -> Dict[str, Dict[str, List[int]]]:
    """
    Build variable domains for the CSP.

    Variables  : {zone_id: {resource: value}}
    Domains    : integer ranges [0..max_available] per zone per resource.
    Reduction  : RED-alert zones get a minimum of 1 spray team.
    """
    # Total available resources (city-wide)
    spray_total = int(city.infrastructure["spray_teams_stationed"].sum())
    surv_total  = int(city.infrastructure["surveillance_teams_stationed"].sum())

    domains: Dict[str, Dict[str, List[int]]] = {}
    for zid in city.zone_ids:
        snap = city.get_snapshot(zid)
        # Kit capacity is per-zone (facility serving that zone)
        facs      = city.infrastructure[city.infrastructure["zone_id"] == zid]
        kit_cap   = int(facs["daily_pcr_kit_capacity"].sum())
        # Spray domain: if RED, must deploy ≥ 1
        spray_min = 1 if snap.alert_level == "RED" else 0
        domains[zid] = {
            "spray":  list(range(spray_min, spray_total + 1)),
            "surv":   list(range(0, surv_total + 1)),
            "kits":   list(range(0, kit_cap + 1)),
        }
    return domains, spray_total, surv_total


def _check_global(
    assignment: Dict[str, Dict[str, int]],
    spray_total: int,
    surv_total: int,
) -> bool:
    """Check global resource cap constraints."""
    used_spray = sum(v.get("spray", 0) for v in assignment.values())
    used_surv  = sum(v.get("surv",  0) for v in assignment.values())
    return used_spray <= spray_total and used_surv <= surv_total


def ac3(
    domains: Dict[str, Dict[str, List[int]]],
    zones: List[str],
    spray_total: int,
    surv_total: int,
) -> bool:
    """
    AC-3 arc consistency for the global resource constraints.

    We model the global constraint as binary arcs between every pair of zones.
    For each arc (Xi, Xj):  domain(Xi) retains value x iff ∃ y in domain(Xj)
    such that x + y ≤ global_cap.

    Time  : O(e · d²)  where e = Z² arc pairs, d = max domain size.
    Space : O(e).

    Returns False if any domain becomes empty (no solution possible).
    """
    # Build arc queue
    queue = deque()
    for zi in zones:
        for zj in zones:
            if zi != zj:
                queue.append((zi, zj))

    while queue:
        xi, xj = queue.popleft()
        revised = False
        new_spray, new_surv = [], []

        for sx in domains[xi]["spray"]:
            # ∃ sy in domain[xj]["spray"] such that sx+sy ≤ spray_total
            if any(sx + sy <= spray_total for sy in domains[xj]["spray"]):
                new_spray.append(sx)
            else:
                revised = True

        for sv in domains[xi]["surv"]:
            if any(sv + sv2 <= surv_total for sv2 in domains[xj]["surv"]):
                new_surv.append(sv)
            else:
                revised = True

        if not new_spray:
            return False  # domain wipe-out
        if not new_surv:
            return False

        if revised:
            domains[xi]["spray"] = new_spray
            domains[xi]["surv"]  = new_surv
            for zk in zones:
                if zk != xi:
                    queue.append((zk, xi))
    return True


def backtracking_csp(city: CityState) -> CSPResult:
    """
    Solve the daily resource-allocation CSP using Backtracking + AC-3.

    Strategy:
    - Variable ordering: highest risk_score first (fail-first heuristic).
    - Value ordering  : highest allocation first (greedy preference).
    - Forward checking: after each assignment check global caps.

    Time  : O(d^Z) worst case (exponential) but pruned heavily by AC-3
            and forward checking. In practice Z ≤ 10 and d ≤ 10.
    Space : O(Z · d) for domain copies.
    """
    zones = sorted(
        city.zone_ids,
        key=lambda z: city.get_snapshot(z).risk_score,
        reverse=True,
    )
    domains, spray_total, surv_total = _build_domains(city)
    consistent = ac3(domains, zones, spray_total, surv_total)

    if not consistent:
        return CSPResult(
            day         = city.current_day,
            feasible    = False,
            assignment  = {},
            violations  = ["AC-3 detected no feasible solution (domain wipe-out)."],
            bottlenecks = [f"Insufficient resources for {len(zones)} zones."],
        )

    assignment: Dict[str, Dict[str, int]] = {}
    violations: List[str] = []

    def backtrack(idx: int) -> bool:
        if idx == len(zones):
            return _check_global(assignment, spray_total, surv_total)
        zid = zones[idx]
        facs    = city.infrastructure[city.infrastructure["zone_id"] == zid]
        kit_cap = int(facs["daily_pcr_kit_capacity"].sum()) if not facs.empty else 0

        for spray in sorted(domains[zid]["spray"], reverse=True):
            for surv in sorted(domains[zid]["surv"], reverse=True):
                for kits in [kit_cap, kit_cap // 2, 0]:
                    assignment[zid] = {"spray": spray, "surv": surv, "kits": kits}
                    if _check_global(assignment, spray_total, surv_total):
                        if backtrack(idx + 1):
                            return True
        assignment.pop(zid, None)
        violations.append(f"No assignment found for {zid}")
        return False

    feasible = backtrack(0)
    bottlenecks = _diagnose_bottlenecks(city, assignment, spray_total, surv_total)
    return CSPResult(
        day         = city.current_day,
        feasible    = feasible,
        assignment  = assignment,
        violations  = violations,
        bottlenecks = bottlenecks,
    )


def _diagnose_bottlenecks(
    city:        CityState,
    assignment:  Dict[str, Dict[str, int]],
    spray_total: int,
    surv_total:  int,
) -> List[str]:
    """
    Identify resource bottlenecks by comparing demand vs. supply.

    Demand  = sum of minimum allocations for RED zones.
    Supply  = total available (from infrastructure).
    Surplus = supply − demand.
    """
    bottlenecks = []
    red_zones    = [z for z in city.zone_ids if city.get_snapshot(z).alert_level == "RED"]
    demand_spray = len(red_zones)            # 1 team min per RED zone
    demand_surv  = len(red_zones)
    surplus_s    = spray_total - demand_spray
    surplus_v    = surv_total  - demand_surv

    if surplus_s < 0:
        bottlenecks.append(
            f"SPRAY SHORTAGE: need {demand_spray} teams for RED zones, "
            f"only {spray_total} available (deficit {-surplus_s})."
        )
    if surplus_v < 0:
        bottlenecks.append(
            f"SURVEILLANCE SHORTAGE: need {demand_surv} teams, "
            f"only {surv_total} available (deficit {-surplus_v})."
        )
    # Kit bottleneck
    for zid in city.zone_ids:
        snap = city.get_snapshot(zid)
        alloc = assignment.get(zid, {})
        if alloc.get("kits", 0) == 0 and snap.alert_level in ("RED", "ORANGE"):
            bottlenecks.append(f"KIT SHORTAGE in {zid}: alert={snap.alert_level}, zero kits allocated.")

    if not bottlenecks:
        bottlenecks.append("No bottlenecks detected – all constraints satisfied.")
    return bottlenecks


# ════════════════════════════════════════════════════════════════════════════
# 3 · Module runner
# ════════════════════════════════════════════════════════════════════════════

def run_module1(
    city:        CityState,
    output_dir:  Path,
    day:         int,
) -> Dict[str, Any]:
    """
    Entry point called by main_demo.py.

    Returns a summary dict with:
    - routing_results : list of RoutingResult for UCS and A*
    - csp_result      : CSPResult
    - bottleneck_log  : logged to city.bottleneck_log
    """
    logger.info("=== Module 1 – Search & CSP | Day %d ===", day)

    # ── 1.a Routing ─────────────────────────────────────────────────────────
    routing_results: List[RoutingResult] = []

    # For each RED/ORANGE zone, route the nearest depot to it
    depots = city.infrastructure[
        city.infrastructure["facility_type"].str.contains("Depot", case=False, na=False)
    ]
    red_zones = [z for z in city.zone_ids
                 if city.get_snapshot(z).alert_level in ("RED", "ORANGE", "YELLOW")]
    if not red_zones:
        red_zones = [city.zone_ids[0]]  # always route at least one

    for zid in red_zones:
        # Pick closest depot by haversine
        zrow = city.zones[city.zones["zone_id"] == zid].iloc[0]
        best_depot = None
        best_dist  = math.inf
        for _, drow in depots.iterrows():
            d = haversine_km(
                float(zrow["centroid_lat"]), float(zrow["centroid_lon"]),
                float(drow["lat"]),          float(drow["lon"]),
            )
            if d < best_dist:
                best_dist  = d
                best_depot = drow["facility_id"]

        if best_depot is None:
            best_depot = city.zone_ids[0]

        for algo, fn in [("UCS", ucs), ("A*", astar)]:
            result = fn(city, best_depot, zid)
            if result:
                routing_results.append(result)
                logger.info(
                    "%s  %s → %s | path=%s | cost=%.2f km | time=%.1f min",
                    algo, best_depot, zid, result.path, result.cost, result.time_min,
                )
            else:
                logger.warning("%s: no path from %s to %s.", algo, best_depot, zid)

    city.routing_results.extend(routing_results)

    # ── 1.b CSP ─────────────────────────────────────────────────────────────
    csp_result = backtracking_csp(city)
    city.bottleneck_log.append({
        "day":         day,
        "feasible":    csp_result.feasible,
        "violations":  csp_result.violations,
        "bottlenecks": csp_result.bottlenecks,
        "assignment":  csp_result.assignment,
    })

    logger.info(
        "CSP day %d: feasible=%s | violations=%d | bottlenecks=%d",
        day, csp_result.feasible, len(csp_result.violations), len(csp_result.bottlenecks),
    )

    # ── Persist logs ────────────────────────────────────────────────────────
    log_path = output_dir / "logs" / f"module1_csp_day{day:02d}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        json.dump(
            {
                "day":     day,
                "routing": [vars(r) for r in routing_results],
                "csp":     {
                    "feasible":    csp_result.feasible,
                    "assignment":  csp_result.assignment,
                    "violations":  csp_result.violations,
                    "bottlenecks": csp_result.bottlenecks,
                },
            },
            f, indent=2,
        )

    return {
        "routing_results": routing_results,
        "csp_result":      csp_result,
    }
