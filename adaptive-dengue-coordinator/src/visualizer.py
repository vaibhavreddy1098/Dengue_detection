"""
visualizer.py – Matplotlib / Seaborn / NetworkX Plotting Engine
===============================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
Generates all publication-quality PNG plots (300 DPI) required for the
professor demo.  Called by main_demo.py after each module run.

PLOTS PRODUCED
--------------
1.  city_network_mobility.png        – NetworkX dual-layer graph map
2.  zone_risk_heatmap_dayXX.png      – Seaborn heatmap of Bayesian risk
3.  daily_cases_timeline.png         – Multi-zone confirmed case timeseries
4.  breteau_index_timeseries.png     – Entomological vector pressure
5.  rl_reward_convergence.png        – Q-Learning vs SARSA learning curves
6.  rl_policy_comparison_dayXX.png   – Per-day reward bar chart (4 policies)
7.  csp_resource_allocation_dayXX.png– Stacked bar of CSP assignments
8.  routing_paths_dayXX.png          – UCS vs A* paths on the infra graph
9.  alert_level_timeline.png         – Zone alert transitions over 30 days
10. critical_path_gantt.png          – CPM Gantt chart
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering – no display required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .city_state import CityState
from .module1_search_csp import CSPResult, RoutingResult
from .module2_planning import CPMTask
from .module4_rl import ACTION_LIST, N_TRAIN, QLearningAgent, SARSAAgent

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

DPI   = 300
ALPHA = 0.85

ALERT_COLORS = {
    "GREEN":  "#2ecc71",
    "YELLOW": "#f1c40f",
    "ORANGE": "#e67e22",
    "RED":    "#e74c3c",
}
POLICY_COLORS = {
    "qlearning": "#3498db",
    "sarsa":     "#9b59b6",
    "greedy":    "#e67e22",
    "static":    "#95a5a6",
}


# ════════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ════════════════════════════════════════════════════════════════════════════
# 1 · City Network Graph
# ════════════════════════════════════════════════════════════════════════════

def plot_city_network(city: CityState, plot_dir: Path) -> None:
    """
    Visualise the dual-layer NetworkX graph on a lat/lon canvas.
    Zone nodes are coloured by vulnerability tier; edges by mobility weight.
    """
    G = city.infra_graph
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (graph, title) in zip(
        axes,
        [(city.mobility_graph, "Mobility Graph (commuter flow)"),
         (city.infra_graph,    "Infrastructure Graph (road distance)")],
    ):
        # Build pos dict from lat/lon
        pos: Dict[str, tuple] = {}
        for node in graph.nodes():
            z = city.zones[city.zones["zone_id"] == node]
            if not z.empty:
                pos[node] = (float(z["centroid_lon"].iloc[0]),
                             float(z["centroid_lat"].iloc[0]))
            else:
                f = city.infrastructure[city.infrastructure["facility_id"] == node]
                if not f.empty:
                    pos[node] = (float(f["lon"].iloc[0]), float(f["lat"].iloc[0]))
                else:
                    pos[node] = (83.30, 17.72)

        # Node colours
        node_colors = []
        node_sizes  = []
        for node in graph.nodes():
            z = city.zones[city.zones["zone_id"] == node]
            if not z.empty:
                tier = str(z["vulnerability_tier"].iloc[0])
                c = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}.get(tier, "#95a5a6")
                node_colors.append(c)
                node_sizes.append(400)
            else:
                node_colors.append("#3498db")
                node_sizes.append(200)

        # Edge weights for width
        edges    = list(graph.edges(data=True))
        widths   = [max(0.5, math.log1p(d.get("flow", d.get("weight", 1)))) * 0.5
                    for _, _, d in edges]

        nx.draw_networkx_nodes(graph, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes, alpha=ALPHA)
        nx.draw_networkx_labels(graph, pos, ax=ax,
                                font_size=7, font_color="white", font_weight="bold")
        nx.draw_networkx_edges(graph, pos, ax=ax,
                               width=widths, alpha=0.5, edge_color="#34495e",
                               arrows=isinstance(graph, nx.DiGraph))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="High vulnerability"),
        mpatches.Patch(color="#f39c12", label="Medium vulnerability"),
        mpatches.Patch(color="#2ecc71", label="Low vulnerability"),
        mpatches.Patch(color="#3498db", label="Facility"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Visakhapatnam Dengue Surveillance Network", fontsize=14, fontweight="bold")
    _save(fig, plot_dir / "city_network_mobility.png")


# ════════════════════════════════════════════════════════════════════════════
# 2 · Risk Heatmap
# ════════════════════════════════════════════════════════════════════════════

def plot_risk_heatmap(city: CityState, plot_dir: Path, day: int) -> None:
    """Seaborn heatmap: zones × risk metrics."""
    rows = []
    for zid in city.zone_ids:
        snap = city.get_snapshot(zid)
        rows.append({
            "Zone":          zid.replace("ZONE_", ""),
            "Risk Score":    snap.risk_score,
            "Breteau Index": snap.breteau_index,
            "Cases (true)":  snap.estimated_true_cases,
            "Rainfall (mm)": snap.rainfall_mm,
            "Humidity (%)":  snap.humidity_pct,
            "Temp (°C)":     snap.temp_c,
        })
    df = pd.DataFrame(rows).set_index("Zone")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        df, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Zone Risk Heatmap – Day {day}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Zone")
    _save(fig, plot_dir / f"zone_risk_heatmap_day{day:02d}.png")


# ════════════════════════════════════════════════════════════════════════════
# 3 · Daily Cases Timeline
# ════════════════════════════════════════════════════════════════════════════

def plot_cases_timeline(city: CityState, plot_dir: Path, max_day: int = 30) -> None:
    """Multi-zone confirmed + estimated case timeseries."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for zid in city.zone_ids:
        zone_data = city.clinical_data[city.clinical_data["zone_id"] == zid]
        zone_data = zone_data[zone_data["day_index"] <= max_day].sort_values("day_index")
        if zone_data.empty:
            continue
        zrow = city.zones[city.zones["zone_id"] == zid].iloc[0]
        urf  = float(zrow.get("under_reporting_factor", 3.0))
        label = zid.replace("ZONE_", "")

        axes[0].plot(zone_data["day_index"], zone_data["confirmed_cases"],
                     marker="o", ms=3, label=label)
        axes[1].plot(zone_data["day_index"], zone_data["confirmed_cases"] * urf,
                     marker="s", ms=3, linestyle="--", label=label)

    axes[0].set_title("Confirmed Cases (Reported)", fontsize=12, fontweight="bold")
    axes[1].set_title("Estimated True Cases (Under-Reporting Corrected × URF)", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_ylabel("Cases")
        ax.legend(fontsize=8, ncol=4)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1].set_xlabel("Simulation Day")
    fig.suptitle("Dengue Case Burden – 30-Day Simulation", fontsize=14, fontweight="bold")
    _save(fig, plot_dir / "daily_cases_timeline.png")


# ════════════════════════════════════════════════════════════════════════════
# 4 · Breteau Index Timeseries
# ════════════════════════════════════════════════════════════════════════════

def plot_breteau_timeseries(city: CityState, plot_dir: Path, max_day: int = 30) -> None:
    """Per-zone Breteau Index with WHO alert thresholds."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for zid in city.zone_ids:
        env_data = city.env_data[city.env_data["zone_id"] == zid]
        env_data = env_data[env_data["day_index"] <= max_day].sort_values("day_index")
        if env_data.empty:
            continue
        ax.plot(env_data["day_index"], env_data["breteau_index"],
                marker=".", label=zid.replace("ZONE_", ""))

    ax.axhline(20, color="gold",   linestyle="--", linewidth=1, label="BI=20 (WHO Alert)")
    ax.axhline(50, color="tomato", linestyle="--", linewidth=1, label="BI=50 (WHO Emergency)")
    ax.set_title("Breteau Index (Larval Vector Density) – All Zones", fontsize=13, fontweight="bold")
    ax.set_xlabel("Simulation Day")
    ax.set_ylabel("Breteau Index")
    ax.legend(fontsize=8, ncol=4)
    _save(fig, plot_dir / "breteau_index_timeseries.png")


# ════════════════════════════════════════════════════════════════════════════
# 5 · RL Reward Convergence
# ════════════════════════════════════════════════════════════════════════════

def plot_rl_convergence(
    ql_rewards: List[float],
    sa_rewards: List[float],
    plot_dir:   Path,
    zone_id:    str = "ZONE_CENTRAL",
) -> None:
    """Smoothed learning curves for Q-Learning vs SARSA."""
    def smooth(data: List[float], w: int = 20) -> List[float]:
        if len(data) < w:
            return data
        return [sum(data[max(0, i - w):i]) / min(i, w) for i in range(1, len(data) + 1)]

    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = list(range(1, len(ql_rewards) + 1))

    ax.plot(episodes, smooth(ql_rewards), color=POLICY_COLORS["qlearning"],
            label="Q-Learning (smoothed)", linewidth=2)
    ax.plot(episodes, smooth(sa_rewards), color=POLICY_COLORS["sarsa"],
            label="SARSA (smoothed)", linewidth=2, linestyle="--")
    ax.fill_between(episodes, smooth(ql_rewards), smooth(sa_rewards), alpha=0.15)

    ax.set_title(f"RL Training Convergence – {zone_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.legend()
    _save(fig, plot_dir / f"rl_reward_convergence_{zone_id}.png")


# ════════════════════════════════════════════════════════════════════════════
# 6 · RL Policy Comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_rl_policy_comparison(rl_log: List[Dict], plot_dir: Path, day: int) -> None:
    """Bar chart: per-zone reward breakdown by policy on a given day."""
    day_data = [r for r in rl_log if r["day"] == day]
    if not day_data:
        return

    df = pd.DataFrame(day_data)
    reward_cols = [c for c in df.columns if c.startswith("reward_")]
    if not reward_cols:
        return

    zones  = df["zone_id"].str.replace("ZONE_", "").tolist()
    x      = np.arange(len(zones))
    width  = 0.20
    n      = len(reward_cols)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(reward_cols):
        policy = col.replace("reward_", "")
        ax.bar(x + i * width, df[col].tolist(),
               width, label=policy.replace("_", " ").title(),
               color=POLICY_COLORS.get(policy, "#95a5a6"), alpha=ALPHA)

    ax.set_title(f"RL Policy Reward Comparison – Day {day}", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(zones, rotation=30, ha="right")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    _save(fig, plot_dir / f"rl_policy_comparison_day{day:02d}.png")


# ════════════════════════════════════════════════════════════════════════════
# 7 · CSP Resource Allocation
# ════════════════════════════════════════════════════════════════════════════

def plot_csp_allocation(csp_result: CSPResult, plot_dir: Path, day: int) -> None:
    """Stacked bar chart of CSP resource allocations per zone."""
    if not csp_result.assignment:
        return

    zones  = list(csp_result.assignment.keys())
    spray  = [csp_result.assignment[z].get("spray", 0) for z in zones]
    surv   = [csp_result.assignment[z].get("surv", 0)  for z in zones]
    kits_k = [csp_result.assignment[z].get("kits", 0) / 10 for z in zones]  # scale

    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, spray,  0.5, label="Spray Teams",     color="#e74c3c", alpha=ALPHA)
    ax.bar(x, surv,   0.5, label="Surveillance Teams", color="#3498db", alpha=ALPHA,
           bottom=spray)
    b2 = [s + v for s, v in zip(spray, surv)]
    ax.bar(x, kits_k, 0.5, label="Kits (÷10)",       color="#2ecc71", alpha=ALPHA,
           bottom=b2)

    ax.set_title(f"CSP Resource Allocation – Day {day} | Feasible={csp_result.feasible}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([z.replace("ZONE_", "") for z in zones], rotation=30, ha="right")
    ax.set_ylabel("Units")
    ax.legend()
    _save(fig, plot_dir / f"csp_resource_allocation_day{day:02d}.png")


# ════════════════════════════════════════════════════════════════════════════
# 8 · Routing Paths
# ════════════════════════════════════════════════════════════════════════════

def plot_routing_paths(
    city:    CityState,
    results: List[RoutingResult],
    plot_dir: Path,
    day:     int,
) -> None:
    """Visualise UCS and A* paths on the infrastructure graph."""
    if not results:
        return

    G   = city.infra_graph
    pos: Dict[str, tuple] = {}
    for node in G.nodes():
        z = city.zones[city.zones["zone_id"] == node]
        if not z.empty:
            pos[node] = (float(z["centroid_lon"].iloc[0]),
                         float(z["centroid_lat"].iloc[0]))
        else:
            f = city.infrastructure[city.infrastructure["facility_id"] == node]
            if not f.empty:
                pos[node] = (float(f["lon"].iloc[0]), float(f["lat"].iloc[0]))
            else:
                pos[node] = (83.30, 17.72)

    # Group by algorithm
    algos = list({r.algorithm for r in results})
    n     = len(algos)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(9 * max(n, 1), 7))
    if n == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        algo_results = [r for r in results if r.algorithm == algo and r.day == day]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#bdc3c7", node_size=200, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#95a5a6")

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(algo_results), 1)))
        for idx, result in enumerate(algo_results):
            path  = result.path
            edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, ax=ax,
                edge_color=[colors[idx % len(colors)]],
                width=3, alpha=0.9,
            )
            if path:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[path[0], path[-1]], ax=ax,
                    node_color=[colors[idx % len(colors)]],
                    node_size=400,
                )

        ax.set_title(f"{algo} Routing – Day {day}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle("UCS vs A* Shortest Paths – Resource Routing", fontsize=13, fontweight="bold")
    _save(fig, plot_dir / f"routing_paths_day{day:02d}.png")


# ════════════════════════════════════════════════════════════════════════════
# 9 · Alert Level Timeline
# ════════════════════════════════════════════════════════════════════════════

def plot_alert_timeline(rl_log: List[Dict], plot_dir: Path, max_day: int = 30) -> None:
    """Colour-coded alert level matrix: zones × days."""
    if not rl_log:
        return

    df = pd.DataFrame(rl_log)
    if "alert" not in df.columns or "zone_id" not in df.columns:
        return

    pivot = df.pivot_table(index="zone_id", columns="day", values="alert", aggfunc="first")
    pivot = pivot[[c for c in pivot.columns if c <= max_day]]

    alert_num = pivot.map(
        lambda x: {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}.get(str(x), 0)
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    cmap   = matplotlib.colors.ListedColormap(
        [ALERT_COLORS[k] for k in ["GREEN", "YELLOW", "ORANGE", "RED"]]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(alert_num.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([z.replace("ZONE_", "") for z in pivot.index], fontsize=9)
    ax.set_xlabel("Simulation Day")
    ax.set_title("Alert Level Timeline – All Zones (30 Days)", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["GREEN", "YELLOW", "ORANGE", "RED"])
    _save(fig, plot_dir / "alert_level_timeline.png")


# ════════════════════════════════════════════════════════════════════════════
# 10 · CPM Gantt Chart
# ════════════════════════════════════════════════════════════════════════════

def plot_critical_path_gantt(cpm_tasks: List[CPMTask], plot_dir: Path) -> None:
    """Horizontal Gantt chart with critical path highlighted in red."""
    if not cpm_tasks:
        return

    tasks_sorted = sorted(cpm_tasks, key=lambda t: (t.ES, t.task_id))
    fig, ax      = plt.subplots(figsize=(16, max(4, len(tasks_sorted) * 0.4)))

    for i, task in enumerate(tasks_sorted):
        color = "#e74c3c" if task.critical else "#3498db"
        ax.barh(i, task.duration_days, left=task.ES, color=color, alpha=0.8, height=0.6)
        ax.text(task.ES + task.duration_days / 2, i, task.task_id.split("_")[0],
                ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    ax.set_yticks(range(len(tasks_sorted)))
    ax.set_yticklabels([t.task_id[:30] for t in tasks_sorted], fontsize=6)
    ax.set_xlabel("Day")
    ax.set_title("Critical Path Method (CPM) Gantt Chart", fontsize=13, fontweight="bold")

    red_patch  = mpatches.Patch(color="#e74c3c", label="Critical (TF=0)")
    blue_patch = mpatches.Patch(color="#3498db", label="Non-critical")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    _save(fig, plot_dir / "critical_path_gantt.png")


# ════════════════════════════════════════════════════════════════════════════
# 11 · Bayesian Posterior Summary
# ════════════════════════════════════════════════════════════════════════════

def plot_bayesian_posteriors(bayes_data: List[Dict], plot_dir: Path, day: int) -> None:
    """
    Grouped bar chart: P(AlertLevel) posterior per zone on a given day.
    Shows GREEN/YELLOW/ORANGE/RED probability bars side-by-side.
    """
    if not bayes_data:
        return

    day_data = [r for r in bayes_data if r["day"] == day]
    if not day_data:
        return

    zones  = [r["zone_id"].replace("ZONE_", "") for r in day_data]
    greens = [r.get("P_GREEN",  0) for r in day_data]
    yellows= [r.get("P_YELLOW", 0) for r in day_data]
    oranges= [r.get("P_ORANGE", 0) for r in day_data]
    reds   = [r.get("P_RED",    0) for r in day_data]

    x = np.arange(len(zones))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 1.5 * w, greens,  w, label="P(GREEN)",  color=ALERT_COLORS["GREEN"],  alpha=ALPHA)
    ax.bar(x - 0.5 * w, yellows, w, label="P(YELLOW)", color=ALERT_COLORS["YELLOW"], alpha=ALPHA)
    ax.bar(x + 0.5 * w, oranges, w, label="P(ORANGE)", color=ALERT_COLORS["ORANGE"], alpha=ALPHA)
    ax.bar(x + 1.5 * w, reds,    w, label="P(RED)",    color=ALERT_COLORS["RED"],    alpha=ALPHA)

    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=30, ha="right")
    ax.set_ylabel("Posterior Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Bayesian AlertLevel Posteriors – Day {day}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, plot_dir / f"bayesian_posteriors_day{day:02d}.png")


# ════════════════════════════════════════════════════════════════════════════
# Master caller (called by main_demo after all modules)
# ════════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    city:        CityState,
    output_dir:  Path,
    day:         int,
    csp_result:  Optional[CSPResult]   = None,
    cpm_tasks:   Optional[List[CPMTask]] = None,
    agents:      Optional[Dict]        = None,
    bayes_rows:  Optional[List[Dict]]  = None,
) -> None:
    """
    Generate all plots for the given simulation day.
    Called from main_demo.py after all four modules execute.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_city_network(city, plot_dir)
    except Exception as exc:
        logger.warning("city_network plot failed: %s", exc)

    try:
        plot_risk_heatmap(city, plot_dir, day)
    except Exception as exc:
        logger.warning("risk_heatmap plot failed: %s", exc)

    try:
        plot_cases_timeline(city, plot_dir, max_day=day)
    except Exception as exc:
        logger.warning("cases_timeline plot failed: %s", exc)

    try:
        plot_breteau_timeseries(city, plot_dir, max_day=day)
    except Exception as exc:
        logger.warning("breteau_timeseries plot failed: %s", exc)

    if csp_result is not None:
        try:
            plot_csp_allocation(csp_result, plot_dir, day)
        except Exception as exc:
            logger.warning("csp_allocation plot failed: %s", exc)

    if city.routing_results:
        try:
            plot_routing_paths(city, city.routing_results, plot_dir, day)
        except Exception as exc:
            logger.warning("routing_paths plot failed: %s", exc)

    if city.rl_episode_log:
        try:
            plot_rl_policy_comparison(city.rl_episode_log, plot_dir, day)
        except Exception as exc:
            logger.warning("rl_policy_comparison plot failed: %s", exc)

        try:
            plot_alert_timeline(city.rl_episode_log, plot_dir, max_day=day)
        except Exception as exc:
            logger.warning("alert_timeline plot failed: %s", exc)

    if agents is not None:
        # Plot convergence for the first zone
        zid = list(agents.keys())[0]
        ag  = agents[zid]
        if "episode_rewards_ql" in ag and "episode_rewards_sa" in ag:
            try:
                plot_rl_convergence(
                    ag["episode_rewards_ql"],
                    ag["episode_rewards_sa"],
                    plot_dir, zone_id=zid,
                )
            except Exception as exc:
                logger.warning("rl_convergence plot failed: %s", exc)

    if cpm_tasks is not None:
        try:
            plot_critical_path_gantt(cpm_tasks, plot_dir)
        except Exception as exc:
            logger.warning("critical_path_gantt plot failed: %s", exc)

    if bayes_rows:
        try:
            plot_bayesian_posteriors(bayes_rows, plot_dir, day)
        except Exception as exc:
            logger.warning("bayesian_posteriors plot failed: %s", exc)

    logger.info("All plots generated for day %d → %s", day, plot_dir)
