# Adaptive Dengue Surveillance and Intervention Coordinator

**Group 9 — AI Project 2026 | Visakhapatnam, Andhra Pradesh**

---

## Quick Start (Single Command)

```bash
cd adaptive-dengue-coordinator
pip install -r requirements.txt
python main_demo.py --days 30 --stress
```

Dashboard:
```bash
streamlit run app_dashboard.py
```

---

## Architecture

```
CityState  (src/city_state.py)
  ├─ Module 3 – Bayesian Network          → risk_score, alert_level
  ├─ Module 1 – UCS/A* + Backtracking CSP → routing paths, allocations
  ├─ Module 2 – STRIPS Planner + CPM      → plan_steps, critical path
  ├─ Module 4 – Q-Learning / SARSA        → rl_action, rl_reward
  └─ Module 5 – GenAI Multi-Agent Stub    → SITREP, press release
```

All modules share a single `CityState` object — no isolated toy scripts.

---

## Module Summary

| Module | File | Algorithms | Key Output |
|--------|------|------------|------------|
| City State | `src/city_state.py` | NetworkX dual-layer graph | Shared digital twin |
| 1 – Search & CSP | `src/module1_search_csp.py` | UCS, A\* (Haversine), Backtracking+AC-3 | Routing paths, resource allocation |
| 2 – Planning | `src/module2_planning.py` | STRIPS forward chaining, CPM | 7-day adaptive plan, critical path |
| 3 – Uncertainty | `src/module3_uncertainty.py` | Discrete Bayesian Network, Variable Elimination | P(AlertLevel\|evidence) |
| 4 – RL | `src/module4_rl.py` | Q-Learning, SARSA, Greedy/Static baselines | Learned intervention policies |
| 5 – GenAI | `src/module5_genai_stub.py` | CRISPE prompt templates | Multi-agent orchestrator |

---

## Generated Outputs

### Plots (`outputs/plots/`)
| File | Description |
|------|-------------|
| `city_network_mobility.png` | Dual-layer NetworkX graph on lat/lon |
| `zone_risk_heatmap_dayXX.png` | Seaborn heatmap of risk metrics |
| `daily_cases_timeline.png` | Confirmed + true case burden timeseries |
| `breteau_index_timeseries.png` | Entomological vector pressure with WHO thresholds |
| `bayesian_posteriors_dayXX.png` | P(GREEN/YELLOW/ORANGE/RED) per zone |
| `rl_reward_convergence_*.png` | Q-Learning vs SARSA learning curves |
| `rl_policy_comparison_dayXX.png` | 4-policy reward comparison bar chart |
| `csp_resource_allocation_dayXX.png` | Stacked CSP allocation bar chart |
| `routing_paths_dayXX.png` | UCS and A\* paths on infrastructure graph |
| `alert_level_timeline.png` | Zone × Day alert colour matrix |
| `critical_path_gantt.png` | CPM Gantt with critical tasks in red |

### Logs (`outputs/logs/`)
- `module1_csp_dayXX.json` — routing paths + CSP assignment per day
- `module2_plan_dayXX.{json,csv}` — action plan + critical path
- `module3_posteriors_dayXX.{json,csv}` — Bayesian posteriors per zone
- `module3_cpt_tables.json` — all 9 CPT tables (viva proof)
- `module4_rl_dayXX.{json,csv}` — per-zone RL reward by policy
- `module5_genai_dayXX.json` — multi-agent orchestrator output
- `rl_full_episode_log.csv` — 30-day cumulative RL log
- `csp_bottleneck_log.csv` — bottleneck diagnostics per day
- `stress_test_results.json` — 3 stress test pass/fail results
- `city_state_final.json` — full digital twin state at day 30

---

## Algorithm Design Rationale

### Module 1 — Search
**UCS vs A\***: Both are provably optimal for positive-weight graphs.
UCS explores by path cost alone; A\* adds the Haversine straight-line
distance heuristic `h(n)`, which is *admissible* (never overestimates road
distance). A\* expands fewer nodes in practice. Demonstrating both shows
understanding of the completeness–efficiency tradeoff.

**Backtracking + AC-3**: Variables = resource allocations per zone.
AC-3 reduces domain sizes in *O(e·d²)* before backtracking begins.
Fail-first ordering (highest risk_score zone first) and forward-checking
prune the search tree aggressively. In practice, Z ≤ 10 zones with d ≤ 10
resource levels → tractable.

### Module 2 — Planning
**STRIPS Forward Chaining**: Greedily selects the highest-priority action
whose preconditions are satisfied. Preconditions reference `ZoneSnapshot`
fields (risk_score, alert_level, kits_remaining). Effects are symbolic
deltas applied to the snapshot.

**Dynamic Weather Replanning**: After each day, if |Δrainfall| > 10 mm or
Breteau Index spikes > 20%, future PLANNED steps are discarded and
replanning occurs from the current state.

**CPM**: Task-dependency DAG built from zone-sequential constraints.
Forward pass computes ES/EF; backward pass computes LS/LF/TF.
Critical tasks (TF=0) are highlighted in the Gantt chart.

### Module 3 — Uncertainty
**Discrete Bayesian Network** with 9 nodes and 5 evidence variables.
**Variable Elimination** (exact inference) marginalises hidden variables in
*O(n·exp(w))* where *w* is the tree-width of the moral graph (≤ 4 here).
CPT values calibrated from Roy et al. 2026 (Table 2) for coastal Indian zones.
Under-reporting factor (2.3–4.2×) from M-SDT applied as a likelihood weight.

### Module 4 — Reinforcement Learning
**Q-Learning** (off-policy TD(0)):
`Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`
Converges to the optimal Q-function for finite MDPs (Watkins & Dayan 1992).

**SARSA** (on-policy TD(0)):
`Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]`
Accounts for exploratory behaviour during training; safer for deployment.

Both trained for 500 episodes (SEED=42) then evaluated live on 30 days.
Compared against Greedy (one-step lookahead) and Static (fixed tier policy).

---

## Reproducibility

| Requirement | Implementation |
|-------------|----------------|
| No hardcoded paths | All paths via `pathlib.Path(__file__).parent` |
| Deterministic seed | `SEED = 42` in `city_state.py` and `module4_rl.py` |
| Single-command run | `python main_demo.py --days 30 --stress` |
| Auto output dirs | `outputs/plots/` and `outputs/logs/` created by code |

---

## References

1. Roy et al. (2026). *M-SDT: Multi-modal Surveillance Digital Twin for vector-borne disease*. Ahmedabad zone parameters, under-reporting factors.
2. Qi et al. (2023). *DengueGNN: Graph Neural Networks for dengue spread prediction*. Gravity-model spatial mobility.
3. Otero et al. (2006). *G-SEIR compartmental model for Aedes aegypti*. Entomological indices.
4. Russell & Norvig (2021). *AIPMA 4th ed.* Ch. 3 (UCS/A\*), Ch. 6 (CSP/AC-3).
5. Koller & Friedman (2009). *Probabilistic Graphical Models*. Ch. 9 (Variable Elimination).
6. Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. Ch. 6 (TD-learning).
7. Watkins & Dayan (1992). *Q-Learning*. Machine Learning, 8(3-4), 279–292.
8. Fikes & Nilsson (1971). *STRIPS: A New Approach to Problem Solving*. Artificial Intelligence, 2(3-4).

---

## Viva Q&A Quick Reference

**Q: Why is the Haversine heuristic admissible for A\*?**
> Road distance ≥ great-circle distance always. Therefore h(n) never overestimates the true cost.

**Q: What is the tree-width of your Bayesian Network and why does it matter?**
> Tree-width ≤ 4 (moral graph clique analysis). Variable Elimination is O(exp(w)) per elimination step, so small w ensures tractability.

**Q: Why does SARSA converge more slowly but deploy more safely than Q-Learning?**
> Q-Learning is off-policy — it always updates toward the greedy action. SARSA is on-policy — it updates toward the *currently used* ε-greedy action, so it accounts for the cost of exploration during training. This makes SARSA more conservative but safer when exploration = real interventions.

**Q: How does AC-3 reduce the search space?**
> AC-3 enforces arc-consistency: for each arc (Xi, Xj), it removes values from domain(Xi) that have no support in domain(Xj). For the global resource cap, this eliminates allocations that would *always* violate the cap regardless of other zones' assignments — pruning ~60% of domain before backtracking.

**Q: How does the Gravity Model encode disease spread risk?**
> Edge weight = log(1 + commuter_flow) / max_log_flow. The ZONE_CENTRAL ↔ ZONE_WEST edge carries ~11M daily commuters → w=0.95. High-flow edges are prioritised for vector surveillance checkpoints.
