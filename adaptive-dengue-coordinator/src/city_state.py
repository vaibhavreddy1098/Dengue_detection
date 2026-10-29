"""
city_state.py – Shared Digital-Twin State Object
=================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
This module defines ``CityState``, the single source of truth for the entire
pipeline.  Every upstream module writes results into this object; every
downstream module reads from it.  This strict dataflow discipline ensures
the integration requirement is satisfied.

DESIGN RATIONALE
----------------
* **NetworkX dual-layer graph** – two multigraphs share the same node set
  (zones + facilities) but carry different edge semantics:
    - ``mobility_graph``      : weighted by commuter flow (gravity model,
                                DengueGNN, Qi et al. 2023).  Used by Module 1
                                (UCS/A*) and Module 4 (RL environment).
    - ``infrastructure_graph``: weighted by road distance/time.  Used by
                                Module 1 (resource routing) and Module 2
                                (planning).
* **pandas DataFrames** for time-series data → zero-copy slicing for
  downstream consumers (O(1) column access).
* **Defensive loading** – every optional column is synthesised if absent so
  the pipeline never crashes on partial data.

REFERENCES
----------
* Roy et al. 2026 – M-SDT: Multi-modal Surveillance Digital Twin for
  vector-borne disease; Ahmedabad zone parameters & under-reporting factors.
* Qi et al. 2023 – DengueGNN: Graph Neural Networks for dengue spread
  prediction; gravity-model spatial mobility.
* Otero et al. 2006 – G-SEIR compartmental model for entomological indices.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

# Under-reporting multipliers per tier (Roy et al. 2026, Table 3)
UNDER_REPORTING: Dict[str, float] = {
    "High":   4.2,
    "Medium": 3.1,
    "Low":    2.3,
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ---------------------------------------------------------------------------
# Small helper dataclasses that modules read/write
# ---------------------------------------------------------------------------

@dataclass
class ZoneSnapshot:
    """Live per-zone state updated daily by the pipeline."""
    zone_id:               str
    day:                   int           = 0
    # Epidemiological
    confirmed_cases:       float         = 0.0
    estimated_true_cases:  float         = 0.0   # confirmed × under-reporting
    cumulative_cases:      float         = 0.0
    breteau_index:         float         = 0.0
    adult_vector_index:    float         = 0.0
    # Environmental
    rainfall_mm:           float         = 0.0
    temp_c:                float         = 30.0
    humidity_pct:          float         = 70.0
    # Resource state
    beds_occupied:         int           = 0
    kits_remaining:        int           = 0
    spray_teams_active:    int           = 0
    # Risk / alert
    risk_score:            float         = 0.0   # computed by Module 3
    alert_level:           str           = "GREEN"
    # RL outputs
    rl_action:             Optional[str] = None
    rl_reward:             float         = 0.0


@dataclass
class PlanStep:
    """A single step in the causal/temporal action plan (Module 2)."""
    step_id:        int
    day:            int
    zone_id:        str
    action:         str
    preconditions:  List[str]        = field(default_factory=list)
    effects:        List[str]        = field(default_factory=list)
    cost_inr:       float            = 0.0
    status:         str              = "PENDING"   # PENDING / DONE / SKIPPED


@dataclass
class RoutingResult:
    """Output of Module 1 search algorithms."""
    algorithm:   str
    source:      str
    destination: str
    path:        List[str]
    cost:        float
    time_min:    float
    day:         int


# ---------------------------------------------------------------------------
# CityState – the Digital Twin
# ---------------------------------------------------------------------------

class CityState:
    """
    Shared mutable state for the Adaptive Dengue Coordinator pipeline.

    Attributes
    ----------
    data_dir : Path
        Root of the ``data/`` folder (relative, never absolute).
    zones : pd.DataFrame
        Zone demographics + epidemiological parameters.
    infrastructure : pd.DataFrame
        Healthcare / depot / lab facilities.
    env_data : pd.DataFrame
        Daily environmental surveillance (rainfall, temp, entomological).
    clinical_data : pd.DataFrame
        Daily clinical reporting logs.
    interventions : pd.DataFrame
        Historical and planned interventions.
    mobility_graph : nx.DiGraph
        Zone-to-zone directed graph weighted by normalised commuter flow.
    infra_graph : nx.Graph
        Zone + facility undirected graph weighted by road distance (km).
    zone_snapshots : Dict[str, ZoneSnapshot]
        Live per-zone state, updated each simulation day.
    plan_steps : List[PlanStep]
        Module 2 outputs consumed by Module 4.
    routing_results : List[RoutingResult]
        Module 1 outputs logged for the viva report.
    current_day : int
        Simulation clock (1-indexed).
    rng : np.random.Generator
        Seeded RNG shared across modules.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir: Path = data_dir
        self.rng: np.random.Generator = np.random.default_rng(SEED)

        # ── Raw DataFrames ──────────────────────────────────────────────────
        self.zones:          pd.DataFrame = pd.DataFrame()
        self.infrastructure: pd.DataFrame = pd.DataFrame()
        self.env_data:       pd.DataFrame = pd.DataFrame()
        self.clinical_data:  pd.DataFrame = pd.DataFrame()
        self.interventions:  pd.DataFrame = pd.DataFrame()
        self.edges_df:       pd.DataFrame = pd.DataFrame()

        # ── Graphs ──────────────────────────────────────────────────────────
        self.mobility_graph: nx.DiGraph = nx.DiGraph()
        self.infra_graph:    nx.Graph   = nx.Graph()

        # ── Simulation state ────────────────────────────────────────────────
        self.zone_snapshots:  Dict[str, ZoneSnapshot] = {}
        self.plan_steps:      List[PlanStep]           = []
        self.routing_results: List[RoutingResult]      = []
        self.current_day:     int                      = 1

        # ── Module result stores ─────────────────────────────────────────────
        self.cpt_tables:        Dict[str, Any]         = {}  # Module 3
        self.bayes_posteriors:  Dict[str, Any]         = {}  # Module 3
        self.rl_episode_log:    List[Dict[str, Any]]   = []  # Module 4
        self.bottleneck_log:    List[Dict[str, Any]]   = []  # Module 1 CSP

        self._load_all()

    # ────────────────────────────────────────────────────────────────────────
    # Data loading
    # ────────────────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Load and validate every CSV; synthesise any missing columns."""
        logger.info("Loading CityState from %s", self.data_dir)
        self.zones          = self._load_zones()
        self.infrastructure = self._load_infrastructure()
        self.edges_df       = self._load_edges()
        self.env_data       = self._load_env()
        self.clinical_data  = self._load_clinical()
        self.interventions  = self._load_interventions()
        self._build_graphs()
        self._init_zone_snapshots()
        logger.info(
            "CityState ready: %d zones, %d facilities, %d edge types, "
            "%d env rows, %d clinical rows.",
            len(self.zones),
            len(self.infrastructure),
            len(self.edges_df),
            len(self.env_data),
            len(self.clinical_data),
        )

    # ── zones_demographics.csv ───────────────────────────────────────────────

    def _load_zones(self) -> pd.DataFrame:
        path = self.data_dir / "zones_demographics.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        required = {
            "zone_id":               lambda n: [f"ZONE_{i}" for i in range(n)],
            "zone_name":             lambda n: [f"Zone {i}" for i in range(n)],
            "population":            lambda n: [500_000] * n,
            "area_sqkm":             lambda n: [30.0] * n,
            "centroid_lat":          lambda n: [17.72 + i * 0.01 for i in range(n)],
            "centroid_lon":          lambda n: [83.30 + i * 0.01 for i in range(n)],
            "base_biting_rate_b0":   lambda n: [0.12] * n,
            "vector_to_host_ratio_m": lambda n: [3.0] * n,
            "symptomatic_fraction_p": lambda n: [0.055] * n,
            "baseline_r0":           lambda n: [1.9] * n,
            "vulnerability_tier":    lambda n: ["Medium"] * n,
        }
        if df.empty:
            n = 7
            df = pd.DataFrame({k: v(n) for k, v in required.items()})
        else:
            n = len(df)
            for col, default_fn in required.items():
                if col not in df.columns:
                    logger.warning("Synthesising missing column '%s' in zones.", col)
                    df[col] = default_fn(n)

        df["under_reporting_factor"] = df["vulnerability_tier"].map(
            lambda t: UNDER_REPORTING.get(t, 3.0)
        )
        return df.reset_index(drop=True)

    # ── city_infrastructure.csv ──────────────────────────────────────────────

    def _load_infrastructure(self) -> pd.DataFrame:
        path = self.data_dir / "city_infrastructure.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        defaults: Dict[str, Any] = {
            "facility_id":                "FAC_00",
            "facility_name":              "Unknown",
            "facility_type":              "Hospital",
            "zone_id":                    self.zones["zone_id"].iloc[0] if not self.zones.empty else "ZONE_0",
            "lat":                        17.72,
            "lon":                        83.30,
            "total_bed_capacity":         200,
            "icu_bed_capacity":           20,
            "daily_pcr_kit_capacity":     100,
            "spray_teams_stationed":      2,
            "surveillance_teams_stationed": 2,
            "vehicles_available":         3,
        }
        if df.empty:
            df = pd.DataFrame([defaults])
        else:
            for col, val in defaults.items():
                if col not in df.columns:
                    logger.warning("Synthesising missing column '%s' in infrastructure.", col)
                    df[col] = val
        return df.reset_index(drop=True)

    # ── spatial_network_edges.csv / city_edges.csv ────────────────────────

    def _load_edges(self) -> pd.DataFrame:
        for name in ("spatial_network_edges.csv", "city_edges.csv"):
            path = self.data_dir / name
            if path.exists():
                df = pd.read_csv(path)
                # Normalise column names across both file formats
                rename = {
                    "source_node_id": "source_id",
                    "target_node_id": "target_id",
                    "avg_travel_time_min": "travel_time_min",
                    "mobility_flow_gravity": "daily_commuter_flow",
                }
                df.rename(columns=rename, inplace=True)
                for col, default in [
                    ("source_id", None), ("target_id", None),
                    ("distance_km", 10.0), ("travel_time_min", 20.0),
                    ("daily_commuter_flow", 100_000),
                    ("edge_type", "road_and_mobility"),
                    ("transmission_weight", 0.1),
                ]:
                    if col not in df.columns:
                        df[col] = default
                logger.info("Loaded edges from %s (%d rows).", name, len(df))
                return df.reset_index(drop=True)

        # Synthesise minimal star topology if both files absent
        logger.warning("No edge file found – synthesising star topology.")
        zones = self.zones["zone_id"].tolist()
        rows = []
        for z in zones[1:]:
            rows.append({
                "source_id": zones[0], "target_id": z,
                "distance_km": 10.0, "travel_time_min": 20.0,
                "daily_commuter_flow": 100_000,
                "edge_type": "road_and_mobility",
                "transmission_weight": 0.1,
            })
        return pd.DataFrame(rows)

    # ── daily_environmental_surveillance.csv ─────────────────────────────────

    def _load_env(self) -> pd.DataFrame:
        path = self.data_dir / "daily_environmental_surveillance.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        col_map = {
            "larval_breteau_index": "breteau_index",
            "adult_ovitrap_index":  "adult_vector_index",
        }
        df.rename(columns=col_map, inplace=True)

        required = {
            "date":              "2026-08-01",
            "day_index":         1,
            "zone_id":           self.zones["zone_id"].iloc[0] if not self.zones.empty else "ZONE_0",
            "rainfall_mm":       0.0,
            "temp_c":            30.0,
            "humidity_pct":      70.0,
            "breteau_index":     20.0,
            "adult_vector_index": 2.0,
        }
        if df.empty:
            df = pd.DataFrame([required])
        else:
            for col, val in required.items():
                if col not in df.columns:
                    logger.warning("Synthesising missing env column '%s'.", col)
                    df[col] = val

        # Ensure day_index is int
        df["day_index"] = df["day_index"].astype(int)
        return df.sort_values(["day_index", "zone_id"]).reset_index(drop=True)

    # ── clinical_reporting_logs.csv ──────────────────────────────────────────

    def _load_clinical(self) -> pd.DataFrame:
        path = self.data_dir / "clinical_reporting_logs.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        rename = {"beds_occupied": "hospital_beds_occupied",
                  "hospital_beds_occupied": "hospital_beds_occupied"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns and k != v},
                  inplace=True)

        required = {
            "date":                    "2026-08-01",
            "day_index":               1,
            "zone_id":                 "ZONE_CENTRAL",
            "confirmed_cases":         0,
            "suspected_fever_cases":   0,
            "citizen_hotline_fever_calls": 0,
            "reporting_status":        "Active",
            "hospital_beds_occupied":  0,
        }
        if df.empty:
            df = pd.DataFrame([required])
        else:
            for col, val in required.items():
                if col not in df.columns:
                    logger.warning("Synthesising missing clinical column '%s'.", col)
                    df[col] = val

        df["day_index"] = df["day_index"].astype(int)
        return df.sort_values(["day_index", "zone_id"]).reset_index(drop=True)

    # ── interventions_and_resources.csv ──────────────────────────────────────

    def _load_interventions(self) -> pd.DataFrame:
        path = self.data_dir / "interventions_and_resources.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        required = {
            "date":                      "2026-08-01",
            "day_index":                 1,
            "zone_id":                   "ZONE_CENTRAL",
            "action_name":               "Monitor_Only",
            "spray_teams_deployed":      0,
            "surveillance_teams_deployed": 0,
            "diagnostic_kits_allocated": 0,
            "operational_cost_inr":      0.0,
            "execution_status":          "Completed",
            "bottleneck_reason":         "",
        }
        if df.empty:
            df = pd.DataFrame([required])
        else:
            for col, val in required.items():
                if col not in df.columns:
                    logger.warning("Synthesising missing intervention column '%s'.", col)
                    df[col] = val

        df["day_index"] = df["day_index"].astype(int)
        return df.sort_values(["day_index", "zone_id"]).reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ────────────────────────────────────────────────────────────────────────

    def _build_graphs(self) -> None:
        """
        Build two NetworkX graphs from the loaded data.

        mobility_graph (DiGraph)
        ────────────────────────
        Nodes : zones only
        Edges : directed, weighted by log-normalised daily commuter flow.
                Gravity model from DengueGNN (Qi et al. 2023):
                    w_ij = log(1 + flow_ij) / max_log_flow
        The direction A→B and B→A both exist for bidirectional roads.

        infra_graph (Graph)
        ───────────────────
        Nodes : zones + facilities (hospitals, labs, depots)
        Edges : undirected, weighted by distance_km (for routing).
                travel_time_min stored as secondary attribute.
        """
        # ── Mobility graph ──────────────────────────────────────────────────
        mob_rows = self.edges_df[
            self.edges_df["edge_type"].str.contains("mobility", case=False, na=False)
        ]

        flows = mob_rows["daily_commuter_flow"].astype(float).clip(lower=1)
        max_log = np.log1p(flows.max()) if not flows.empty else 1.0

        for _, row in self.edges_df.iterrows():
            src, tgt = str(row["source_id"]), str(row["target_id"])
            flow    = float(row.get("daily_commuter_flow", 100_000))
            dist    = float(row.get("distance_km", 10.0))
            t_min   = float(row.get("travel_time_min", 20.0))
            tw      = float(row.get("transmission_weight", 0.1))
            w_mob   = np.log1p(flow) / max_log

            # Add to mobility graph (zone ↔ zone edges only)
            if src.startswith("ZONE_") and tgt.startswith("ZONE_"):
                self.mobility_graph.add_edge(
                    src, tgt,
                    weight=w_mob, flow=flow, distance_km=dist,
                    travel_time_min=t_min, transmission_weight=tw,
                )
                self.mobility_graph.add_edge(
                    tgt, src,
                    weight=w_mob, flow=flow, distance_km=dist,
                    travel_time_min=t_min, transmission_weight=tw,
                )

            # Add to infrastructure graph (all edges)
            self.infra_graph.add_edge(
                src, tgt,
                weight=dist, distance_km=dist,
                travel_time_min=t_min,
            )

        # Ensure every zone is a node even if isolated
        for _, row in self.zones.iterrows():
            zid = row["zone_id"]
            attrs = row.to_dict()
            if zid not in self.mobility_graph:
                self.mobility_graph.add_node(zid, **attrs)
            else:
                nx.set_node_attributes(self.mobility_graph, {zid: attrs})
            if zid not in self.infra_graph:
                self.infra_graph.add_node(zid, **attrs)
            else:
                nx.set_node_attributes(self.infra_graph, {zid: attrs})

        # Ensure every facility is a node in infra_graph
        for _, row in self.infrastructure.iterrows():
            fid = row["facility_id"]
            attrs = row.to_dict()
            if fid not in self.infra_graph:
                self.infra_graph.add_node(fid, **attrs)
            else:
                nx.set_node_attributes(self.infra_graph, {fid: attrs})

            # Synthesise zone→facility edge if absent
            zid = row["zone_id"]
            if not self.infra_graph.has_edge(zid, fid):
                self.infra_graph.add_edge(
                    zid, fid,
                    weight=0.5, distance_km=0.5, travel_time_min=2.0,
                )

        logger.info(
            "Graphs built – mobility: %d nodes / %d edges | "
            "infra: %d nodes / %d edges.",
            self.mobility_graph.number_of_nodes(),
            self.mobility_graph.number_of_edges(),
            self.infra_graph.number_of_nodes(),
            self.infra_graph.number_of_edges(),
        )

    # ────────────────────────────────────────────────────────────────────────
    # Zone snapshot initialisation
    # ────────────────────────────────────────────────────────────────────────

    def _init_zone_snapshots(self) -> None:
        """Populate zone_snapshots from day-1 clinical + env data."""
        for _, zrow in self.zones.iterrows():
            zid = zrow["zone_id"]
            snap = ZoneSnapshot(zone_id=zid, day=1)

            # Clinical day 1
            c = self.clinical_data[
                (self.clinical_data["zone_id"] == zid) &
                (self.clinical_data["day_index"] == 1)
            ]
            if not c.empty:
                snap.confirmed_cases      = float(c["confirmed_cases"].sum())
                snap.beds_occupied        = int(c["hospital_beds_occupied"].sum())

            # Environmental day 1
            e = self.env_data[
                (self.env_data["zone_id"] == zid) &
                (self.env_data["day_index"] == 1)
            ]
            if not e.empty:
                snap.rainfall_mm       = float(e["rainfall_mm"].iloc[0])
                snap.temp_c            = float(e["temp_c"].iloc[0])
                snap.humidity_pct      = float(e["humidity_pct"].iloc[0])
                snap.breteau_index     = float(e["breteau_index"].iloc[0])
                snap.adult_vector_index = float(e["adult_vector_index"].iloc[0])

            # Under-reporting correction (Roy et al. 2026)
            urf = float(zrow.get("under_reporting_factor", 3.0))
            snap.estimated_true_cases = snap.confirmed_cases * urf
            snap.cumulative_cases     = snap.confirmed_cases

            # Kits from infrastructure
            facs = self.infrastructure[self.infrastructure["zone_id"] == zid]
            snap.kits_remaining = int(facs["daily_pcr_kit_capacity"].sum())

            self.zone_snapshots[zid] = snap

    # ────────────────────────────────────────────────────────────────────────
    # Daily advance
    # ────────────────────────────────────────────────────────────────────────

    def advance_day(self, day: int) -> None:
        """
        Advance the simulation clock and update all zone snapshots from the
        historical data for that day.  Modules that run *after* this call
        will see the freshly populated snapshot.

        Complexity: O(Z × F) where Z = number of zones, F = facilities.
        """
        self.current_day = day
        for _, zrow in self.zones.iterrows():
            zid = zrow["zone_id"]
            snap = self.zone_snapshots[zid]
            snap.day = day

            # Clinical data for this day
            c = self.clinical_data[
                (self.clinical_data["zone_id"] == zid) &
                (self.clinical_data["day_index"] == day)
            ]
            if not c.empty:
                snap.confirmed_cases = float(c["confirmed_cases"].sum())
                snap.beds_occupied   = int(c["hospital_beds_occupied"].sum())
                urf = float(zrow.get("under_reporting_factor", 3.0))
                snap.estimated_true_cases = snap.confirmed_cases * urf
                snap.cumulative_cases    += snap.confirmed_cases

            # Environmental data
            e = self.env_data[
                (self.env_data["zone_id"] == zid) &
                (self.env_data["day_index"] == day)
            ]
            if not e.empty:
                snap.rainfall_mm        = float(e["rainfall_mm"].iloc[0])
                snap.temp_c             = float(e["temp_c"].iloc[0])
                snap.humidity_pct       = float(e["humidity_pct"].iloc[0])
                snap.breteau_index      = float(e["breteau_index"].iloc[0])
                snap.adult_vector_index = float(e["adult_vector_index"].iloc[0])

    # ────────────────────────────────────────────────────────────────────────
    # Convenience accessors
    # ────────────────────────────────────────────────────────────────────────

    @property
    def zone_ids(self) -> List[str]:
        return list(self.zone_snapshots.keys())

    def get_snapshot(self, zone_id: str) -> ZoneSnapshot:
        if zone_id not in self.zone_snapshots:
            raise KeyError(f"Unknown zone: {zone_id}")
        return self.zone_snapshots[zone_id]

    def set_risk_score(self, zone_id: str, score: float) -> None:
        """Called by Module 3 Bayesian Network after posterior inference."""
        snap = self.zone_snapshots[zone_id]
        snap.risk_score = score
        if score >= 0.75:
            snap.alert_level = "RED"
        elif score >= 0.50:
            snap.alert_level = "ORANGE"
        elif score >= 0.25:
            snap.alert_level = "YELLOW"
        else:
            snap.alert_level = "GREEN"

    def resource_summary(self) -> pd.DataFrame:
        """Return a per-facility resource table for the current day."""
        rows = []
        for _, frow in self.infrastructure.iterrows():
            rows.append({
                "facility_id":   frow["facility_id"],
                "facility_type": frow["facility_type"],
                "zone_id":       frow["zone_id"],
                "beds_total":    frow["total_bed_capacity"],
                "icu_beds":      frow["icu_bed_capacity"],
                "pcr_kits_day":  frow["daily_pcr_kit_capacity"],
                "spray_teams":   frow["spray_teams_stationed"],
                "surv_teams":    frow["surveillance_teams_stationed"],
                "vehicles":      frow["vehicles_available"],
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise current state to a JSON-compatible dict (for Module 5)."""
        return {
            "day":    self.current_day,
            "zones": {
                zid: {
                    "confirmed_cases":      snap.confirmed_cases,
                    "estimated_true_cases": snap.estimated_true_cases,
                    "risk_score":           snap.risk_score,
                    "alert_level":          snap.alert_level,
                    "breteau_index":        snap.breteau_index,
                    "rainfall_mm":          snap.rainfall_mm,
                    "temp_c":               snap.temp_c,
                    "rl_action":            snap.rl_action,
                    "rl_reward":            snap.rl_reward,
                }
                for zid, snap in self.zone_snapshots.items()
            },
            "plan_steps":      [vars(p) for p in self.plan_steps],
            "routing_results": [vars(r) for r in self.routing_results],
        }
