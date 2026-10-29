"""
app_dashboard.py – Interactive Streamlit Dashboard
====================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

USAGE
-----
    streamlit run app_dashboard.py

FEATURES
--------
- Live simulation control (day slider)
- Bayesian posterior gauge charts
- Alert-level zone map (scatter plot on lat/lon canvas)
- RL reward timeline
- CSP resource allocation viewer
- Routing path viewer
- Downloadable logs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.city_state import CityState
from src.module1_search_csp import run_module1
from src.module2_planning import run_module2
from src.module3_uncertainty import run_module3
from src.module4_rl import run_module4

REPO_ROOT  = Path(__file__).parent
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

ALERT_COLORS = {
    "GREEN": "#2ecc71", "YELLOW": "#f1c40f",
    "ORANGE": "#e67e22", "RED": "#e74c3c",
}


# ════════════════════════════════════════════════════════════════════════════
# State caching
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading CityState…")
def get_city() -> CityState:
    return CityState(DATA_DIR)


@st.cache_data(show_spinner="Running modules…")
def run_day(day: int) -> Dict[str, Any]:
    city = get_city()
    city.advance_day(day)
    m3 = run_module3(city, OUTPUT_DIR, day)
    m1 = run_module1(city, OUTPUT_DIR, day)
    m2 = run_module2(city, OUTPUT_DIR, day, horizon=5)
    m4 = run_module4(city, OUTPUT_DIR, day)
    return {
        "snapshots":  {z: vars(city.get_snapshot(z)) for z in city.zone_ids},
        "csp":        m1["csp_result"],
        "plan":       [vars(p) for p in m2["plan"][:10]],
        "rl_log":     m4["day_log"],
        "bayes":      m3["summary"],
    }


# ════════════════════════════════════════════════════════════════════════════
# Dashboard layout
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Dengue Coordinator Dashboard",
        page_icon="🦟",
        layout="wide",
    )

    st.title("🦟 Adaptive Dengue Surveillance Coordinator – Group 9")
    st.caption("Visakhapatnam, Andhra Pradesh | AI Project 2026")

    city = get_city()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        available_days = sorted(city.clinical_data["day_index"].unique().tolist())
        day = st.slider("Simulation Day", min_value=1,
                        max_value=max(available_days), value=1)
        run_btn = st.button("▶ Run Day")
        st.markdown("---")
        st.subheader("ℹ️ Architecture")
        st.markdown("""
        **Module 1** – UCS/A* + CSP  
        **Module 2** – STRIPS Planner + CPM  
        **Module 3** – Bayesian Network  
        **Module 4** – Q-Learning / SARSA  
        **Module 5** – GenAI Stub  
        """)

    if run_btn or True:  # auto-run on load
        results = run_day(day)

        # ── Alert Overview ──────────────────────────────────────────────────
        st.subheader(f"📍 Zone Alert Status – Day {day}")
        cols = st.columns(len(city.zone_ids))
        for col, zid in zip(cols, city.zone_ids):
            snap = results["snapshots"][zid]
            alert = snap["alert_level"]
            color = ALERT_COLORS.get(alert, "#95a5a6")
            col.markdown(
                f"<div style='background:{color};padding:8px;border-radius:8px;"
                f"text-align:center;color:white;font-weight:bold;'>"
                f"{zid.replace('ZONE_','')}<br>{alert}<br>"
                f"Risk: {snap['risk_score']:.2f}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Two-column layout ───────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            # Bayesian posteriors
            st.subheader("📊 Bayesian Alert Posteriors")
            bayes_df = pd.DataFrame(results["bayes"])
            if not bayes_df.empty:
                bayes_df["zone"] = bayes_df["zone_id"].str.replace("ZONE_", "")
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(bayes_df))
                w = 0.2
                ax.bar(x - 1.5*w, bayes_df["P_GREEN"],  w, label="GREEN",  color="#2ecc71")
                ax.bar(x - 0.5*w, bayes_df["P_YELLOW"], w, label="YELLOW", color="#f1c40f")
                ax.bar(x + 0.5*w, bayes_df["P_ORANGE"], w, label="ORANGE", color="#e67e22")
                ax.bar(x + 1.5*w, bayes_df["P_RED"],    w, label="RED",    color="#e74c3c")
                ax.set_xticks(x)
                ax.set_xticklabels(bayes_df["zone"].tolist(), rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                ax.set_title("P(AlertLevel | Evidence)")
                st.pyplot(fig)
                plt.close(fig)

            # RL rewards
            st.subheader("🎮 RL Policy Rewards")
            rl_df = pd.DataFrame(results["rl_log"])
            if not rl_df.empty:
                reward_cols = [c for c in rl_df.columns if c.startswith("reward_")]
                if reward_cols:
                    disp_df = rl_df[["zone_id"] + reward_cols].copy()
                    disp_df["zone_id"] = disp_df["zone_id"].str.replace("ZONE_", "")
                    disp_df.set_index("zone_id", inplace=True)
                    disp_df.columns = [c.replace("reward_", "") for c in reward_cols]
                    st.bar_chart(disp_df)

        with col_right:
            # CSP allocation
            st.subheader("🔧 CSP Resource Allocation")
            csp = results["csp"]
            if csp.assignment:
                csp_rows = [
                    {"Zone": z.replace("ZONE_", ""),
                     "Spray Teams": v.get("spray", 0),
                     "Surv Teams":  v.get("surv", 0),
                     "Kits":        v.get("kits", 0)}
                    for z, v in csp.assignment.items()
                ]
                st.dataframe(pd.DataFrame(csp_rows), use_container_width=True)
                feasible_color = "🟢" if csp.feasible else "🔴"
                st.caption(f"{feasible_color} Feasible: {csp.feasible}")
                if csp.bottlenecks:
                    with st.expander("⚠️ Bottlenecks"):
                        for b in csp.bottlenecks:
                            st.warning(b)

            # Plan steps
            st.subheader("📋 Today's Plan Steps")
            plan_df = pd.DataFrame(results["plan"])
            if not plan_df.empty:
                cols_show = [c for c in ["day", "zone_id", "action", "cost_inr", "status"] if c in plan_df.columns]
                st.dataframe(plan_df[cols_show], use_container_width=True)

        st.markdown("---")

        # ── Data tables ─────────────────────────────────────────────────────
        with st.expander("📁 Raw Zone Snapshots"):
            snap_df = pd.DataFrame(results["snapshots"]).T
            st.dataframe(snap_df, use_container_width=True)

        # ── Download ─────────────────────────────────────────────────────────
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            snap_csv = pd.DataFrame(results["snapshots"]).T.to_csv()
            st.download_button("⬇️ Download Snapshots CSV", snap_csv,
                               f"snapshots_day{day}.csv", "text/csv")
        with col_dl2:
            rl_json = json.dumps(results["rl_log"], indent=2)
            st.download_button("⬇️ Download RL Log JSON", rl_json,
                               f"rl_day{day}.json", "application/json")

        # Pre-generated plots
        st.markdown("---")
        st.subheader("🖼️ Pre-generated Plots")
        plot_dir = OUTPUT_DIR / "plots"
        plots    = sorted(plot_dir.glob("*.png")) if plot_dir.exists() else []
        if plots:
            n_cols   = 3
            plot_cols = st.columns(n_cols)
            for i, p in enumerate(plots[:9]):  # show up to 9
                plot_cols[i % n_cols].image(str(p), caption=p.name, use_column_width=True)
        else:
            st.info("Run `python main_demo.py` first to generate plots.")


if __name__ == "__main__":
    main()
