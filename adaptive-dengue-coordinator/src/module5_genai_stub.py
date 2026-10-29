"""
module5_genai_stub.py – Multi-Agent GenAI Interface Stub
=========================================================
Group 9: Adaptive Dengue Surveillance and Intervention Coordinator

PURPOSE
-------
Clean JSON-to-Prompt interface ready for integration with:
  - Google Gemini API (gemini-1.5-pro / gemini-2.0-flash)
  - OpenAI GPT-4o
  - Anthropic Claude
  - LangChain multi-agent frameworks

INTERFACE CONTRACT
------------------
Input  : CityState.to_dict()  →  structured JSON snapshot of the current
         simulation day including risk scores, alert levels, RL actions,
         plan steps, and routing results.
Output : Structured JSON response with:
  - narrative_summary   : plain-English situation report (SITREP)
  - recommended_actions : list of {zone, action, rationale, priority}
  - escalation_flags    : list of zones requiring immediate human review
  - press_release_draft : public communication draft
  - next_day_forecast   : model-generated prediction

DESIGN
------
This module uses the *Coordinator-Executor* multi-agent pattern:
  - CoordinatorAgent : reads the full city snapshot; delegates tasks.
  - EpidemicAgent    : specialised in case burden and R₀ interpretation.
  - ResourceAgent    : specialised in logistics and routing.
  - CommunicationAgent: generates public-facing messages.

All prompt templates follow the CRISPE framework (Context, Role, Instruction,
Style, Personality, Examples) for reproducible, high-quality outputs.

NOTE: This stub runs without a live API key (returns mock responses) but
is structured so that replacing `_call_llm()` with the actual API call
(one line change per provider) makes it fully functional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Prompt templates (CRISPE framework)
# ════════════════════════════════════════════════════════════════════════════

COORDINATOR_PROMPT = """
Context: You are the AI Coordinator for the Adaptive Dengue Surveillance
system in Visakhapatnam, India. The city has {n_zones} surveillance zones
and {n_red} zones currently on RED alert.

Role: Principal Epidemic Intelligence Officer

Instruction: Analyse the following city state snapshot (day {day}) and
produce a structured JSON response with keys:
  - narrative_summary (str): 3-sentence SITREP
  - escalation_flags (list[str]): zone_ids requiring immediate human review
  - coordination_directive (str): top-priority city-wide action

City Snapshot:
{city_json}

Style: Precise, factual, no hedging. Use dengue epidemiology terminology.
Output ONLY valid JSON.
"""

EPIDEMIC_PROMPT = """
Context: Dengue surveillance system, Visakhapatnam, Day {day}.

Role: Epidemic Intelligence Specialist

Instruction: For each zone in the snapshot, interpret:
  1. Bayesian risk score → epidemic phase (pre-epidemic / epidemic / post-peak)
  2. Breteau Index trend → vector pressure level
  3. Estimated true case burden (under-reporting corrected)
  4. Recommended R0 containment strategy

Zone data:
{zones_json}

Output: JSON list of per-zone assessments.
"""

RESOURCE_PROMPT = """
Context: Resource allocation, Visakhapatnam dengue response, Day {day}.

Role: Logistics and Operations Specialist

Instruction: Given current routing results and CSP allocation, identify:
  1. Which depots have excess capacity (underutilised spray teams)
  2. Which routes have the highest transmission-weighted commuter flow
     (priority for checkpoints / vector surveillance)
  3. Recommended kit redistribution

Routing data: {routing_json}

Output: JSON with keys: surplus_depots, high_risk_routes, kit_redistribution.
"""

COMMUNICATION_PROMPT = """
Context: Public health communication, Visakhapatnam, Day {day}.
{n_red} zones are on RED alert. {n_orange} zones are on ORANGE alert.

Role: Public Health Communication Officer

Instruction: Draft:
  1. A 100-word press release for local media.
  2. A 280-character SMS/WhatsApp alert for citizens.
  3. A school/office advisory (2 bullet points max).

Tone: Calm, informative, actionable. Avoid panic language.
Output: JSON with keys: press_release, sms_alert, advisory.
"""


# ════════════════════════════════════════════════════════════════════════════
# LLM caller (stub – replace body for live API)
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, model: str = "gemini-1.5-pro") -> str:
    """
    Stub LLM caller.  Replace with actual API call:

    # Google Gemini:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_obj = genai.GenerativeModel(model)
    response  = model_obj.generate_content(prompt)
    return response.text

    # OpenAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content
    """
    logger.info("[GenAI STUB] LLM call to %s (no API key – returning mock).", model)
    return json.dumps({
        "stub": True,
        "model": model,
        "prompt_length": len(prompt),
        "message": "Live LLM response available when GEMINI_API_KEY / OPENAI_API_KEY is set.",
    })


# ════════════════════════════════════════════════════════════════════════════
# Agent definitions
# ════════════════════════════════════════════════════════════════════════════

class CoordinatorAgent:
    """Top-level coordinator: reads full city snapshot → delegates sub-tasks."""

    def run(self, city_dict: Dict[str, Any]) -> Dict[str, Any]:
        n_red    = sum(1 for z in city_dict["zones"].values() if z["alert_level"] == "RED")
        n_zones  = len(city_dict["zones"])
        prompt   = COORDINATOR_PROMPT.format(
            n_zones  = n_zones,
            n_red    = n_red,
            day      = city_dict["day"],
            city_json= json.dumps(city_dict, indent=2)[:3000],  # token limit guard
        )
        raw      = _call_llm(prompt, model="gemini-1.5-pro")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}


class EpidemicAgent:
    """Specialised epidemic intelligence; interprets Bayesian posteriors."""

    def run(self, city_dict: Dict[str, Any]) -> Dict[str, Any]:
        prompt = EPIDEMIC_PROMPT.format(
            day       = city_dict["day"],
            zones_json= json.dumps(city_dict["zones"], indent=2)[:2000],
        )
        raw = _call_llm(prompt, model="gemini-1.5-pro")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}


class ResourceAgent:
    """Logistics specialist; interprets routing + CSP results."""

    def run(self, city_dict: Dict[str, Any]) -> Dict[str, Any]:
        prompt = RESOURCE_PROMPT.format(
            day          = city_dict["day"],
            routing_json = json.dumps(city_dict.get("routing_results", [])[:5], indent=2),
        )
        raw = _call_llm(prompt, model="gemini-1.5-pro")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}


class CommunicationAgent:
    """Public communication specialist; drafts press releases + alerts."""

    def run(self, city_dict: Dict[str, Any]) -> Dict[str, Any]:
        zones     = city_dict["zones"]
        n_red     = sum(1 for z in zones.values() if z["alert_level"] == "RED")
        n_orange  = sum(1 for z in zones.values() if z["alert_level"] == "ORANGE")
        prompt    = COMMUNICATION_PROMPT.format(
            day      = city_dict["day"],
            n_red    = n_red,
            n_orange = n_orange,
        )
        raw = _call_llm(prompt, model="gemini-1.5-flash")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ════════════════════════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """
    Coordinates all sub-agents using a sequential pipeline.

    Pipeline:
      CoordinatorAgent → EpidemicAgent → ResourceAgent → CommunicationAgent

    Each agent's output is merged into a single structured response.
    In a live system, EpidemicAgent and ResourceAgent can run in parallel
    (async / threading) since they operate on disjoint data slices.
    """

    def __init__(self) -> None:
        self.coordinator   = CoordinatorAgent()
        self.epidemic      = EpidemicAgent()
        self.resource      = ResourceAgent()
        self.communication = CommunicationAgent()

    def run(self, city_dict: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("MultiAgentOrchestrator: Day %d | %d zones.",
                    city_dict["day"], len(city_dict["zones"]))

        coord_out = self.coordinator.run(city_dict)
        epi_out   = self.epidemic.run(city_dict)
        res_out   = self.resource.run(city_dict)
        comm_out  = self.communication.run(city_dict)

        return {
            "day":           city_dict["day"],
            "coordinator":   coord_out,
            "epidemic":      epi_out,
            "resource":      res_out,
            "communication": comm_out,
        }


# ════════════════════════════════════════════════════════════════════════════
# Module runner
# ════════════════════════════════════════════════════════════════════════════

def run_module5(
    city_dict:  Dict[str, Any],
    output_dir: Path,
    day:        int,
) -> Dict[str, Any]:
    """
    Entry point called by main_demo.py.

    Takes city_dict = CityState.to_dict() and returns the orchestrated
    multi-agent response.  Persists to JSON for the viva log.
    """
    logger.info("=== Module 5 – GenAI Stub | Day %d ===", day)

    orchestrator = MultiAgentOrchestrator()
    result       = orchestrator.run(city_dict)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"module5_genai_day{day:02d}.json").open("w") as f:
        json.dump(result, f, indent=2)

    logger.info("Module 5 stub output written to %s.", log_dir)
    return result
