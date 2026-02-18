#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import floor
from typing import Any


FORMAT_VERSION = "1.0.0"


def make_delta(
    kind: str,
    agent: str,
    attribute: str,
    op: str,
    value: Any,
    *,
    agent_b: str | None = None,
    reason_code: str = "",
    reason_display: str = "",
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": kind,
        "agent": agent,
        "agent_b": agent_b,
        "attribute": attribute,
        "op": op,
        "value": value,
        "reason_code": reason_code,
        "reason_display": reason_display,
    }
    if agent_b is None:
        d.pop("agent_b")
    return d


def base_components(event_type: str, sim_time: float) -> dict[str, float]:
    # Shared time pressure curve: low early, higher late.
    tp = min((sim_time / 120.0) ** 2, 1.0)

    if event_type == "chat":
        v = dict(
            danger=0.05,
            time_pressure=tp,
            goal_frustration=0.10,
            relationship_volatility=0.10,
            information_gap=0.10,
            resource_scarcity=0.05,
            moral_cost=0.05,
            irony_density=0.10,
        )
    elif event_type == "physical":
        v = dict(
            danger=0.05,
            time_pressure=tp,
            goal_frustration=0.10,
            relationship_volatility=0.05,
            information_gap=0.05,
            resource_scarcity=0.12,
            moral_cost=0.02,
            irony_density=0.05,
        )
    elif event_type == "observe":
        v = dict(
            danger=0.10,
            time_pressure=tp,
            goal_frustration=0.20,
            relationship_volatility=0.05,
            information_gap=0.65,
            resource_scarcity=0.05,
            moral_cost=0.05,
            irony_density=0.65,
        )
    elif event_type == "social_move":
        v = dict(
            danger=0.10,
            time_pressure=tp,
            goal_frustration=0.15,
            relationship_volatility=0.10,
            information_gap=0.20,
            resource_scarcity=0.05,
            moral_cost=0.05,
            irony_density=0.15,
        )
    elif event_type == "confide":
        v = dict(
            danger=0.10,
            time_pressure=tp,
            goal_frustration=0.30,
            relationship_volatility=0.45,
            information_gap=0.70,
            resource_scarcity=0.05,
            moral_cost=0.55,
            irony_density=0.45,
        )
    elif event_type == "lie":
        v = dict(
            danger=0.20,
            time_pressure=tp,
            goal_frustration=0.45,
            relationship_volatility=0.30,
            information_gap=0.80,
            resource_scarcity=0.05,
            moral_cost=0.75,
            irony_density=0.55,
        )
    elif event_type == "reveal":
        v = dict(
            danger=0.40,
            time_pressure=tp,
            goal_frustration=0.50,
            relationship_volatility=0.60,
            information_gap=0.85,
            resource_scarcity=0.05,
            moral_cost=0.65,
            irony_density=0.70,
        )
    elif event_type == "conflict":
        v = dict(
            danger=0.85,
            time_pressure=tp,
            goal_frustration=0.60,
            relationship_volatility=0.85,
            information_gap=0.45,
            resource_scarcity=0.05,
            moral_cost=0.65,
            irony_density=0.50,
        )
    elif event_type == "internal":
        v = dict(
            danger=0.05,
            time_pressure=tp,
            goal_frustration=0.75,
            relationship_volatility=0.10,
            information_gap=0.35,
            resource_scarcity=0.05,
            moral_cost=0.65,
            irony_density=0.55,
        )
    elif event_type == "catastrophe":
        v = dict(
            danger=1.00,
            time_pressure=tp,
            goal_frustration=0.90,
            relationship_volatility=0.95,
            information_gap=0.75,
            resource_scarcity=0.05,
            moral_cost=0.90,
            irony_density=0.85,
        )
    else:
        v = dict(
            danger=0.10,
            time_pressure=tp,
            goal_frustration=0.10,
            relationship_volatility=0.10,
            information_gap=0.10,
            resource_scarcity=0.10,
            moral_cost=0.10,
            irony_density=0.10,
        )

    for k in v:
        v[k] = float(max(0.0, min(1.0, v[k])))
    return v


def scale_components_to_tension(components: dict[str, float], target: float) -> dict[str, float]:
    keys = [
        "danger",
        "time_pressure",
        "goal_frustration",
        "relationship_volatility",
        "information_gap",
        "resource_scarcity",
        "moral_cost",
        "irony_density",
    ]
    mean = sum(components[k] for k in keys) / len(keys)
    if mean <= 1e-9:
        return {k: 0.0 for k in keys}

    factor = target / mean
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(max(0.0, min(1.0, components[k] * factor)))
    return out


def mean_tension(components: dict[str, float]) -> float:
    keys = [
        "danger",
        "time_pressure",
        "goal_frustration",
        "relationship_volatility",
        "information_gap",
        "resource_scarcity",
        "moral_cost",
        "irony_density",
    ]
    return float(sum(components[k] for k in keys) / len(keys))


def eid(n: int) -> str:
    return f"evt_{n:04d}"


def main() -> None:
    agents = [
        {
            "id": "thorne",
            "name": "James Thorne",
            "initial_location": "dining_table",
            "goal_summary": "Maintain status and order; keep the evening smooth; stay loyal.",
            "primary_flaw": "pride",
        },
        {
            "id": "elena",
            "name": "Elena Thorne",
            "initial_location": "dining_table",
            "goal_summary": "Protect secrecy and safety; avoid confrontation; seek autonomy.",
            "primary_flaw": "guilt",
        },
        {
            "id": "marcus",
            "name": "Marcus Webb",
            "initial_location": "dining_table",
            "goal_summary": "Maximize secrecy and safety; deflect suspicion; control the narrative.",
            "primary_flaw": "ambition",
        },
        {
            "id": "lydia",
            "name": "Lydia Cross",
            "initial_location": "dining_table",
            "goal_summary": "Seek truth while staying safe; protect Thorne without triggering conflict.",
            "primary_flaw": "cowardice",
        },
        {
            "id": "diana",
            "name": "Diana Forrest",
            "initial_location": "dining_table",
            "goal_summary": "Balance loyalties; keep secrets contained; avoid financial collapse.",
            "primary_flaw": "guilt",
        },
        {
            "id": "victor",
            "name": "Victor Hale",
            "initial_location": "dining_table",
            "goal_summary": "Pursue the truth about Marcus; probe for evidence; tolerate social discomfort.",
            "primary_flaw": "obsession",
        },
    ]

    locations = [
        {
            "id": "dining_table",
            "name": "Dining Table",
            "privacy": 0.1,
            "capacity": 6,
            "adjacent": ["kitchen", "balcony", "foyer", "bathroom"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "The main dining table. Seats six. The social center of the evening — everyone can see and hear everything here.",
        },
        {
            "id": "kitchen",
            "name": "Kitchen",
            "privacy": 0.5,
            "capacity": 3,
            "adjacent": ["dining_table"],
            "overhear_from": ["dining_table"],
            "overhear_probability": 0.3,
            "description": "An open-plan kitchen adjacent to the dining area. Semi-private — you can slip away under the pretense of getting more wine, but the dining table is within earshot.",
        },
        {
            "id": "balcony",
            "name": "Balcony",
            "privacy": 0.7,
            "capacity": 3,
            "adjacent": ["dining_table"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "A small balcony off the dining room. Private enough for real conversations. Stepping outside signals you want a moment — or a confrontation away from the group.",
        },
        {
            "id": "foyer",
            "name": "Foyer",
            "privacy": 0.2,
            "capacity": 4,
            "adjacent": ["dining_table"],
            "overhear_from": ["dining_table"],
            "overhear_probability": 0.15,
            "description": "The entrance hallway. People pass through here to arrive, leave, or make phone calls. Semi-public — you can catch snippets from the dining room.",
        },
        {
            "id": "bathroom",
            "name": "Bathroom",
            "privacy": 0.9,
            "capacity": 2,
            "adjacent": ["dining_table"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "A small bathroom down the hall from the dining room. The most private space available. Ideal for intense private conversations — or hiding.",
        },
    ]

    secrets = [
        {
            "id": "secret_affair_01",
            "description": "Elena and Marcus are having a romantic affair behind Thorne's back.",
            "truth_value": True,
            "holder": ["elena", "marcus"],
            "about": "elena",
            "content_type": "affair",
            "initial_knowers": ["elena", "marcus", "diana"],
            "initial_suspecters": ["lydia"],
            "dramatic_weight": 1.0,
            "reveal_consequences": "Thorne's marriage and business partnership are both destroyed. The social fabric of the evening tears apart.",
        },
        {
            "id": "secret_embezzle_01",
            "description": "Marcus has been embezzling money from the business he shares with Thorne.",
            "truth_value": True,
            "holder": ["marcus"],
            "about": "marcus",
            "content_type": "financial",
            "initial_knowers": ["marcus"],
            "initial_suspecters": ["lydia", "victor"],
            "dramatic_weight": 0.9,
            "reveal_consequences": "Marcus faces criminal charges. Thorne realizes his business partner betrayed him twice. Victor gets his story.",
        },
        {
            "id": "secret_diana_debt",
            "description": "Diana owes Marcus a large sum of money, making her financially dependent on him.",
            "truth_value": True,
            "holder": ["diana", "marcus"],
            "about": "diana",
            "content_type": "financial",
            "initial_knowers": ["diana", "marcus"],
            "initial_suspecters": [],
            "dramatic_weight": 0.6,
            "reveal_consequences": "Diana's loyalty conflict becomes public. Elena realizes Diana might not be a trustworthy confidante. Marcus has leverage.",
        },
        {
            "id": "secret_lydia_knows",
            "description": "Lydia has noticed financial discrepancies in the firm's books and suspects Marcus of wrongdoing.",
            "truth_value": True,
            "holder": ["lydia"],
            "about": "lydia",
            "content_type": "knowledge",
            "initial_knowers": ["lydia"],
            "initial_suspecters": [],
            "dramatic_weight": 0.5,
            "reveal_consequences": "If Marcus learns Lydia suspects him, she becomes a threat he needs to neutralize. If Thorne learns, he has an ally.",
        },
        {
            "id": "secret_victor_investigation",
            "description": "Victor is secretly investigating Marcus's business dealings for a journalistic expose.",
            "truth_value": True,
            "holder": ["victor"],
            "about": "victor",
            "content_type": "investigation",
            "initial_knowers": ["victor"],
            "initial_suspecters": ["marcus"],
            "dramatic_weight": 0.7,
            "reveal_consequences": "If Marcus confirms Victor is investigating, he'll try to buy Victor off, discredit him, or flee. If Thorne learns, Victor becomes an ally or a threat.",
        },
    ]

    # Fixed tier-1 and anchor events by sequence number.
    fixed: dict[int, dict[str, Any]] = {
        1: dict(
            sim_time=0.0,
            type="chat",
            source="thorne",
            targets=["elena", "marcus", "lydia", "diana", "victor"],
            location="dining_table",
            description="Thorne welcomes the guests and proposes a toast.",
            tension=0.05,
            beat_type="setup",
        ),
        5: dict(
            sim_time=5.0,
            type="observe",
            source="lydia",
            targets=[],
            location="dining_table",
            description="Lydia notices Marcus checking his phone nervously under the table.",
            tension=0.15,
        ),
        12: dict(
            sim_time=15.0,
            type="social_move",
            source="elena",
            targets=[],
            location="kitchen",
            description="Elena excuses herself to check on dessert.",
            tension=0.12,
            content_metadata={"from": "dining_table", "to": "kitchen"},
        ),
        14: dict(
            sim_time=18.0,
            type="confide",
            source="elena",
            targets=["marcus"],
            location="kitchen",
            description="Elena whispers that she fears Thorne suspects something.",
            tension=0.36,
            content_metadata={"secret_id": "secret_affair_01"},
        ),
        16: dict(
            sim_time=20.0,
            type="observe",
            source="lydia",
            targets=[],
            location="kitchen",
            description="Lydia overhears Elena and Marcus whispering in the kitchen.",
            tension=0.35,
            content_metadata={"secret_id": "secret_affair_01", "confidence": 0.75},
        ),
        20: dict(
            sim_time=25.0,
            type="physical",
            source="thorne",
            targets=["elena", "marcus", "lydia", "diana", "victor"],
            location="dining_table",
            description="Thorne refills wine glasses.",
            tension=0.05,
        ),
        22: dict(
            sim_time=30.0,
            type="confide",
            source="elena",
            targets=["diana"],
            location="bathroom",
            description="Elena confides to Diana about the affair, voice shaking.",
            tension=0.40,
            content_metadata={"secret_id": "secret_affair_01"},
        ),
        27: dict(
            sim_time=38.0,
            type="internal",
            source="diana",
            targets=[],
            location="dining_table",
            description="Diana realizes her debt to Marcus traps her no matter what happens tonight.",
            tension=0.32,
            content_metadata={"secret_id": "secret_diana_debt"},
        ),
        31: dict(
            sim_time=42.0,
            type="lie",
            source="marcus",
            targets=["thorne"],
            location="foyer",
            description="Marcus deflects Thorne's quiet question about the firm's finances.",
            tension=0.45,
            content_metadata={"secret_id": "secret_embezzle_01"},
            beat_type="escalation",
        ),
        35: dict(
            sim_time=48.0,
            type="observe",
            source="victor",
            targets=[],
            location="dining_table",
            description="Victor notices hints of a discrepancy in the numbers Marcus just quoted.",
            tension=0.38,
            content_metadata={"secret_id": "secret_embezzle_01"},
        ),
        38: dict(
            sim_time=52.0,
            type="social_move",
            source="thorne",
            targets=[],
            location="balcony",
            description="Thorne steps out to the balcony to get air, jaw tight.",
            tension=0.28,
            content_metadata={"from": "dining_table", "to": "balcony"},
        ),
        41: dict(
            sim_time=55.0,
            type="conflict",
            source="victor",
            targets=["marcus"],
            location="dining_table",
            description="Victor asks a pointed question about the firm's accounts.",
            tension=0.55,
            content_metadata={"secret_id": "secret_embezzle_01"},
        ),
        44: dict(
            sim_time=60.0,
            type="social_move",
            source="marcus",
            targets=[],
            location="balcony",
            description="Marcus follows Thorne to the balcony.",
            tension=0.42,
            content_metadata={"from": "dining_table", "to": "balcony"},
        ),
        47: dict(
            sim_time=65.0,
            type="conflict",
            source="thorne",
            targets=["marcus"],
            location="balcony",
            description="Thorne confronts Marcus about the ledger, voice low and dangerous.",
            tension=0.72,
            content_metadata={"secret_id": "secret_embezzle_01"},
            beat_type="escalation",
        ),
        50: dict(
            sim_time=72.0,
            type="reveal",
            source="lydia",
            targets=["thorne"],
            location="balcony",
            description="Lydia finally tells Thorne what she overheard between Elena and Marcus.",
            tension=0.68,
            content_metadata={"secret_id": "secret_affair_01"},
        ),
        53: dict(
            sim_time=78.0,
            type="observe",
            source="elena",
            targets=[],
            location="dining_table",
            description="Elena sees Lydia on the balcony with Thorne and panics.",
            tension=0.58,
            content_metadata={"secret_id": "secret_affair_01"},
        ),
        56: dict(
            sim_time=85.0,
            type="lie",
            source="marcus",
            targets=["thorne", "elena", "lydia", "diana", "victor"],
            location="dining_table",
            description="Marcus tries to publicly deny everything with a practiced smile.",
            tension=0.62,
            content_metadata={"secret_id": "secret_embezzle_01"},
            beat_type="escalation",
        ),
        60: dict(
            sim_time=92.0,
            type="catastrophe",
            source="elena",
            targets=["thorne", "marcus", "lydia", "diana", "victor"],
            location="dining_table",
            description="Elena breaks down and confesses the affair with Marcus.",
            tension=0.95,
            content_metadata={"secret_id": "secret_affair_01"},
            beat_type="turning_point",
        ),
        63: dict(
            sim_time=98.0,
            type="conflict",
            source="thorne",
            targets=["marcus"],
            location="dining_table",
            description="Thorne turns on Marcus. The betrayal is now double.",
            tension=0.88,
            content_metadata={"secret_id": "secret_embezzle_01"},
            beat_type="consequence",
        ),
        66: dict(
            sim_time=105.0,
            type="social_move",
            source="thorne",
            targets=[],
            location="foyer",
            description="Thorne storms toward the front door.",
            tension=0.70,
            content_metadata={"from": "dining_table", "to": "foyer"},
        ),
        68: dict(
            sim_time=110.0,
            type="internal",
            source="diana",
            targets=[],
            location="dining_table",
            description="Diana realizes she's free of Marcus's leverage now that everything is out.",
            tension=0.35,
            content_metadata={"secret_id": "secret_diana_debt"},
        ),
        70: dict(
            sim_time=120.0,
            type="chat",
            source="victor",
            targets=["lydia"],
            location="foyer",
            description="Victor quietly asks Lydia if she'll go on record when she's ready.",
            tension=0.18,
            content_metadata={"note": "closing_beat"},
        ),
        # Tier-3 ambiguity anchors (non-tier1): one event has no causal links,
        # and one tick has simultaneous conversations.
        29: dict(
            sim_time=40.0,
            type="observe",
            source="victor",
            targets=[],
            location="dining_table",
            description="Victor gets a text from an unknown number: 'Don't trust Marcus.'",
            tension=0.12,
            content_metadata={"external": True},
        ),
        33: dict(
            sim_time=46.0,
            type="chat",
            source="victor",
            targets=["lydia"],
            location="dining_table",
            description="Victor and Lydia trade careful small talk that edges toward money.",
            tension=0.14,
        ),
        34: dict(
            sim_time=46.0,
            type="chat",
            source="elena",
            targets=["thorne"],
            location="dining_table",
            description="Elena overcompensates with tenderness toward Thorne; Marcus watches.",
            tension=0.16,
            content_metadata={"contradiction": "smiles_through_fear"},
        ),
        52: dict(
            sim_time=76.0,
            type="chat",
            source="marcus",
            targets=["diana"],
            location="dining_table",
            description="Marcus jokes too loudly as if nothing is wrong; his hands shake.",
            tension=0.22,
            content_metadata={"contradictory_signal": True},
        ),
    }

    times: dict[int, float] = {k: float(v["sim_time"]) for k, v in fixed.items() if "sim_time" in v}
    assert 1 in times and 70 in times

    fixed_keys = sorted(times.keys())
    for idx in range(len(fixed_keys) - 1):
        a = fixed_keys[idx]
        b = fixed_keys[idx + 1]
        ta = times[a]
        tb = times[b]
        gap = b - a
        if gap <= 1:
            continue
        step = (tb - ta) / gap
        for k in range(a + 1, b):
            if k not in times:
                times[k] = float(ta + step * (k - a))

    for i in range(2, 71):
        if times[i] < times[i - 1] - 1e-9:
            raise RuntimeError(f"time not monotonic at {i}")

    def default_event_fields(n: int) -> dict[str, Any]:
        t = times[n]

        if t < 15:
            etype = "chat" if n % 3 else "physical"
            loc = "dining_table"
            src = ["thorne", "victor", "diana", "marcus", "elena", "lydia"][n % 6]
            tgt = [["elena"], ["diana"], ["victor"], ["thorne"], ["marcus"], ["lydia"]][n % 6]
            tension = 0.02 + 0.01 * (n % 8)
        elif t < 30:
            etype = "chat"
            loc = "dining_table"
            src = ["thorne", "victor", "lydia", "marcus", "diana", "elena"][n % 6]
            tgt = [["victor"], ["lydia"], ["thorne"], ["elena"], ["marcus"], ["diana"]][n % 6]
            tension = 0.06 + 0.02 * (n % 6)
            if n in (13, 15, 19, 21):
                etype = "social_move"
                loc = {13: "kitchen", 15: "kitchen", 19: "bathroom", 21: "dining_table"}[n]
                src = {13: "marcus", 15: "lydia", 19: "diana", 21: "elena"}[n]
                tgt = []
                tension = 0.10 + 0.02 * (n % 5)
            if n in (17, 18):
                etype = "observe"
                src = "lydia"
                tgt = []
                loc = "kitchen" if n == 17 else "dining_table"
                tension = 0.14 if n == 17 else 0.12
        elif t < 52:
            etype = "chat" if n % 4 else "physical"
            loc = "dining_table"
            src = ["victor", "thorne", "marcus", "lydia", "diana", "elena"][n % 6]
            tgt = [["marcus"], ["victor"], ["thorne"], ["victor"], ["elena"], ["diana"]][n % 6]
            tension = 0.10 + 0.02 * (n % 5)
            if n in (23, 24, 25):
                etype = "social_move"
                src = {23: "elena", 24: "marcus", 25: "lydia"}[n]
                tgt = []
                loc = "dining_table"
                tension = 0.12 + 0.01 * (n - 23)
            if n in (26, 28, 30, 32, 36, 37):
                etype = "physical" if n in (26, 36) else "chat"
                tension = 0.14 + 0.02 * (n % 3)
            if n in (28,):
                etype = "chat"
                src = "diana"
                tgt = ["marcus"]
                tension = 0.16
            if n in (30,):
                etype = "chat"
                src = "thorne"
                tgt = ["marcus"]
                tension = 0.22
        elif t < 78:
            etype = "chat" if n % 3 else "physical"
            loc = "dining_table"
            src = ["thorne", "victor", "marcus", "lydia", "diana", "elena"][n % 6]
            tgt = [["victor"], ["marcus"], ["thorne"], ["thorne"], ["elena"], ["diana"]][n % 6]
            tension = 0.18 + 0.02 * (n % 6)
            if n in (39, 40, 42, 43):
                etype = "chat"
                tension = 0.22 + 0.02 * (n % 3)
            if n in (45, 46):
                etype = "chat"
                tension = 0.26 if n == 45 else 0.24
            if n in (48, 49):
                etype = "physical" if n == 48 else "observe"
                tension = 0.20 if n == 48 else 0.28
            if n in (51,):
                etype = "social_move"
                src = "thorne"
                tgt = []
                loc = "dining_table"
                tension = 0.24
        elif t < 95:
            etype = "chat" if n % 3 else "physical"
            loc = "dining_table"
            src = ["marcus", "elena", "victor", "thorne", "diana", "lydia"][n % 6]
            tgt = [["thorne"], ["diana"], ["marcus"], ["marcus"], ["elena"], ["victor"]][n % 6]
            tension = 0.20 + 0.03 * (n % 5)
            if n in (54, 55):
                etype = "social_move" if n == 54 else "chat"
                src = "marcus" if n == 54 else "victor"
                tgt = [] if n == 54 else ["lydia"]
                loc = "dining_table"
                tension = 0.24 if n == 54 else 0.26
            if n in (57, 58, 59):
                etype = "internal" if n == 58 else "observe"
                src = "elena" if n == 58 else "victor"
                tgt = []
                loc = "dining_table"
                tension = 0.28 + 0.02 * (n - 57)
            if n in (61, 62):
                etype = "internal" if n == 61 else "chat"
                src = "thorne" if n == 61 else "diana"
                tgt = [] if n == 61 else ["elena"]
                loc = "dining_table"
                tension = 0.50 if n == 61 else 0.40
        else:
            etype = "chat" if n % 2 else "physical"
            loc = "dining_table" if n < 66 else "foyer"
            src = ["thorne", "victor", "diana", "lydia", "elena", "marcus"][n % 6]
            tgt = [["diana"], ["lydia"], ["victor"], ["thorne"], ["diana"], ["elena"]][n % 6]
            tension = 0.16 + 0.02 * (n % 4)
            if n in (64, 65):
                etype = "chat" if n == 64 else "physical"
                loc = "dining_table"
                src = "victor" if n == 64 else "lydia"
                tgt = ["thorne"] if n == 64 else []
                tension = 0.48 if n == 64 else 0.28
            if n in (67, 69):
                etype = "chat" if n == 69 else "physical"
                src = "victor" if n == 69 else "marcus"
                tgt = ["thorne"] if n == 69 else []
                loc = "foyer" if n == 69 else "dining_table"
                tension = 0.22 if n == 69 else 0.20

        return {
            "sim_time": float(round(t, 3)),
            "type": etype,
            "source": src,
            "targets": tgt,
            "location": loc,
            "description": f"(texture) {etype} #{n}",
            "tension": float(max(0.02, min(0.30, tension))),
        }

    def build_event(n: int, prev_id: str | None) -> dict[str, Any]:
        base = fixed.get(n) or default_event_fields(n)

        sim_time = float(base["sim_time"])
        etype = base["type"]
        src = base["source"]
        targets = list(base.get("targets", []))
        location = base["location"]
        description = base["description"]
        tension_target = float(base["tension"])

        causal_links: list[str] = []
        if n != 1 and n != 29 and prev_id is not None:
            causal_links = [prev_id]

        # Enrich the causal spine so hover cones feel like a story.
        if n == 14:
            causal_links = [eid(12)]
        if n == 16:
            causal_links = [eid(14), eid(15)]
        if n == 22:
            causal_links = [eid(14), eid(16)]
        if n == 27:
            causal_links = [eid(22)]
        if n == 31:
            causal_links = [eid(30)]
        if n == 35:
            causal_links = [eid(31), eid(33)]
        if n == 38:
            causal_links = [eid(35), eid(37)]
        if n == 41:
            causal_links = [eid(35), eid(34)]
        if n == 44:
            causal_links = [eid(41), eid(38)]
        if n == 47:
            causal_links = [eid(44), eid(31)]
        if n == 50:
            causal_links = [eid(47), eid(16)]
        if n == 53:
            causal_links = [eid(50), eid(52)]
        if n == 56:
            causal_links = [eid(47), eid(50)]
        if n == 60:
            causal_links = [eid(53), eid(56), eid(22)]
        if n == 63:
            causal_links = [eid(60)]
        if n == 66:
            causal_links = [eid(63)]
        if n == 68:
            causal_links = [eid(60), eid(66)]

        deltas: list[dict[str, Any]] = []

        def rel(agent: str, other: str, attr: str, change: float) -> None:
            deltas.append(
                make_delta(
                    "relationship",
                    agent=agent,
                    agent_b=other,
                    attribute=attr,
                    op="add",
                    value=float(change),
                    reason_code="REL_DELTA",
                    reason_display="Relationship shift.",
                )
            )

        def belief(agent: str, secret_id: str, state: str, reason_code: str) -> None:
            deltas.append(
                make_delta(
                    "belief",
                    agent=agent,
                    attribute=secret_id,
                    op="set",
                    value=state,
                    reason_code=reason_code,
                    reason_display="Belief updated.",
                )
            )

        def pace(agent: str, attr: str, change: float, reason_code: str) -> None:
            deltas.append(
                make_delta(
                    "pacing",
                    agent=agent,
                    attribute=attr,
                    op="add",
                    value=float(change),
                    reason_code=reason_code,
                    reason_display="Pacing shift.",
                )
            )

        def move(agent: str, to_loc: str, reason_code: str) -> None:
            deltas.append(
                make_delta(
                    "agent_location",
                    agent=agent,
                    attribute="",
                    op="set",
                    value=to_loc,
                    reason_code=reason_code,
                    reason_display=f"Moved to {to_loc}.",
                )
            )

        if etype == "chat" and targets:
            rel(targets[0], src, "affection", 0.02)
        elif etype == "observe":
            if n == 5:
                belief("lydia", "secret_embezzle_01", "suspects", "SUSPICIOUS_BEHAVIOR")
            elif n == 16:
                belief("lydia", "secret_affair_01", "suspects", "OVERHEARD_WHISPER")
            elif n == 35:
                belief("victor", "secret_embezzle_01", "suspects", "NUMBER_MISMATCH")
            elif n == 53:
                pace("elena", "stress", 0.10, "PANIC_RISE")
            else:
                pace(src, "stress", 0.02, "NOTICED_TENSION")
        elif etype == "social_move":
            move(src, location, "MOVE")
        elif etype == "confide" and targets:
            rel(targets[0], src, "trust", 0.05)
            pace(src, "stress", 0.04, "CONFESSION_WEIGHT")
        elif etype == "lie" and targets:
            rel(targets[0], src, "trust", -0.08)
            if n == 31:
                belief("thorne", "secret_embezzle_01", "believes_false", "REASSURED")
        elif etype == "reveal" and targets:
            rel(targets[0], src, "trust", 0.10)
            if n == 50:
                belief("thorne", "secret_affair_01", "suspects", "NEW_INFO")
        elif etype == "conflict" and targets:
            rel(targets[0], src, "trust", -0.25)
            pace(src, "stress", 0.06, "CONFLICT_ESCALATION")
        elif etype == "internal":
            pace(src, "stress", 0.03, "RUMINATION")
        elif etype == "catastrophe":
            for a in ["thorne", "marcus", "lydia", "diana", "victor", "elena"]:
                belief(a, "secret_affair_01", "believes_true", "PUBLIC_CONFESSION")
            pace("elena", "dramatic_budget", -0.80, "BREAKDOWN")

        # Special-case: move both Thorne and Marcus to foyer in one beat.
        if n == 30:
            move("thorne", "foyer", "STEP_ASIDE")
            move("marcus", "foyer", "FOLLOW")

        dialogue: str | None = None
        content_metadata = base.get("content_metadata")
        if n == 52:
            dialogue = "\"Relax. It's just dinner.\""
            pace("marcus", "stress", 0.04, "MASK_SLIP")
        if n == 60:
            dialogue = "\"I can't do this anymore. Marcus and I... it's true.\""

        comps = scale_components_to_tension(base_components(etype, sim_time), tension_target)
        tension = mean_tension(comps)

        irony = float(round(min(5.0, max(0.0, tension * 2.2 + (0.4 if etype in ("lie", "observe") else 0.0))), 3))
        significance = float(round(min(1.0, max(0.0, (tension_target**1.2))), 3))

        irony_collapse = None
        if n == 60:
            irony_collapse = {
                "detected": True,
                "drop": 0.62,
                "collapsed_beliefs": [
                    {"agent": "thorne", "secret": "secret_affair_01", "from": "unknown", "to": "believes_true"},
                    {"agent": "victor", "secret": "secret_affair_01", "from": "unknown", "to": "believes_true"},
                    {"agent": "lydia", "secret": "secret_affair_01", "from": "suspects", "to": "believes_true"},
                ],
                "score": 0.82,
            }

        thematic_shift: dict[str, float] = {}
        if etype == "conflict":
            thematic_shift = {"loyalty_betrayal": -0.15}
        if etype == "catastrophe":
            thematic_shift = {"truth_deception": 0.4, "innocence_corruption": -0.25}

        beat_type = base.get("beat_type")

        event: dict[str, Any] = {
            "id": eid(n),
            "sim_time": sim_time,
            "tick_id": 0,  # assigned later
            "order_in_tick": 0,  # assigned later
            "type": etype,
            "source_agent": src,
            "target_agents": targets,
            "location_id": location,
            "causal_links": causal_links,
            "deltas": deltas,
            "description": description,
            "dialogue": dialogue,
            "content_metadata": content_metadata,
            "beat_type": beat_type,
            "metrics": {
                "tension": float(round(tension, 3)),
                "irony": irony,
                "significance": significance,
                "thematic_shift": thematic_shift,
                "tension_components": comps,
                "irony_collapse": irony_collapse,
            },
        }

        if event["dialogue"] is None:
            event.pop("dialogue")
        if event.get("content_metadata") is None:
            event.pop("content_metadata")
        if event.get("beat_type") is None:
            event.pop("beat_type")
        if event["metrics"]["irony_collapse"] is None:
            event["metrics"]["irony_collapse"] = None

        return event

    events: list[dict[str, Any]] = []
    prev: str | None = None
    for n in range(1, 71):
        e = build_event(n, prev)
        events.append(e)
        prev = e["id"]

    # Assign tick_id and order_in_tick (tick = floor(sim_time)).
    groups: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        tick = int(floor(float(e["sim_time"])))
        e["tick_id"] = tick
        groups.setdefault(tick, []).append(e)

    for evs in groups.values():
        evs.sort(key=lambda x: (x["sim_time"], x["id"]))
        for idx, e in enumerate(evs):
            e["order_in_tick"] = idx

    events.sort(key=lambda x: (x["tick_id"], x["order_in_tick"]))

    if len(events) != 70 or len({e["id"] for e in events}) != 70:
        raise RuntimeError("expected 70 unique events")

    # Coarse scenes matching the visual spec boundaries.
    scene_bounds: list[tuple[str, float, float, str]] = [
        ("scene_001", 0.0, 15.0, "Arrival"),
        ("scene_002", 15.0, 30.0, "Kitchen Meetings"),
        ("scene_003", 30.0, 52.0, "Escalating Suspicion"),
        ("scene_004", 52.0, 78.0, "Balcony Confrontation"),
        ("scene_005", 78.0, 95.0, "The Unraveling"),
        ("scene_006", 95.0, 120.0, "Aftermath"),
    ]

    scenes: list[dict[str, Any]] = []
    for sid, start, end, label in scene_bounds:
        evs = [e for e in events if start <= float(e["sim_time"]) <= end]
        if not evs:
            continue
        event_ids = [e["id"] for e in evs]
        participants = sorted({e["source_agent"] for e in evs} | {t for e in evs for t in e["target_agents"]})

        loc_counts: dict[str, int] = {}
        for e in evs:
            loc_counts[e["location_id"]] = loc_counts.get(e["location_id"], 0) + 1
        primary_loc = max(loc_counts.items(), key=lambda kv: kv[1])[0]

        tension_arc = [float(e["metrics"]["tension"]) for e in evs]
        peak = max(tension_arc) if tension_arc else 0.0
        meanv = sum(tension_arc) / len(tension_arc) if tension_arc else 0.0

        dominant_theme = ""
        scene_type = "maintenance"
        if any(e["type"] == "catastrophe" for e in evs):
            scene_type = "catastrophe"
            dominant_theme = "truth_deception"
        elif any(e["type"] == "conflict" for e in evs):
            scene_type = "confrontation"
            dominant_theme = "loyalty_betrayal"
        elif any(e["type"] == "reveal" for e in evs):
            scene_type = "revelation"
            dominant_theme = "truth_deception"
        elif any(e["type"] == "confide" for e in evs):
            scene_type = "bonding"

        scenes.append(
            {
                "id": sid,
                "event_ids": event_ids,
                "location": primary_loc,
                "participants": participants,
                "time_start": float(evs[0]["sim_time"]),
                "time_end": float(evs[-1]["sim_time"]),
                "tick_start": int(evs[0]["tick_id"]),
                "tick_end": int(evs[-1]["tick_id"]),
                "tension_arc": tension_arc,
                "tension_peak": float(round(peak, 3)),
                "tension_mean": float(round(meanv, 3)),
                "dominant_theme": dominant_theme,
                "scene_type": scene_type,
                "summary": f"{label} ({primary_loc}).",
            }
        )

    # Belief snapshots (coarse) showing evolution.
    initial_beliefs = {
        "thorne": {
            "secret_affair_01": "unknown",
            "secret_embezzle_01": "unknown",
            "secret_diana_debt": "unknown",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "unknown",
        },
        "elena": {
            "secret_affair_01": "believes_true",
            "secret_embezzle_01": "unknown",
            "secret_diana_debt": "suspects",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "unknown",
        },
        "marcus": {
            "secret_affair_01": "believes_true",
            "secret_embezzle_01": "believes_true",
            "secret_diana_debt": "unknown",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "suspects",
        },
        "lydia": {
            "secret_affair_01": "suspects",
            "secret_embezzle_01": "suspects",
            "secret_diana_debt": "unknown",
            "secret_lydia_knows": "believes_true",
            "secret_victor_investigation": "unknown",
        },
        "diana": {
            "secret_affair_01": "believes_true",
            "secret_embezzle_01": "unknown",
            "secret_diana_debt": "believes_true",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "unknown",
        },
        "victor": {
            "secret_affair_01": "unknown",
            "secret_embezzle_01": "suspects",
            "secret_diana_debt": "unknown",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "believes_true",
        },
    }

    mid_beliefs = json.loads(json.dumps(initial_beliefs))
    mid_beliefs["thorne"]["secret_affair_01"] = "suspects"
    mid_beliefs["lydia"]["secret_victor_investigation"] = "suspects"

    post_beliefs = json.loads(json.dumps(mid_beliefs))
    for a in post_beliefs:
        post_beliefs[a]["secret_affair_01"] = "believes_true"
    post_beliefs["thorne"]["secret_embezzle_01"] = "suspects"
    post_beliefs["victor"]["secret_embezzle_01"] = "believes_true"

    close_beliefs = json.loads(json.dumps(post_beliefs))
    close_beliefs["thorne"]["secret_embezzle_01"] = "believes_true"

    snapshot_points = [
        (0, 0.0, initial_beliefs, 1.38),
        (42, 42.0, mid_beliefs, 1.05),
        (78, 78.0, mid_beliefs, 0.92),
        (92, 92.0, post_beliefs, 0.63),
        (110, 110.0, close_beliefs, 0.40),
    ]

    belief_snapshots: list[dict[str, Any]] = []
    for tick, stime, beliefs, scene_irony in snapshot_points:
        belief_snapshots.append(
            {
                "tick_id": tick,
                "sim_time": stime,
                "beliefs": beliefs,
                "agent_irony": {
                    "thorne": 2.5,
                    "elena": 1.25,
                    "marcus": 1.25,
                    "lydia": 0.5,
                    "diana": 1.0,
                    "victor": 0.0,
                },
                "scene_irony": scene_irony,
            }
        )

    metadata = {
        "simulation_id": "fake_dinner_party_001",
        "scenario": "dinner_party",
        "total_ticks": max(e["tick_id"] for e in events) + 1,
        "total_sim_time": 120.0,
        "agent_count": len(agents),
        "event_count": len(events),
        "snapshot_interval": 20,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    payload = {
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "agents": agents,
        "locations": locations,
        "secrets": secrets,
        "events": events,
        "scenes": scenes,
        "belief_snapshots": belief_snapshots,
    }

    with open("data/fake-dinner-party.nf-viz.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


if __name__ == "__main__":
    main()

