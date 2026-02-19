from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .schemas import Choice, ProposalEvent, SceneProposal

try:
    from narrativefield.llm.config import PipelineConfig
    from narrativefield.llm.gateway import LLMGateway, ModelTier
except Exception:  # pragma: no cover - only used when live mode is requested
    PipelineConfig = None  # type: ignore[assignment]
    LLMGateway = None  # type: ignore[assignment]
    ModelTier = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MockScene:
    scene_id: str
    location: str
    characters_present: list[str]
    prose: str
    choices: list[Choice]
    events: list[ProposalEvent]


SCENE_CHAIN: dict[str, str] = {
    "scene_0": "scene_1",
    "scene_1": "scene_2",
    "scene_2": "scene_3",
    "scene_3": "scene_4",
    "scene_4": "scene_5",
}


TRANSITION_LINES: dict[str, str] = {
    "s1_balcony": "You slip toward the balcony rail, where the night air sharpens every whisper.",
    "s1_toast": "You remain by the table, close enough to feel every pause in Thorne's smile.",
    "s1_kitchen": "You drift toward the kitchen doorway, where service chatter masks private intent.",
    "s2_press_marcus": "You keep Marcus in your periphery, waiting for his composure to crack.",
    "s2_follow_elena": "You follow Elena's retreat with the care of someone tracking fault lines.",
    "s2_stay_quiet": "You say nothing and let the room incriminate itself.",
    "s3_interrupt": "You step in before the silence can harden into conspiracy.",
    "s3_keep_listening": "You stay still and let another sentence arrive unguarded.",
    "s3_signal_thorne": "You send a small signal toward Thorne and watch who notices.",
    "s4_confront_marcus": "You choose direct pressure and give Marcus no room to perform innocence.",
    "s4_shield_elena": "You redirect the room's gaze before it can settle on Elena.",
    "s4_force_vote": "You force the table into a public decision no one wanted yet.",
    "s5_leave_silently": "You leave the words in the air and let silence do the final damage.",
    "s5_confront_host": "You turn to the hosts and ask for the one truth no toast can conceal.",
    "s5_offer_truce": "You offer a truce not because the room deserves it, but because dawn is coming.",
}


SCENES: dict[str, MockScene] = {
    "scene_1": MockScene(
        scene_id="scene_1",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "The apartment receives everyone in a wash of amber light, the city beyond the glass reduced to a soft"
            " scatter of distant signals. Thorne stands near the table with a host's practiced ease, touching"
            " shoulders, finishing names, keeping the room stitched together by charm and timing.\n\n"
            "Elena moves at his flank with quieter precision, redirecting conversation before it catches on anything"
            " sharp. Marcus arrives laughing half a beat too late. Lydia watches everyone as if she is taking notes"
            " for a report she has not yet decided to file.\n\n"
            "Nothing is wrong. Everything is wrong. The first glasses rise before anyone has said what tonight is for."
        ),
        choices=[
            Choice(id="s1_balcony", label="Go to the balcony", description="Catch Elena away from the table and read her mood."),
            Choice(id="s1_toast", label="Stay for the toast", description="Remain near Thorne and watch who performs for him."),
            Choice(id="s1_kitchen", label="Drift toward the kitchen", description="Listen where private voices mistake noise for cover."),
        ],
        events=[
            ProposalEvent(
                id="s1_evt_01",
                actor_id="thorne",
                action="welcomes the guests and opens the evening",
                event_type="social_move",
                target_ids=["elena", "marcus", "diana", "lydia", "victor"],
                location_id="dining_table",
                pivot_actor_id="thorne",
            )
        ],
    ),
    "scene_2": MockScene(
        scene_id="scene_2",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "Thorne taps a glass and the room obeys. His toast is polished enough to pass for sincerity and old"
            " enough to sound rehearsed: gratitude, continuity, the privilege of loyal company.\n\n"
            "On the word loyal, Elena looks down. Marcus reaches for his drink and misses the stem on the first try."
            " The correction is small, almost elegant, but Lydia sees it and does not forget.\n\n"
            "By the time the applause arrives, the table has split into people who heard a blessing and people who"
            " heard a warning."
        ),
        choices=[
            Choice(id="s2_press_marcus", label="Watch Marcus closely", description="Pressure him with questions wrapped as compliments."),
            Choice(id="s2_follow_elena", label="Track Elena's retreat", description="Follow her glance and see where she seeks safety."),
            Choice(id="s2_stay_quiet", label="Stay silent", description="Let the room expose itself without intervention."),
        ],
        events=[
            ProposalEvent(
                id="s2_evt_01",
                actor_id="thorne",
                action="delivers a formal toast about loyalty",
                event_type="social_move",
                target_ids=["elena", "marcus", "diana", "lydia", "victor"],
                location_id="dining_table",
            ),
            ProposalEvent(
                id="s2_evt_02",
                actor_id="lydia",
                action="notes Marcus's hesitation when loyalty is mentioned",
                event_type="observe",
                target_ids=["marcus"],
                location_id="dining_table",
            ),
        ],
    ),
    "scene_3": MockScene(
        scene_id="scene_3",
        location="kitchen",
        characters_present=["elena", "marcus", "diana"],
        prose=(
            "In the kitchen, the hum of the refrigerator and the clatter of serving spoons make a private weather"
            " system for dangerous sentences. Elena stands at the counter with both hands flat against the stone as"
            " if balance could be negotiated by posture alone.\n\n"
            "Marcus speaks first and too softly for innocence. Diana answers in fragments, each one shaped like a"
            " favor that has already become a debt. Their words keep circling the same absent center.\n\n"
            "From the hall, the dinner still sounds polite. Inside this room, politeness has already expired."
        ),
        choices=[
            Choice(id="s3_interrupt", label="Interrupt directly", description="Break the triangle before they settle on a story."),
            Choice(id="s3_keep_listening", label="Keep listening", description="Stay hidden and let the truth come uninvited."),
            Choice(id="s3_signal_thorne", label="Signal Thorne", description="Bring the host into the room before they are ready."),
        ],
        events=[
            ProposalEvent(
                id="s3_evt_01",
                actor_id="marcus",
                action="presses Diana to keep a financial promise quiet",
                event_type="confide",
                target_ids=["diana"],
                location_id="kitchen",
                required_knowledge_ids=["secret_diana_debt"],
            ),
            ProposalEvent(
                id="s3_evt_02",
                actor_id="elena",
                action="asks for time before anything reaches Thorne",
                event_type="confide",
                target_ids=["marcus"],
                location_id="kitchen",
            ),
        ],
    ),
    "scene_4": MockScene(
        scene_id="scene_4",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "When everyone returns to the table, the arrangement is unchanged but the geometry is not. Chairs"
            " become allegiances. Eye contact becomes a tactical error.\n\n"
            "Thorne asks a harmless question about next quarter and nobody answers with numbers. Marcus answers with"
            " confidence. Elena answers with timing. Lydia answers by saying nothing at all.\n\n"
            "The table has reached that thin moment where one honest sentence could end the evening or finally start it."
        ),
        choices=[
            Choice(id="s4_confront_marcus", label="Confront Marcus", description="Force him to explain the gaps in his story."),
            Choice(id="s4_shield_elena", label="Shield Elena", description="Deflect attention long enough to buy her room."),
            Choice(id="s4_force_vote", label="Force a table vote", description="Make everyone declare where they stand."),
        ],
        events=[
            ProposalEvent(
                id="s4_evt_01",
                actor_id="thorne",
                action="demands a clear account before dessert is served",
                event_type="conflict",
                target_ids=["marcus"],
                location_id="dining_table",
                pivot_actor_id="thorne",
            ),
            ProposalEvent(
                id="s4_evt_02",
                actor_id="marcus",
                action="deflects with a rehearsed explanation",
                event_type="lie",
                target_ids=["thorne", "lydia"],
                location_id="dining_table",
            ),
        ],
    ),
    "scene_5": MockScene(
        scene_id="scene_5",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "The first secret breaks not with shouting but with a name spoken at the wrong speed. Diana says"
            " Marcus's name as if it belongs to a ledger entry, not a friend. Lydia finally looks at Thorne and"
            " does not look away.\n\n"
            "Elena closes her eyes for a single breath. When she opens them, the room has already moved beyond the"
            " point where grace can repair it. Every face at the table is now a witness.\n\n"
            "Outside, the city continues without pause. Inside, this night finally becomes irreversible."
        ),
        choices=[
            Choice(id="s5_leave_silently", label="Leave in silence", description="Walk out and let the truth settle without you."),
            Choice(id="s5_confront_host", label="Confront the hosts", description="Ask for the final truth before anyone can rewrite it."),
            Choice(id="s5_offer_truce", label="Offer a truce", description="Try to stop collapse long enough for terms to exist."),
        ],
        events=[
            ProposalEvent(
                id="s5_evt_01",
                actor_id="diana",
                action="admits the debt tying her to Marcus",
                event_type="reveal",
                target_ids=["thorne", "lydia", "victor", "elena"],
                location_id="dining_table",
                reveals_secret_ids=["secret_diana_debt"],
            ),
            ProposalEvent(
                id="s5_evt_02",
                actor_id="victor",
                action="references Elena and Marcus as established fact",
                event_type="observe",
                target_ids=["thorne"],
                location_id="dining_table",
                required_knowledge_ids=["secret_affair_01"],
            ),
        ],
    ),
}


LIVE_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)
GENERIC_JSON_RE = re.compile(r"(\{.*\})", flags=re.DOTALL)


class SceneGenerator:
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._gateway = None

    async def generate_scene(
        self,
        *,
        engine,
        current_scene_id: str,
        choice_id: str,
        turn: int,
        attempt: int = 1,
    ) -> SceneProposal:
        if self.mode == "live":
            return await self._generate_live_scene(
                engine=engine,
                current_scene_id=current_scene_id,
                choice_id=choice_id,
                turn=turn,
                attempt=attempt,
            )
        return self._generate_mock_scene(
            engine=engine,
            current_scene_id=current_scene_id,
            choice_id=choice_id,
            turn=turn,
        )

    def _generate_mock_scene(self, *, engine, current_scene_id: str, choice_id: str, turn: int) -> SceneProposal:
        if current_scene_id == "scene_5":
            ending_line = TRANSITION_LINES.get(choice_id, "The night ends without consensus, only consequences.")
            prose = f"{ending_line}\n\nThe room does not resolve; it records. Tomorrow will choose what tonight merely exposed."
            return SceneProposal(
                scene_id="scene_5",
                prose=prose,
                location="dining_table",
                characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
                choices=[],
                events=[],
                transition_note="terminal",
            )

        next_scene_id = SCENE_CHAIN.get(current_scene_id, "scene_1")
        scene = SCENES[next_scene_id]

        if current_scene_id == "scene_0":
            prose = scene.prose
        else:
            transition = TRANSITION_LINES.get(choice_id, "You hold your position as the next movement begins.")
            prose = f"{transition}\n\n{scene.prose}"

        return SceneProposal(
            scene_id=scene.scene_id,
            prose=prose,
            location=scene.location,
            characters_present=list(scene.characters_present),
            choices=list(scene.choices),
            events=[event.model_copy(deep=True) for event in scene.events],
            transition_note=choice_id,
        )

    async def _generate_live_scene(
        self,
        *,
        engine,
        current_scene_id: str,
        choice_id: str,
        turn: int,
        attempt: int,
    ) -> SceneProposal:
        if LLMGateway is None or PipelineConfig is None or ModelTier is None:
            raise RuntimeError(
                "Live mode requested but Lorien LLM gateway is unavailable. "
                "Install Lorien engine dependency first."
            )

        if self._gateway is None:
            self._gateway = LLMGateway(config=PipelineConfig())

        world_context = engine.get_world_context()
        default_choices = self._default_choices(engine=engine, turn=turn)
        selected = choice_id if choice_id != "start" else "(start scene)"

        system_prompt = (
            "You are generating one scene for an interactive literary narrative. "
            "First output a strict JSON object, then output prose."
        )
        user_prompt = (
            "World context:\n"
            f"{world_context}\n\n"
            f"Current scene: {current_scene_id}\n"
            f"Selected choice: {selected}\n"
            f"Attempt: {attempt}\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "scene_id": "scene_live_<turn>",\n'
            '  "location": "<location_id>",\n'
            '  "characters_present": ["character_id", "..."],\n'
            '  "events": [\n'
            "    {\n"
            '      "id": "evt_live_1",\n'
            '      "actor_id": "character_id",\n'
            '      "action": "short action",\n'
            '      "event_type": "social_move|reveal|conflict|observe|confide|lie|physical",\n'
            '      "target_ids": ["character_id"],\n'
            '      "location_id": "location_id",\n'
            '      "reveals_secret_ids": [],\n'
            '      "required_knowledge_ids": [],\n'
            '      "regresses_secret_ids": [],\n'
            '      "pivot_actor_id": null\n'
            "    }\n"
            "  ],\n"
            '  "choices": [\n'
            '    {"id": "choice_1", "label": "...", "description": "..."},\n'
            '    {"id": "choice_2", "label": "...", "description": "..."},\n'
            '    {"id": "choice_3", "label": "...", "description": "..."}\n'
            "  ],\n"
            '  "transition_note": "optional"\n'
            "}\n\n"
            "After the JSON object, write prose in 2-3 atmospheric paragraphs in third person present tense."
        )

        raw = await self._gateway.generate(
            tier=ModelTier.CREATIVE,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1400,
        )

        payload = self._extract_json_payload(raw)
        prose = self._extract_prose(raw)

        events = [
            ProposalEvent(
                id=str(item.get("id")) if item.get("id") else None,
                actor_id=str(item.get("actor_id", "")),
                action=str(item.get("action", "")),
                event_type=str(item.get("event_type", "social_move")),
                target_ids=[str(v) for v in item.get("target_ids", [])],
                location_id=str(item.get("location_id")) if item.get("location_id") else None,
                reveals_secret_ids=[str(v) for v in item.get("reveals_secret_ids", [])],
                required_knowledge_ids=[str(v) for v in item.get("required_knowledge_ids", [])],
                regresses_secret_ids=[str(v) for v in item.get("regresses_secret_ids", [])],
                pivot_actor_id=str(item.get("pivot_actor_id")) if item.get("pivot_actor_id") else None,
            )
            for item in payload.get("events", [])
            if isinstance(item, dict)
        ]

        choices = [
            Choice(
                id=str(item.get("id", f"choice_{idx+1}")),
                label=str(item.get("label", f"Choice {idx+1}")),
                description=str(item.get("description", "")),
            )
            for idx, item in enumerate(payload.get("choices", []))
            if isinstance(item, dict)
        ]
        if len(choices) < 3:
            choices = default_choices

        location = str(payload.get("location") or engine.state.current_location)
        characters_present = [
            str(cid)
            for cid in payload.get("characters_present", [])
            if isinstance(cid, str)
        ]
        if not characters_present:
            characters_present = engine.character_ids_at_location(location)

        scene_id = str(payload.get("scene_id") or f"scene_live_{turn}")
        if not prose.strip():
            prose = "The room shifts by degrees rather than declarations, and everyone pretends that is enough."

        return SceneProposal(
            scene_id=scene_id,
            prose=prose,
            location=location,
            characters_present=characters_present,
            choices=choices[:3],
            events=events,
            transition_note=str(payload.get("transition_note") or "live"),
        )

    def safe_fallback_scene(self, *, engine, current_scene_id: str, turn: int) -> SceneProposal:
        location = engine.state.current_location
        return SceneProposal(
            scene_id=current_scene_id if current_scene_id != "scene_0" else "scene_1",
            prose=(
                "For a moment, no one speaks. The conversation thins to a careful quiet while everyone takes stock "
                "of what has already been said."
            ),
            location=location,
            characters_present=engine.character_ids_at_location(location),
            choices=self._default_choices(engine=engine, turn=turn),
            events=[],
            transition_note="fallback",
        )

    def _default_choices(self, *, engine, turn: int) -> list[Choice]:
        current = engine.state.current_location
        options = [loc for loc in sorted(engine.state.locations.keys()) if loc != current]
        options = options[:2] + [current]
        while len(options) < 3:
            options.append(current)
        return [
            Choice(
                id=f"turn{turn}_goto_{loc}_{idx}",
                label=f"Go to {engine.state.locations[loc].name}",
                description=f"Shift perspective toward {engine.state.locations[loc].name.lower()}.",
            )
            for idx, loc in enumerate(options[:3], start=1)
        ]

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any]:
        match = LIVE_JSON_RE.search(raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        generic = GENERIC_JSON_RE.search(raw)
        if generic:
            blob = generic.group(1)
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _extract_prose(raw: str) -> str:
        if "```" in raw:
            parts = re.split(r"```(?:json)?", raw)
            cleaned = [part.strip() for part in parts if part.strip() and "{" not in part]
            if cleaned:
                return "\n\n".join(cleaned)

        marker = "PROSE:"
        if marker in raw:
            return raw.split(marker, 1)[1].strip()

        if "}" in raw:
            tail = raw.rsplit("}", 1)[-1].strip()
            if tail:
                return tail

        return raw.strip()
