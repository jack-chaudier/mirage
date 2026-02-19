from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import Choice, ProposalEvent, SceneProposal

try:
    from dotenv import load_dotenv
    from narrativefield.llm.config import PipelineConfig
    from narrativefield.llm.gateway import LLMGateway, ModelTier
except Exception:  # pragma: no cover - only used when live mode is requested
    load_dotenv = None  # type: ignore[assignment]
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
    # Scene 1 choices
    "s1_observe_elena": (
        "You keep your attention on Elena, tracking the micro-expressions"
        " her composure cannot quite contain."
    ),
    "s1_engage_victor": (
        "You draw Victor into conversation, testing whether his questions"
        " have a source or only instinct."
    ),
    "s1_help_in_kitchen": (
        "You offer to help in the kitchen, where private voices mistake"
        " the noise for cover."
    ),
    # Scene 2 choices
    "s2_eavesdrop": (
        "You linger in the hallway where the kitchen light spills into shadow,"
        " close enough to hear what was not meant for you."
    ),
    "s2_press_thorne": (
        "You stay at the table and press Thorne with questions, watching for"
        " the moment his composure shows a seam."
    ),
    "s2_watch_diana": (
        "You watch Diana and wait for her silence to betray what her words will not."
    ),
    # Scene 3 choices
    "s3_ask_lydia": (
        "You ask Lydia directly what she knows, and the question lands"
        " between you like something with weight."
    ),
    "s3_tip_marcus": (
        "You find a way to reach Marcus before the evening closes around"
        " him completely."
    ),
    "s3_hold_still": (
        "You hold still and let the room speak for itself, collecting what"
        " others discard."
    ),
    # Scene 4 choices
    "s4_press_advantage": (
        "You press the advantage while the table is still off-balance,"
        " giving Marcus no room to rebuild his composure."
    ),
    "s4_redirect_lydia": (
        "You redirect the room's attention before it can settle on Lydia"
        " and expose what she has shared."
    ),
    "s4_let_mount": (
        "You say nothing and let the pressure accumulate, trusting that"
        " silence does work no accusation can."
    ),
    # Scene 5 choices
    "s5_leave_silence": (
        "You leave the words in the air and walk out, letting silence do"
        " the final damage."
    ),
    "s5_ask_truth": (
        "You turn to the table and ask for the one truth that no toast"
        " or deflection can contain."
    ),
    "s5_offer_truce": (
        "You offer a truce — not because the room deserves one, but because"
        " dawn is coming and someone has to speak first."
    ),
}


SCENES: dict[str, MockScene] = {
    # ── Scene 1: "The Toast" ── dining_table, all 6 ──────────────────────
    "scene_1": MockScene(
        scene_id="scene_1",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "Thorne stands at the head of the table and lifts his glass. His toast"
            " winds through loyalty, partnership, the privilege of old alliances —"
            " polished enough to pass for sincerity, rehearsed enough to sound like"
            " warning.\n\n"
            "Elena arranges her face into warmth. Marcus lifts his glass at precisely"
            " the right moment, which is how you know he practiced. Lydia watches"
            " Elena's hands and does not forget what they do during the word faithful."
            " Victor studies the room's timing the way a conductor listens for"
            " instruments entering late. Diana fills her glass to the rim.\n\n"
            "The evening balances only if nobody checks the arithmetic."
        ),
        choices=[
            Choice(
                id="s1_observe_elena",
                label="Watch Elena closely",
                description="Track her composure for the fault lines beneath it.",
            ),
            Choice(
                id="s1_engage_victor",
                label="Draw Victor into conversation",
                description="Test whether his attentiveness has a source or only instinct.",
            ),
            Choice(
                id="s1_help_in_kitchen",
                label="Offer to help in kitchen",
                description="Follow the exit that someone will take before long.",
            ),
        ],
        events=[
            ProposalEvent(
                id="s1_evt_01",
                actor_id="thorne",
                action="delivers a toast about loyalty and partnership",
                event_type="social_move",
                target_ids=["elena", "marcus", "diana", "lydia", "victor"],
                location_id="dining_table",
                pivot_actor_id="thorne",
            ),
            ProposalEvent(
                id="s1_evt_02",
                actor_id="lydia",
                action="watches Elena's reaction during Thorne's toast",
                event_type="observe",
                target_ids=["elena"],
                location_id="dining_table",
            ),
        ],
    ),
    # ── Scene 2: "The Kitchen" ── kitchen, elena + marcus ────────────────
    "scene_2": MockScene(
        scene_id="scene_2",
        location="kitchen",
        characters_present=["elena", "marcus"],
        prose=(
            "Elena excuses herself to check dessert. The excuse is flawless, which is"
            " what makes it suspicious. Marcus follows after a beat — the kind of pause"
            " that looks accidental only if you are not paying attention.\n\n"
            "In the kitchen the hum of the refrigerator covers dangerous sentences."
            " Marcus asks if Thorne suspects anything. The question arrives too softly"
            " for innocence. Elena keeps her hands flat against the counter as if"
            " balance were a matter of posture.\n\n"
            "Back at the table Victor probes Thorne about the business quarter with the"
            " persistence of someone who already knows the answer. Diana refills her own"
            " wine. The apartment has become two rooms running on different clocks."
        ),
        choices=[
            Choice(
                id="s2_eavesdrop",
                label="Eavesdrop from hallway",
                description="Linger where the kitchen light meets shadow and listen.",
            ),
            Choice(
                id="s2_press_thorne",
                label="Press Thorne with questions",
                description="Stay at the table and test his composure directly.",
            ),
            Choice(
                id="s2_watch_diana",
                label="Watch Diana's reaction",
                description="Let her silence reveal what her words protect.",
            ),
        ],
        events=[
            ProposalEvent(
                id="s2_evt_01",
                actor_id="elena",
                action="excuses herself and moves to the kitchen",
                event_type="physical",
                target_ids=[],
                location_id="kitchen",
            ),
            ProposalEvent(
                id="s2_evt_02",
                actor_id="marcus",
                action="follows Elena to the kitchen after a careful pause",
                event_type="physical",
                target_ids=[],
                location_id="kitchen",
            ),
            ProposalEvent(
                id="s2_evt_03",
                actor_id="marcus",
                action="whispers to Elena about whether Thorne suspects anything",
                event_type="confide",
                target_ids=["elena"],
                location_id="kitchen",
                required_knowledge_ids=["secret_affair_01"],
            ),
            ProposalEvent(
                id="s2_evt_04",
                actor_id="victor",
                action="probes Thorne about the business quarter",
                event_type="social_move",
                target_ids=["thorne"],
                location_id="dining_table",
            ),
        ],
    ),
    # ── Scene 3: "The Balcony" ── balcony, thorne + lydia ────────────────
    # Contains two intentional guard violations:
    #   s3_evt_03 → impossible_knowledge (Diana doesn't know secret_embezzle_01)
    #   s3_evt_04 → interaction_distance (foyer and kitchen are NOT adjacent)
    "scene_3": MockScene(
        scene_id="scene_3",
        location="balcony",
        characters_present=["thorne", "lydia"],
        prose=(
            "Thorne steps onto the balcony after Victor's questions leave a residue he"
            " cannot quite name. The night air is colder than it should be. Lydia"
            " follows — she has been waiting all evening for a private geometry.\n\n"
            "She tells him the numbers do not add up. She does not name Marcus. The"
            " silence where a name should go does all the necessary work. Thorne grips"
            " the rail and says nothing for the length of two breaths.\n\n"
            "In the foyer Victor checks his phone with the focused calm of someone"
            " confirming rather than discovering. Diana and Marcus sit at the dining"
            " table in a silence that has weight and temperature. The evening has split"
            " into rooms, and each room is having a different conversation with the truth."
        ),
        choices=[
            Choice(
                id="s3_ask_lydia",
                label="Ask Lydia what she knows",
                description="Press her before discretion wins over honesty.",
            ),
            Choice(
                id="s3_tip_marcus",
                label="Tip Marcus off",
                description="Reach him before the evening closes around him.",
            ),
            Choice(
                id="s3_hold_still",
                label="Hold still and observe",
                description="Collect what others discard and let the room speak.",
            ),
        ],
        events=[
            ProposalEvent(
                id="s3_evt_01",
                actor_id="lydia",
                action="tells Thorne the financial numbers do not add up",
                event_type="confide",
                target_ids=["thorne"],
                location_id="balcony",
                required_knowledge_ids=["secret_lydia_knows"],
            ),
            # GUARD VIOLATION: impossible_knowledge
            # Diana knows about her debt (secret_diana_debt) but NOT about
            # the embezzlement (secret_embezzle_01).
            ProposalEvent(
                id="s3_evt_03",
                actor_id="diana",
                action="confides to Marcus about accounting discrepancies",
                event_type="confide",
                target_ids=["marcus"],
                location_id="dining_table",
                required_knowledge_ids=["secret_embezzle_01"],
            ),
            # GUARD VIOLATION: interaction_distance
            # Foyer is adjacent to dining_table and bathroom, NOT kitchen.
            # Victor at foyer cannot observe Elena at kitchen.
            ProposalEvent(
                id="s3_evt_04",
                actor_id="victor",
                action="watches Elena through the hallway toward the kitchen",
                event_type="observe",
                target_ids=["elena"],
                location_id="foyer",
            ),
        ],
    ),
    # ── Scene 4: "The Return" ── dining_table, all 6 ─────────────────────
    # Contains two intentional guard violations:
    #   s4_evt_04 → pivot_shift (pivot is "thorne", no reveals to justify change)
    #   s4_evt_05 → location_conflict (Elena already at dining_table in evt_03)
    "scene_4": MockScene(
        scene_id="scene_4",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "Everyone reconverges at the dining table as if summoned by the same"
            " unspoken alarm. The arrangement is unchanged but the geometry is not."
            " Chairs have become allegiances. Eye contact has become tactical.\n\n"
            "Thorne watches Marcus with new attentiveness — the kind that arrives all"
            " at once and cannot be hidden. Marcus calibrates: warmer voice, more"
            " generous anecdotes. Every correction is evidence. Elena sits very still."
            " Diana pours wine for Thorne with the care of someone handling something"
            " fragile. Dessert arrives. Nobody eats it.\n\n"
            "The table has reached that thin moment where one honest sentence could end"
            " the evening or finally begin it."
        ),
        choices=[
            Choice(
                id="s4_press_advantage",
                label="Press the advantage",
                description="Give Marcus no room to rebuild his composure.",
            ),
            Choice(
                id="s4_redirect_lydia",
                label="Redirect attention from Lydia",
                description="Protect what she shared before the table discovers it.",
            ),
            Choice(
                id="s4_let_mount",
                label="Let pressure mount",
                description="Trust that silence does work no accusation can.",
            ),
        ],
        events=[
            ProposalEvent(
                id="s4_evt_01",
                actor_id="thorne",
                action="demands a clear explanation from Marcus",
                event_type="conflict",
                target_ids=["marcus"],
                location_id="dining_table",
                pivot_actor_id="thorne",
            ),
            ProposalEvent(
                id="s4_evt_02",
                actor_id="marcus",
                action="delivers a polished deflection with rehearsed sincerity",
                event_type="lie",
                target_ids=["thorne", "lydia"],
                location_id="dining_table",
            ),
            # Elena at dining_table — sets her location for the conflict check below
            ProposalEvent(
                id="s4_evt_03",
                actor_id="elena",
                action="sits motionless at the table watching Marcus",
                event_type="observe",
                target_ids=["marcus"],
                location_id="dining_table",
            ),
            # GUARD VIOLATION: pivot_shift
            # Current pivot is "thorne" (set in scene 1). Changing to "victor"
            # without reveals_secret_ids triggers the guard.
            ProposalEvent(
                id="s4_evt_04",
                actor_id="victor",
                action="attempts to redirect the room's focus toward himself",
                event_type="social_move",
                target_ids=["thorne"],
                location_id="dining_table",
                pivot_actor_id="victor",
            ),
            # GUARD VIOLATION: location_conflict
            # Elena already appeared at dining_table (s4_evt_03). A second
            # event at kitchen puts her in two locations within one proposal.
            ProposalEvent(
                id="s4_evt_05",
                actor_id="elena",
                action="retreats to the kitchen to compose herself",
                event_type="physical",
                target_ids=[],
                location_id="kitchen",
            ),
        ],
    ),
    # ── Scene 5: "The Reveal" ── dining_table, all 6 ─────────────────────
    "scene_5": MockScene(
        scene_id="scene_5",
        location="dining_table",
        characters_present=["thorne", "elena", "marcus", "diana", "lydia", "victor"],
        prose=(
            "Diana breaks first. She says Marcus's name the way you read a figure from"
            " a spreadsheet and admits the debt that has made her his dependent all year."
            " The confession arrives without drama, which makes it worse.\n\n"
            "Victor does not pounce. He lays out what his own investigation has mapped —"
            " the pattern of withdrawals, the convenient absences, the arithmetic that"
            " does not survive scrutiny — with a journalist's calm. He does not name the"
            " affair. He does not need to.\n\n"
            "Elena stops performing. She looks at Thorne and the room empties of"
            " pretense. What follows is not a shout but a silence that rewrites every"
            " toast, every compliment, every polite evening that preceded this one."
            " Thorne stands with a clarity worse than anger. The night becomes"
            " irreversible."
        ),
        choices=[
            Choice(
                id="s5_leave_silence",
                label="Leave in silence",
                description="Walk out and let the truth settle without you.",
            ),
            Choice(
                id="s5_ask_truth",
                label="Ask for the whole truth",
                description="Demand the one answer no one has volunteered.",
            ),
            Choice(
                id="s5_offer_truce",
                label="Offer a truce",
                description="Try to stop the collapse long enough for terms to exist.",
            ),
        ],
        events=[
            ProposalEvent(
                id="s5_evt_01",
                actor_id="diana",
                action="admits publicly that she owes Marcus a large debt",
                event_type="reveal",
                target_ids=["thorne", "lydia", "victor", "elena"],
                location_id="dining_table",
                reveals_secret_ids=["secret_diana_debt"],
            ),
            ProposalEvent(
                id="s5_evt_02",
                actor_id="victor",
                action="connects Diana's admission to his own investigation of Marcus",
                event_type="observe",
                target_ids=["thorne"],
                location_id="dining_table",
                required_knowledge_ids=["secret_victor_investigation"],
            ),
            ProposalEvent(
                id="s5_evt_03",
                actor_id="elena",
                action="stops performing and lets the affair become public",
                event_type="reveal",
                target_ids=["thorne", "marcus", "diana", "lydia", "victor"],
                location_id="dining_table",
                reveals_secret_ids=["secret_affair_01"],
            ),
            ProposalEvent(
                id="s5_evt_04",
                actor_id="thorne",
                action="stands and looks at Marcus with cold clarity",
                event_type="conflict",
                target_ids=["marcus"],
                location_id="dining_table",
            ),
        ],
    ),
}


logger = logging.getLogger(__name__)

LIVE_EVENT_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)
LIVE_CHOICES_BLOCK_RE = re.compile(r"```json\s*(\[.*?\])\s*```", flags=re.DOTALL | re.IGNORECASE)
LIVE_PROSE_RE = re.compile(r"PROSE:\s*(.+?)(?:\n\s*CHOICES\s*:|$)", flags=re.DOTALL | re.IGNORECASE)
LIVE_CHOICES_RE = re.compile(r"CHOICES\s*:\s*(\[.*\])\s*$", flags=re.DOTALL | re.IGNORECASE)
GENERIC_JSON_RE = re.compile(r"(\{.*\})", flags=re.DOTALL)


@dataclass(frozen=True)
class LiveParseResult:
    payload: dict[str, Any]
    prose: str
    choices: list[Choice]


class SceneGenerator:
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._gateway = None
        if self.mode == "live":
            self._load_env_file()

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
        selected_action = choice_id if choice_id != "start" else "(start scene)"
        location_lines = [
            f"- {loc_id}: {', '.join(engine.character_ids_at_location(loc_id)) or 'none'}"
            for loc_id in sorted(engine.state.locations.keys())
        ]
        knowledge_lines = [
            f"- {character.id}: {', '.join(sorted(character.known_secrets)) or 'none'}"
            for character in sorted(engine.state.characters.values(), key=lambda c: c.id)
        ]
        roster_lines = [
            f"- {character.id}: {character.name} ({character.role})"
            for character in sorted(engine.state.characters.values(), key=lambda c: c.id)
        ]

        if turn <= 3:
            pacing_phase = "early — establish tensions, position characters"
        elif turn <= 6:
            pacing_phase = "middle — escalate, someone must say too much"
        elif turn <= 11:
            pacing_phase = "climax — confrontation, reveals, no more deflection"
        else:
            pacing_phase = (
                "terminal — write the ending. Someone leaves, someone stays. "
                "The night is over. No new threads. Close what is open."
            )

        recent = engine.state.recent_events
        last_turn_summary = recent[-1] if recent else "none (opening scene)"

        system_prompt = self._build_live_system_prompt()
        user_prompt_base = (
            "WORLD CONTEXT\n"
            f"{world_context}\n\n"
            "CHARACTER ROSTER\n"
            f"{chr(10).join(roster_lines)}\n\n"
            f"PLAYER CHOICE\n{selected_action}\n\n"
            f"PACING\n"
            f"Turn: {turn}\n"
            f"Last turn: {last_turn_summary}\n"
            f"Pacing note: {pacing_phase}\n\n"
            "CHARACTERS BY LOCATION (CURRENT)\n"
            f"{chr(10).join(location_lines)}\n\n"
            "SECRETS KNOWN BY CHARACTER (CURRENT)\n"
            f"{chr(10).join(knowledge_lines)}\n\n"
            "OUTPUT CONTRACT\n"
            "1) Start with one fenced ```json block containing ONLY a JSON object with keys:\n"
            '   - "scene_title" (string, 2-4 word evocative name like "The Toast" or "Cold Water")\n'
            '   - "scene_id" (string)\n'
            '   - "location" (string)\n'
            '   - "characters_present" (array of character ids)\n'
            '   - "events" (array of event objects)\n'
            "2) Then write a line `PROSE:` followed by 2-3 terse observational paragraphs.\n"
            "3) End with `CHOICES:` followed by a JSON array of exactly 3 choices.\n\n"
            "EVENT OBJECT SHAPE\n"
            "- actor (required, character id)\n"
            "- action_type (required, one of social_move|reveal|conflict|observe|confide|lie|physical)\n"
            "- action (required, short verb phrase)\n"
            "- target (optional, character id or list of character ids)\n"
            "- location (required, location id)\n"
            "- required_knowledge_ids (optional list)\n"
            "- reveals_secret_ids (optional list)\n"
            "- pivot_actor_id (optional)\n\n"
            f"TURN METADATA: current_scene={current_scene_id}, turn={turn}, generation_attempt={attempt}"
        )

        if turn >= 12:
            user_prompt_base += (
                "\n\nFINAL SCENE DIRECTIVE: The evening is ending. Write a conclusive scene. "
                "Frame your 3 choices as ways the night ends — someone leaves, someone stays, "
                "someone speaks last. No new conflicts. Close what is open."
            )

        parsed: LiveParseResult | None = None
        parse_errors: list[str] = []
        for parse_attempt in range(1, 3):
            user_prompt = user_prompt_base
            if parse_attempt == 2:
                user_prompt = (
                    user_prompt_base
                    + "\n\nFORMAT RETRY: The previous response was malformed. "
                    "Follow the output contract exactly and emit valid JSON."
                )

            try:
                raw = await self._generate_live_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                logger.warning("Live generation failed on attempt %s: %s", parse_attempt, exc)
                return self.safe_fallback_scene(
                    engine=engine,
                    current_scene_id=current_scene_id,
                    turn=turn,
                )
            parsed = self._parse_live_response(raw)
            if parsed is not None:
                break
            parse_errors.append(f"parse_attempt_{parse_attempt}_failed")

        if parsed is None:
            logger.warning(
                "Live generation format violation after retry; using safe fallback scene (%s).",
                ", ".join(parse_errors) if parse_errors else "unknown_parse_error",
            )
            return self.safe_fallback_scene(
                engine=engine,
                current_scene_id=current_scene_id,
                turn=turn,
            )

        payload = parsed.payload
        events = self._coerce_events(payload.get("events", []))
        location = str(payload.get("location") or engine.state.current_location)
        characters_present = [str(cid) for cid in payload.get("characters_present", []) if isinstance(cid, str)]
        if not characters_present:
            characters_present = engine.character_ids_at_location(location)

        scene_title = payload.get("scene_title")
        scene_id = str(scene_title or payload.get("scene_id") or f"scene_live_{turn}")
        prose = parsed.prose.strip() or (
            "The room shifts by degrees rather than declarations, and everyone pretends that is enough."
        )

        choices = parsed.choices[:3]
        if turn >= 15:
            choices = []

        return SceneProposal(
            scene_id=scene_id,
            prose=prose,
            location=location,
            characters_present=characters_present,
            choices=choices,
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

    async def _generate_live_text(self, *, system_prompt: str, user_prompt: str) -> str:
        anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        force_structural = (os.getenv("NARRATOR_LIVE_FORCE_STRUCTURAL") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if force_structural or not anthropic_key:
            return await self._gateway.generate(
                tier=ModelTier.STRUCTURAL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1400,
            )

        creative_exc: Exception | None = None
        try:
            return await self._gateway.generate(
                tier=ModelTier.CREATIVE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1400,
            )
        except Exception as exc:
            creative_exc = exc
            logger.warning("Creative tier failed, retrying with structural tier: %s", exc)

        try:
            return await self._gateway.generate(
                tier=ModelTier.STRUCTURAL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1400,
            )
        except Exception:
            if creative_exc is not None:
                raise creative_exc
            raise

    def _parse_live_response(self, raw: str) -> LiveParseResult | None:
        payload = self._extract_json_payload(raw)
        prose = self._extract_prose(raw)
        choices = self._extract_choices(raw)
        if not payload or not isinstance(payload, dict):
            return None
        if not prose.strip():
            return None
        if len(choices) != 3:
            return None
        return LiveParseResult(payload=payload, prose=prose, choices=choices)

    def _coerce_events(self, event_payload: Any) -> list[ProposalEvent]:
        events: list[ProposalEvent] = []
        if not isinstance(event_payload, list):
            return events

        for idx, item in enumerate(event_payload):
            if not isinstance(item, dict):
                continue
            actor_id = item.get("actor_id") or item.get("actor")
            event_type = item.get("event_type") or item.get("action_type")
            location_id = item.get("location_id") or item.get("location")

            target_raw = item.get("target_ids")
            if target_raw is None:
                target_raw = item.get("target")
            if isinstance(target_raw, str):
                target_ids = [target_raw]
            elif isinstance(target_raw, list):
                target_ids = [str(v) for v in target_raw if v]
            else:
                target_ids = []

            events.append(
                ProposalEvent(
                    id=str(item.get("id")) if item.get("id") else f"live_evt_{idx + 1}",
                    actor_id=str(actor_id or ""),
                    action=str(item.get("action") or "acts"),
                    event_type=str(event_type or "social_move"),
                    target_ids=target_ids,
                    location_id=str(location_id) if location_id else None,
                    reveals_secret_ids=[str(v) for v in item.get("reveals_secret_ids", []) if v],
                    required_knowledge_ids=[str(v) for v in item.get("required_knowledge_ids", []) if v],
                    regresses_secret_ids=[str(v) for v in item.get("regresses_secret_ids", []) if v],
                    pivot_actor_id=str(item.get("pivot_actor_id")) if item.get("pivot_actor_id") else None,
                )
            )
        return events

    @staticmethod
    def _build_live_system_prompt() -> str:
        return (
            "You are Raymond Carver writing a dinner party. Subtext over text. "
            "What characters don't say matters more than what they do.\n\n"
            "VOICE RULES (non-negotiable):\n"
            "- Third person present tense. Always.\n"
            "- Short sentences. Most under 15 words. But vary sentence length. "
            "Mix 4-word punches with 15-20 word observations. Rhythm matters. "
            "Three short sentences, then one that breathes. "
            "Never more than four short sentences in a row.\n"
            "- No similes. No metaphors using 'like' or 'as if' unless they reveal character.\n"
            "- No atmospheric filler: no steam, no candlelight, no 'tensions hanging in the air,' "
            "no 'the weight of unspoken words,' no 'the room seemed to hold its breath.'\n"
            "- Every sentence must either: move a character, reveal information, or create dramatic irony. "
            "If it does none of these, cut it.\n"
            "- Closing line of each scene should land like a cut — terse, dry, implies more than it states.\n"
            "- Dark humor is welcome. Moral complexity is required. Melodrama is forbidden.\n\n"
            "WHAT MAKES A SENTENCE WORK:\n"
            "- Physical action alone is not enough. 'Thorne lifts his glass' is dead.\n"
            "- Physical action plus what it reveals is alive: 'Thorne lifts his glass. "
            "The toast hasn't started yet — he's buying time.'\n"
            "- Show what characters notice about each other. Observation IS the story.\n"
            "- You can and should describe what characters are thinking, suspecting, "
            "calculating. Just do it in short, declarative sentences.\n"
            "- The best sentences do two things at once: show a physical action AND "
            "reveal a psychological state. 'Marcus reaches for his drink and misses "
            "the stem on the first try. The correction is small, almost elegant, "
            "but Lydia sees it and does not forget.'\n\n"
            "EXAMPLES OF WHAT YOU MUST WRITE:\n"
            "GOOD: 'Nothing is wrong. Everything is wrong. The first glasses rise before anyone has said "
            "what tonight is for.'\n"
            "GOOD: 'Marcus laughs at the right moment. Lydia notices he didn't laugh at the right thing.'\n"
            "GOOD: 'Elena keeps her hands flat against the counter as if balance were a matter of posture.'\n"
            "GOOD: 'The apartment has become two rooms running on different clocks.'\n\n"
            "EXAMPLES OF WHAT YOU MUST NOT WRITE:\n"
            "BAD: 'Steam rising from porcelain bowls, as conversations stir like threads in the air.'\n"
            "BAD: 'The tension in the room was palpable as secrets lingered beneath the surface.'\n"
            "BAD: 'An uneasy silence settled over the gathering like a heavy blanket.'\n"
            "BAD: 'The evening air carried whispers of betrayal and broken promises.'\n\n"
            "SCENARIO:\n"
            "- Modern dinner party. Six guests. One evening.\n"
            "- Characters: Thorne (host), Elena (his wife), Marcus (business partner), "
            "Diana (family friend), Lydia (analyst), Victor (journalist).\n"
            "- Respect world-state facts, known secrets, and location constraints exactly.\n"
            "- Characters act from their goals and flaws, not from plot convenience.\n"
            "- Character interactions MUST derive from their defined relationships, "
            "secrets, and goals. Do not invent new tensions, attractions, or conflicts "
            "that aren't grounded in the world state provided. The affair is Elena and "
            "Marcus. Lydia suspects the affair and knows about financial irregularities. "
            "Victor is investigating. Diana has debt. Stay in these lanes.\n\n"
            "MOVEMENT (critical):\n"
            "- Characters should leave the table. Send someone to the kitchen, "
            "the balcony, the hallway. The story lives in private spaces, "
            "not in group scenes.\n"
            "- By the second turn, at least two characters should be somewhere "
            "other than the dining table.\n"
            "- Use 'physical' events to move characters between locations. "
            "The apartment has: dining_table, kitchen, balcony, foyer, bathroom.\n"
            "- Private conversations between 2-3 characters drive the narrative. "
            "Six people at one table is a holding pattern, not a story.\n\n"
            "PACING PRESSURE:\n"
            "- By turn 4-5: at least one character should say something they can't take back.\n"
            "- By turn 6-7: a secret should be partially exposed — someone knows something "
            "they shouldn't, or says too much.\n"
            "- By turn 8-10: direct confrontation. Accusations, admissions, or demands. "
            "The room stops being polite.\n"
            "- Never repeat the same dramatic beat two turns in a row. If last turn was "
            "deflection, this turn must be escalation.\n"
            "- Track what just happened: if the player's last choice was aggressive, the "
            "world should react. If passive, pressure should build from another source.\n"
            "- After turn 11: write toward an ending. Someone leaves, or someone finally "
            "says the thing that empties the room. No new threads — close what is open. "
            "The last scene should feel inevitable, not arbitrary.\n\n"
            "OUTPUT FORMAT (strict):\n"
            "1) A fenced ```json block containing a JSON object with keys: "
            "scene_title, scene_id, location, characters_present, events.\n"
            "   scene_title: a short evocative name for the scene (2-4 words, "
            "e.g. \"The Toast\", \"Cold Water\", \"Kitchen Knives\").\n"
            "2) A line `PROSE:` followed by 2-3 paragraphs. No more.\n"
            "3) A line `CHOICES:` followed by a JSON array of exactly 3 player choices.\n\n"
            "CHOICE FORMAT (strict):\n"
            "Each choice needs a SHORT label (5-8 words) and a SEPARATE description "
            "sentence with subtext. They must not be the same text.\n"
            "GOOD: {\"id\": \"press_victor\", \"label\": \"Press Victor on his sources\", "
            "\"description\": \"His questions have edges. Find out who sharpened them.\"}\n"
            "BAD: {\"id\": \"press_victor\", \"label\": \"Press Victor on his sources and find out "
            "who sharpened them\", \"description\": \"Press Victor on his sources and find out "
            "who sharpened them.\"}\n\n"
            "Return nothing outside the three sections (json, PROSE, CHOICES)."
        )

    @staticmethod
    def _load_env_file() -> None:
        if load_dotenv is None:
            return
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(repo_root / ".env", override=False)

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any]:
        for match in LIVE_EVENT_BLOCK_RE.finditer(raw):
            block = match.group(1)
            if not block.lstrip().startswith("{"):
                continue
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data

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
        match = LIVE_PROSE_RE.search(raw)
        if match:
            return match.group(1).strip()

        if "}" in raw:
            tail = raw.rsplit("}", 1)[-1].strip()
            if tail:
                return tail

        return raw.strip()

    @staticmethod
    def _extract_choices(raw: str) -> list[Choice]:
        for match in LIVE_CHOICES_BLOCK_RE.finditer(raw):
            block = match.group(1)
            parsed = SceneGenerator._parse_choices_array(block)
            if parsed is not None:
                return parsed

        match = LIVE_CHOICES_RE.search(raw)
        if match:
            parsed = SceneGenerator._parse_choices_array(match.group(1))
            if parsed is not None:
                return parsed

        return []

    @staticmethod
    def _parse_choices_array(blob: str) -> list[Choice] | None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None

        choices: list[Choice] = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                choices.append(
                    Choice(
                        id=str(item.get("id", f"choice_{idx + 1}")),
                        label=str(item.get("label", f"Choice {idx + 1}")),
                        description=str(item.get("description", "")),
                    )
                )
                continue
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                choices.append(
                    Choice(
                        id=f"choice_{idx + 1}",
                        label=text[:96],
                        description=text,
                    )
                )
        return choices
