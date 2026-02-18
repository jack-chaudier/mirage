from __future__ import annotations

from narrativefield.storyteller.checkpoint import CheckpointManager
from narrativefield.storyteller.types import CharacterState, NarrativeStateObject, NarrativeThread


def _sample_state(*, scene_index: int = 0) -> NarrativeStateObject:
    return NarrativeStateObject(
        summary_so_far="Victor arrives late and scans the room for threats.",
        last_paragraph="He smooths his jacket and forces a smile.",
        current_scene_index=scene_index,
        characters=[
            CharacterState(
                agent_id="victor",
                name="Victor",
                location="dining_table",
                emotional_state="guarded, brittle politeness",
                current_goal="keep control of the conversation",
                knowledge=["Elena is nervous", "Marcus is avoiding eye contact"],
                secrets_revealed=[],
                secrets_held=["He knows about the affair"],
            )
        ],
        active_location="dining_table",
        unresolved_threads=[
            NarrativeThread(
                description="Elena suspects Victor overheard her confession",
                involved_agents=["victor", "elena"],
                tension_level=0.7,
                introduced_at_scene=0,
            )
        ],
        narrative_plan=["Victor will lose composure if pressed again."],
        total_words_generated=250,
        scenes_completed=scene_index,
    )


def test_load_latest_returns_none_when_no_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_id="run-0")
    assert mgr.load_latest() is None
    assert not mgr.has_checkpoint()


def test_save_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_id="run-1")

    state = _sample_state(scene_index=0)
    prose_chunks = ["Scene 0 prose."]
    mgr.save(state=state, prose_chunks=prose_chunks, scene_index=0)

    loaded = mgr.load_latest()
    assert loaded is not None
    loaded_state, loaded_chunks, last_scene = loaded
    assert last_scene == 0
    assert loaded_state == state
    assert loaded_chunks == prose_chunks


def test_resume_loads_last_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_id="run-2")

    s0 = _sample_state(scene_index=0)
    mgr.save(state=s0, prose_chunks=["p0"], scene_index=0)

    s1 = _sample_state(scene_index=1)
    s1.active_location = "balcony"
    mgr.save(state=s1, prose_chunks=["p0", "p1"], scene_index=1)

    s2 = _sample_state(scene_index=2)
    s2.active_location = "kitchen"
    mgr.save(state=s2, prose_chunks=["p0", "p1", "p2"], scene_index=2)

    loaded = mgr.load_latest()
    assert loaded is not None
    loaded_state, loaded_chunks, last_scene = loaded
    assert last_scene == 2
    assert loaded_state.active_location == "kitchen"
    assert loaded_chunks == ["p0", "p1", "p2"]


def test_clear_removes_all_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_id="run-3")
    mgr.save(state=_sample_state(scene_index=0), prose_chunks=["p0"], scene_index=0)
    assert mgr.has_checkpoint()
    mgr.clear()
    assert not mgr.has_checkpoint()
    assert mgr.load_latest() is None


def test_checkpoint_with_unicode_prose(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), run_id="run-4")

    # Unicode: curly apostrophe + accent.
    prose = "Victor’s toast tastes like déjà vu."
    state = _sample_state(scene_index=0)
    state.summary_so_far = "A café hum undercuts the forced laughter."
    mgr.save(state=state, prose_chunks=[prose], scene_index=0)

    loaded = mgr.load_latest()
    assert loaded is not None
    loaded_state, loaded_chunks, last_scene = loaded
    assert last_scene == 0
    assert loaded_chunks == [prose]
    assert loaded_state.summary_so_far == state.summary_so_far


