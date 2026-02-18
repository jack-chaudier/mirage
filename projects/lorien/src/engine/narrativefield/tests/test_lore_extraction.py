from __future__ import annotations

from narrativefield.storyteller.narrator import _extract_lore_updates


def test_extract_lore_from_full_response() -> None:
    response = """
<prose>Scene text.</prose>
<state_update><summary>Summary</summary></state_update>
<lore_updates>
  <canon_facts>
    <fact event_ids="evt_001,evt_003">Victor inferred the affair was real after the kitchen confession.</fact>
    <fact event_ids="evt_004">Thorne changed tone once Marcus denied the claim.</fact>
  </canon_facts>
  <texture_facts>
    <detail type="gesture" entities="victor">Victor rubs his thumb along the glass rim when stalling.</detail>
    <detail type="setting" entities="dining_table">The chandelier leaves one end of the table in shadow.</detail>
  </texture_facts>
</lore_updates>
"""
    parsed = _extract_lore_updates(response, scene_index=0)
    assert parsed.scene_index == 0
    assert len(parsed.canon_facts) == 2
    assert parsed.canon_facts[0].id == "cf_0_0"
    assert parsed.canon_facts[0].source_event_ids == ["evt_001", "evt_003"]
    assert len(parsed.texture_facts) == 2
    assert parsed.texture_facts[0].id == "tf_0_0"
    assert parsed.texture_facts[0].detail_type == "gesture"
    assert parsed.texture_facts[1].entity_refs == ["dining_table"]


def test_extract_lore_missing_block() -> None:
    parsed = _extract_lore_updates("<prose>No lore block</prose>", scene_index=3)
    assert parsed.scene_index == 3
    assert parsed.canon_facts == []
    assert parsed.texture_facts == []


def test_extract_lore_partial() -> None:
    response = """
<prose>Text</prose>
<lore_updates>
  <canon_facts>
    <fact event_ids="evt_010">A grounded canonical fact.</fact>
  </canon_facts>
</lore_updates>
"""
    parsed = _extract_lore_updates(response, scene_index=1)
    assert len(parsed.canon_facts) == 1
    assert parsed.texture_facts == []


def test_extract_lore_malformed_xml_graceful() -> None:
    response = """
<prose>Text</prose>
<canon_facts>
  <fact event_ids="evt_020">Fact survives even with malformed wrapper.</fact>
</canon_facts>
<texture_facts>
  <detail type="habit" entities="elena">Elena straightens napkins before speaking.</detail>
"""
    parsed = _extract_lore_updates(response, scene_index=2)
    assert len(parsed.canon_facts) == 1
    # Missing </texture_facts> means no parsed detail, but parser must not crash.
    assert parsed.scene_index == 2


def test_detail_type_parsing_values() -> None:
    response = """
<lore_updates>
  <texture_facts>
    <detail type="gesture" entities="a">a</detail>
    <detail type="appearance" entities="a">b</detail>
    <detail type="backstory" entities="a">c</detail>
    <detail type="setting" entities="a">d</detail>
    <detail type="relationship_history" entities="a">e</detail>
    <detail type="habit" entities="a">f</detail>
    <detail type="object" entities="a">g</detail>
  </texture_facts>
</lore_updates>
"""
    parsed = _extract_lore_updates(response, scene_index=4)
    assert [f.detail_type for f in parsed.texture_facts] == [
        "gesture",
        "appearance",
        "backstory",
        "setting",
        "relationship_history",
        "habit",
        "object",
    ]


def test_entity_refs_comma_separated() -> None:
    response = """
<lore_updates>
  <texture_facts>
    <detail type="setting" entities="thorne,lydia,dining_table">Detail.</detail>
  </texture_facts>
</lore_updates>
"""
    parsed = _extract_lore_updates(response, scene_index=7)
    assert parsed.texture_facts[0].entity_refs == ["thorne", "lydia", "dining_table"]
