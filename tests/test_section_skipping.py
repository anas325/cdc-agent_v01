from __future__ import annotations

from src.agents.orchestrator import pick_next_section
from src.state import LoopSettings, SectionConfig, SectionStatus


def test_pick_next_section_skips_sections_marked_as_skipped():
    sections = [
        SectionConfig(id="problem", title="Problem", description="desc", required=True, template_slot="problem"),
        SectionConfig(id="functional", title="Functional", description="desc", required=True, template_slot="functional"),
    ]
    state = {
        "sections_config": sections,
        "section_statuses": {
            "problem": SectionStatus(section_id="problem", status="skipped"),
            "functional": SectionStatus(section_id="functional", status="empty"),
        },
        "loop_settings": LoopSettings(max_turns=5),
    }

    assert pick_next_section(state) == "functional"
