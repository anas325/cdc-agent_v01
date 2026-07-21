"""Unit tests for initial_scan_node: one gap_finder pass per section, gaps only.

run_gap_finder is faked so we can assert exactly which sections were scanned
and in what state each call saw.
"""

from __future__ import annotations

import pytest

from src.agents.gap_finder import GapFinderResult
from src.graph import initial_scan_node
from src.state import Gap, SectionConfig, SectionStatus


def make_sections() -> list[SectionConfig]:
    return [
        SectionConfig(id="sec_a", title="A", description="d", required=True, template_slot="a"),
        SectionConfig(id="sec_b", title="B", description="d", required=True, template_slot="b"),
        SectionConfig(id="sec_c", title="C", description="d", required=False, template_slot="c"),
    ]


def make_gap(gid: str, section_id: str, severity: str = "important") -> Gap:
    return Gap(
        id=gid,
        section_ids=[section_id],
        category="business_rule",
        description=f"lacune {gid}",
        severity=severity,
    )


@pytest.fixture
def recorder(monkeypatch):
    """Records (section_id, gaps-seen) per call; returns one gap per section."""
    calls: list[tuple[str, list[str]]] = []

    def fake_run_gap_finder(state, mode, section_id=None, fresh_item_ids=None):
        assert mode == "section"
        calls.append((section_id, [g.id for g in state["gaps"]]))
        return GapFinderResult(
            new_gaps=[make_gap(f"gap_{section_id}", section_id)],
            resolved_gap_ids=[],
            section_complete=True,  # must be ignored by the node
        )

    monkeypatch.setattr("src.agents.gap_finder.run_gap_finder", fake_run_gap_finder)
    return calls


def base_state(statuses: dict[str, SectionStatus] | None = None) -> dict:
    return {
        "sections_config": make_sections(),
        "section_statuses": statuses if statuses is not None else {},
        "gaps": [],
        "context_items": [],
        "turn": 0,
    }


def test_scans_every_section_including_non_required(recorder):
    updates = initial_scan_node(base_state())

    # Sections are scanned concurrently, so call order isn't guaranteed; the
    # merged result order is (zipped back in sections_config order).
    assert {sid for sid, _ in recorder} == {"sec_a", "sec_b", "sec_c"}
    assert [g.id for g in updates["gaps"]] == ["gap_sec_a", "gap_sec_b", "gap_sec_c"]


def test_skipped_sections_are_not_scanned(recorder):
    state = base_state({"sec_b": SectionStatus(section_id="sec_b", status="skipped")})

    updates = initial_scan_node(state)

    assert {sid for sid, _ in recorder} == {"sec_a", "sec_c"}
    assert [g.id for g in updates["gaps"]] == ["gap_sec_a", "gap_sec_c"]


def test_does_not_write_section_statuses(recorder):
    updates = initial_scan_node(base_state())

    assert "section_statuses" not in updates


def test_every_call_sees_the_same_prescan_snapshot(recorder):
    # Parallelized: calls run concurrently, so each sees the same pre-scan gap
    # snapshot (empty here) rather than a rolling accumulation.
    initial_scan_node(base_state())

    assert [seen for _, seen in recorder] == [[], [], []]


def test_preexisting_gaps_are_preserved(recorder):
    state = base_state()
    state["gaps"] = [make_gap("gap_seed", "sec_a")]

    updates = initial_scan_node(state)

    assert updates["gaps"][0].id == "gap_seed"
    # Every parallel call sees the pre-scan snapshot, which includes seed gaps.
    assert all(seen == ["gap_seed"] for _, seen in recorder)


def test_logs_one_entry_per_scanned_section_plus_summary(recorder):
    updates = initial_scan_node(base_state())

    logs = updates["turn_log"]
    assert len(logs) == 4
    assert [entry.details.get("section_id") for entry in logs[:3]] == ["sec_a", "sec_b", "sec_c"]
    assert all(entry.agent == "initial_scan" for entry in logs)
