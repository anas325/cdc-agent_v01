"""Unit tests for pure routing/logic functions in the CDC graph.

No LLM calls, no graph invocation — state dicts are built by hand.
"""

from __future__ import annotations

from src.agents.orchestrator import apply_loop_limits, pick_next_section
from src.graph import route_after_gap_filler, route_after_orchestrator
from src.state import Gap, LoopSettings, PendingQuestion, SectionConfig, SectionStatus


def make_section(id_: str, required: bool = True) -> SectionConfig:
    return SectionConfig(
        id=id_,
        title=id_,
        description="",
        required=required,
        template_slot=id_,
    )


def make_gap(id_: str, status: str = "open", severity: str = "important") -> Gap:
    return Gap(
        id=id_,
        category="functional_ambiguity",
        description=f"gap {id_}",
        severity=severity,
        status=status,
    )


# ---------------------------------------------------------------------------
# route_after_orchestrator
# ---------------------------------------------------------------------------


def test_route_after_orchestrator_done_stops():
    state = {"done": True}
    assert route_after_orchestrator(state) == "end_stopped"


def test_route_after_orchestrator_done_takes_priority_over_pending_questions():
    state = {"done": True, "pending_user_questions": [PendingQuestion(gap_id="g1", text="?")]}
    assert route_after_orchestrator(state) == "end_stopped"


def test_route_after_orchestrator_pending_questions():
    state = {"done": False, "pending_user_questions": [PendingQuestion(gap_id="g1", text="?")]}
    assert route_after_orchestrator(state) == "human_input"


def test_route_after_orchestrator_fresh_mode():
    state = {"done": False, "pending_user_questions": [], "current_mode": "fresh"}
    assert route_after_orchestrator(state) == "gap_finder"


def test_route_after_orchestrator_no_section_goes_to_synthesizer():
    state = {
        "done": False,
        "pending_user_questions": [],
        "current_mode": "section",
        "current_section_id": None,
    }
    assert route_after_orchestrator(state) == "synthesizer"


def test_route_after_orchestrator_otherwise_goes_to_gap_finder():
    state = {
        "done": False,
        "pending_user_questions": [],
        "current_mode": "section",
        "current_section_id": "sec_a",
    }
    assert route_after_orchestrator(state) == "gap_finder"


# ---------------------------------------------------------------------------
# route_after_gap_filler
# ---------------------------------------------------------------------------


def test_route_after_gap_filler_pending_questions():
    state = {"pending_user_questions": [PendingQuestion(gap_id="g1", text="?")]}
    assert route_after_gap_filler(state) == "human_input"


def test_route_after_gap_filler_no_pending_questions():
    state = {"pending_user_questions": []}
    assert route_after_gap_filler(state) == "critic"


# ---------------------------------------------------------------------------
# apply_loop_limits
# ---------------------------------------------------------------------------


def test_apply_loop_limits_below_max_turns_no_op():
    state = {
        "turn": 3,
        "loop_settings": LoopSettings(max_turns=15),
        "gaps": [make_gap("g1", status="open", severity="blocking")],
    }
    result = apply_loop_limits(state)
    assert result.hit_limit is False
    assert result.blocking_stop is False
    assert result.downgraded_gap_ids == []


def test_apply_loop_limits_blocking_gap_at_max_turns_stops():
    state = {
        "turn": 15,
        "loop_settings": LoopSettings(max_turns=15),
        "gaps": [
            make_gap("g1", status="open", severity="blocking"),
            make_gap("g2", status="open", severity="important"),
        ],
    }
    result = apply_loop_limits(state)
    assert result.hit_limit is True
    assert result.blocking_stop is True
    assert "gap g1" in result.stop_message
    assert result.downgraded_gap_ids == []


def test_apply_loop_limits_only_non_blocking_gaps_are_all_deferred():
    state = {
        "turn": 15,
        "loop_settings": LoopSettings(max_turns=15),
        "gaps": [
            make_gap("g1", status="open", severity="important"),
            make_gap("g2", status="open", severity="nice_to_have"),
            make_gap("g3", status="resolved", severity="blocking"),
        ],
    }
    result = apply_loop_limits(state)
    assert result.hit_limit is True
    assert result.blocking_stop is False
    assert sorted(result.downgraded_gap_ids) == ["g1", "g2"]


# ---------------------------------------------------------------------------
# pick_next_section
# ---------------------------------------------------------------------------


def test_pick_next_section_prefers_reopened_over_in_progress_and_empty():
    sections = [make_section("a"), make_section("b"), make_section("c")]
    statuses = {
        "a": SectionStatus(section_id="a", status="empty"),
        "b": SectionStatus(section_id="b", status="in_progress"),
        "c": SectionStatus(section_id="c", status="reopened"),
    }
    state = {"sections_config": sections, "section_statuses": statuses}
    assert pick_next_section(state) == "c"


def test_pick_next_section_prefers_in_progress_over_empty():
    sections = [make_section("a"), make_section("b")]
    statuses = {
        "a": SectionStatus(section_id="a", status="empty"),
        "b": SectionStatus(section_id="b", status="in_progress"),
    }
    state = {"sections_config": sections, "section_statuses": statuses}
    assert pick_next_section(state) == "b"


def test_pick_next_section_falls_back_to_empty_when_nothing_else_pending():
    sections = [make_section("a"), make_section("b")]
    statuses = {
        "a": SectionStatus(section_id="a", status="complete"),
        "b": SectionStatus(section_id="b", status="empty"),
    }
    state = {"sections_config": sections, "section_statuses": statuses}
    assert pick_next_section(state) == "b"


def test_pick_next_section_returns_none_when_all_required_complete():
    sections = [make_section("a"), make_section("b", required=False)]
    statuses = {
        "a": SectionStatus(section_id="a", status="complete"),
        "b": SectionStatus(section_id="b", status="empty"),
    }
    state = {"sections_config": sections, "section_statuses": statuses}
    assert pick_next_section(state) is None


def test_pick_next_section_reopened_can_starve_an_empty_required_section():
    """Documents a real starvation risk, not (yet) a bug in this function.

    pick_next_section scans by status category first (reopened > in_progress
    > empty) across *all* sections, rather than picking whichever required
    section has waited longest. If the critic keeps reopening section "a"
    every turn (e.g. a contradiction it can never fully resolve), section
    "b" — required and still empty — is never selected by this function,
    no matter how many turns pass. There is no fairness/staleness tiebreak.
    """
    sections = [make_section("a"), make_section("b")]
    statuses = {
        "a": SectionStatus(section_id="a", status="reopened", reopen_reason="persistent contradiction"),
        "b": SectionStatus(section_id="b", status="empty"),
    }
    state = {"sections_config": sections, "section_statuses": statuses}

    # Reopened wins every single time, even across many simulated turns,
    # as long as "a" keeps coming back as reopened.
    for _ in range(10):
        assert pick_next_section(state) == "a"
