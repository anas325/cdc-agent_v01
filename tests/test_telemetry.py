"""Tests for the timing collector backing the Streamlit debug tab."""

from __future__ import annotations

import pytest

from src import telemetry


@pytest.fixture(autouse=True)
def clean_telemetry():
    telemetry.reset("test-thread")
    yield
    telemetry.reset()


def _llm(seconds: float = 0.0, attempts: int = 1, ok: bool = True, schema: str = "Out"):
    telemetry.record_llm(
        schema=schema,
        model="fake",
        duration_s=seconds,
        attempts=attempts,
        ok=ok,
        prompt_chars=10,
        response_chars=20,
        started_at=0.0,
    )


def test_llm_calls_attribute_to_enclosing_node():
    with telemetry.record_node("gap_finder"):
        _llm(1.0)
        _llm(2.0)
    _llm(5.0)  # outside any node

    runs = telemetry.node_runs()
    assert len(runs) == 1
    assert [c.node for c in runs[0].llm_calls] == ["gap_finder", "gap_finder"]
    assert runs[0].llm_seconds == 3.0

    assert [c.node for c in telemetry.llm_calls()] == ["gap_finder", "gap_finder", "(hors nœud)"]


def test_map_structured_attributes_fanned_out_llm_calls_to_the_node():
    """The thread fan-out must re-bind the enclosing node (via telemetry.bound_to)
    so concurrent LLM calls land on it instead of "(hors nœud)"."""
    from src.llm import map_structured

    def job(seconds: float):
        return lambda: (_llm(seconds), "done")[1]

    with telemetry.record_node("initial_scan"):
        results = map_structured([job(1.0), job(2.0), job(3.0)])

    assert results == ["done", "done", "done"]  # order preserved
    runs = telemetry.node_runs()
    assert len(runs) == 1
    assert [c.node for c in runs[0].llm_calls] == ["initial_scan"] * 3
    assert runs[0].llm_seconds == pytest.approx(6.0)
    # Nothing leaked out to "(hors nœud)".
    assert all(c.node == "initial_scan" for c in telemetry.llm_calls())


def test_node_run_is_recorded_even_when_the_node_raises():
    with pytest.raises(ValueError):
        with telemetry.record_node("critic"):
            raise ValueError("boom")

    runs = telemetry.node_runs()
    assert len(runs) == 1
    assert runs[0].node == "critic"
    assert "ValueError: boom" in runs[0].error


def test_interrupt_is_control_flow_not_an_error():
    class GraphInterrupt(Exception):
        pass

    with pytest.raises(GraphInterrupt):
        with telemetry.record_node("human_input"):
            raise GraphInterrupt()

    assert telemetry.node_runs()[0].error is None


def test_wait_nodes_are_excluded_from_compute_totals():
    with telemetry.record_node("human_input"):
        pass
    with telemetry.record_node("orchestrator"):
        pass

    summary = telemetry.summary()
    assert summary["node_count"] == 1  # human_input not counted as compute
    assert "human_input" not in summary["by_node"]
    assert "orchestrator" in summary["by_node"]

    labels = [r.label for r in telemetry.node_runs()]
    assert labels == ["human_input (attente)", "orchestrator"]


def test_summary_aggregates_counts_retries_and_failures():
    with telemetry.record_node("gap_finder"):
        _llm(1.0, schema="GapFinderOutput")
        _llm(3.0, attempts=2, schema="GapFinderOutput")
    with telemetry.record_node("gap_finder"):
        _llm(2.0, ok=False, schema="GapFinderOutput")
    with telemetry.record_node("critic"):
        _llm(4.0, schema="CriticOutput")

    summary = telemetry.summary()

    assert summary["llm_count"] == 4
    assert summary["retry_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["llm_s"] == pytest.approx(10.0)

    node = summary["by_node"]["gap_finder"]
    assert node["count"] == 2
    assert node["llm_calls"] == 3

    schema = summary["by_schema"]["GapFinderOutput"]
    assert schema["count"] == 3
    assert schema["total_s"] == pytest.approx(6.0)
    assert schema["mean_s"] == pytest.approx(2.0)
    assert schema["max_s"] == pytest.approx(3.0)
    assert schema["retried"] == 1
    assert schema["failed"] == 1
    assert schema["share"] == pytest.approx(0.6)


def test_reset_clears_everything():
    with telemetry.record_node("ingest"):
        _llm(1.0)

    telemetry.reset("other-thread")

    assert telemetry.node_runs() == []
    assert telemetry.llm_calls() == []
    assert telemetry.run_started_at() is None
    assert telemetry.thread_id() == "other-thread"
    assert telemetry.summary()["compute_s"] == 0.0


def test_summary_on_empty_collector_does_not_divide_by_zero():
    summary = telemetry.summary()
    assert summary["llm_share_of_compute"] == 0.0
    assert summary["by_node"] == {}


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0123, "12 ms"), (1.5, "1.50 s"), (59.9, "59.90 s"), (75.4, "1 min 15.4 s")],
)
def test_format_duration(seconds, expected):
    assert telemetry.format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# Integration: telemetry against the real compiled graph, through an interrupt.
# ---------------------------------------------------------------------------


def test_telemetry_records_the_real_graph_across_an_interrupt(monkeypatch):
    """The node wrapper in build_graph must survive interrupt() and resume."""
    from langgraph.types import Command

    from tests.test_graph_flow import (
        ScriptedLLM,
        base_seed,
        make_config,
        make_sections,
    )
    from src.agents.critic import CriticOutput
    from src.agents.gap_filler import QuestionDraft
    from src.agents.gap_finder import GapCandidate, GapFinderOutput
    from src.agents.orchestrator import DedupVerdict
    from src.graph import build_graph
    from src.state import SectionStatus

    fake = ScriptedLLM()
    for target in ("orchestrator", "gap_finder", "gap_filler", "critic"):
        monkeypatch.setattr(f"src.agents.{target}.call_structured", fake)
    monkeypatch.setattr("src.agents.gap_filler.retrieve", lambda query, top_k=None: [])

    fake.add(
        GapFinderOutput,
        GapFinderOutput(
            new_gaps=[
                GapCandidate(
                    section_ids=["sec_a"],
                    category="business_rule",
                    description="Le volume de production maximal n'est pas précisé.",
                    severity="blocking",
                )
            ],
            resolved_gap_ids=[],
        ),
    )
    fake.add(QuestionDraft, QuestionDraft(question_text="Quel volume ?"))
    fake.add(DedupVerdict, DedupVerdict())

    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="empty"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    graph = build_graph()
    config = make_config()
    graph.update_state(config, base_seed(sections, statuses), as_node="initial_scan")

    telemetry.reset("integration")
    result = graph.invoke(None, config)
    assert result.get("__interrupt__"), "expected the run to pause on human_input"

    names = [r.node for r in telemetry.node_runs()]
    assert names[:4] == ["orchestrator", "gap_finder", "gap_filler", "human_input"]
    # interrupt() unwinds through the wrapper — that is control flow, not failure
    assert all(r.error is None for r in telemetry.node_runs())
    # human_input is a wait node: recorded, but kept out of compute aggregates.
    assert [r.label for r in telemetry.node_runs() if r.is_wait] == ["human_input (attente)"]
    assert "human_input" not in telemetry.summary()["by_node"]
    assert telemetry.summary()["by_node"]["gap_finder"]["count"] == 1

    # Resuming re-enters human_input and keeps appending rather than resetting.
    # The tail of the run is open-ended (it loops until max_turns), so swap the
    # scripted queue for a fake that always yields an empty, terminating answer.
    def quiet(prompt, model, llm=None, max_retries=2):
        return {"GapFinderOutput": GapFinderOutput, "CriticOutput": CriticOutput}.get(
            model.__name__, model
        )()

    for target in ("orchestrator", "gap_finder", "gap_filler", "critic"):
        monkeypatch.setattr(f"src.agents.{target}.call_structured", quiet)

    before = len(telemetry.node_runs())
    gap_id = graph.get_state(config).values["gaps"][0].id
    graph.invoke(Command(resume={gap_id: {"text": "5000 t/an", "skip": False}}), config)
    assert len(telemetry.node_runs()) > before
    assert "integrate_answers" in [r.node for r in telemetry.node_runs()]
