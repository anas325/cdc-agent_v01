"""The decision log: a machine-readable trace of every AI decision.

Unlike turn_log (a narrative for the UI), these entries must be complete enough
to answer "why did the system decide that?" after the fact — which agent, on
what inputs, with what confidence, under which model and prompt version. It
lives in CDCState behind an operator.add reducer, so it must accumulate across
turns rather than being overwritten.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from src.agents.critic import CriticOutput
from src.agents.final_validator import FinalCheckOutput, write_decision_log
from src.agents.gap_filler import AssumptionDraft, QuestionDraft, RagGrade
from src.agents.gap_finder import GapCandidate, GapFinderOutput
from src.agents.orchestrator import DedupVerdict
from src.agents.synthesizer import SlotDraft
from src.graph import build_graph
from src.state import SectionStatus
from tests.test_graph_flow import ScriptedLLM, base_seed, make_config, make_sections

AGENT_MODULES = ("orchestrator", "gap_finder", "gap_filler", "critic", "synthesizer", "final_validator")


def patch_all_agents(monkeypatch, fake) -> None:
    """Each agent imported call_structured by name, so patch it per module."""
    for module in AGENT_MODULES:
        monkeypatch.setattr(f"src.agents.{module}.call_structured", fake)


def quiet(prompt, model, llm=None, max_retries=2, *, prompt_id=None):
    """A terminating response for every wire model, to run a loop to the end."""
    canned = {
        GapFinderOutput: GapFinderOutput(),
        CriticOutput: CriticOutput(),
        FinalCheckOutput: FinalCheckOutput(),
        DedupVerdict: DedupVerdict(),
        QuestionDraft: QuestionDraft(question_text="?"),
        AssumptionDraft: AssumptionDraft(assumption_text="ASSUMPTION: défaut."),
        RagGrade: RagGrade(sufficient=False),
        SlotDraft: SlotDraft(prose="Texte de section."),
    }
    return canned[model]


@pytest.fixture
def seeded_run(monkeypatch):
    """A run seeded past initial_scan, paused on its first question batch."""
    fake = ScriptedLLM()
    patch_all_agents(monkeypatch, fake)
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
                    confidence=0.8,
                )
            ],
            resolved_gap_ids=[],
        ),
    )
    fake.add(QuestionDraft, QuestionDraft(question_text="Quel volume maximal ?"))
    fake.add(DedupVerdict, DedupVerdict())

    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="empty"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    graph = build_graph()
    config = make_config()
    graph.update_state(config, base_seed(sections, statuses), as_node="initial_scan")
    graph.invoke(None, config)
    return graph, config


def entries(graph, config) -> list:
    return graph.get_state(config).values.get("decision_log", [])


@pytest.fixture
def fake_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.agents.final_validator.load_settings",
        lambda: SimpleNamespace(quarto=SimpleNamespace(output_dir="out")),
    )
    monkeypatch.setattr("src.agents.final_validator.ROOT_DIR", tmp_path)


def test_detection_and_question_decisions_are_recorded(seeded_run):
    graph, config = seeded_run
    log = entries(graph, config)

    assert {"gap_detected", "question_drafted"} <= {e.decision_type for e in log}

    gap_id = graph.get_state(config).values["gaps"][0].id
    detected = next(e for e in log if e.decision_type == "gap_detected")
    assert detected.agent == "gap_finder"
    assert detected.confidence == pytest.approx(0.8)
    assert detected.output_ids == [gap_id]

    drafted = next(e for e in log if e.decision_type == "question_drafted")
    assert drafted.input_ids == [gap_id]
    assert "Quel volume maximal ?" in drafted.summary


def test_every_llm_decision_names_its_model_and_prompt_version(seeded_run):
    """Without both, a decision can't be reproduced or attributed to a change."""
    graph, config = seeded_run

    llm_decisions = [
        e for e in entries(graph, config) if e.decision_type in ("gap_detected", "question_drafted")
    ]
    assert llm_decisions
    for entry in llm_decisions:
        assert entry.model, f"{entry.decision_type} has no model"
        assert entry.prompt_version, f"{entry.decision_type} has no prompt_version"


def test_entries_carry_a_turn_and_a_timestamp(seeded_run):
    graph, config = seeded_run
    for entry in entries(graph, config):
        assert entry.timestamp
        assert entry.turn >= 0


def test_the_log_accumulates_across_turns_instead_of_being_replaced(monkeypatch, seeded_run):
    """The operator.add reducer is the whole point: nothing may be clobbered."""
    graph, config = seeded_run
    before = entries(graph, config)
    assert before

    gap_id = graph.get_state(config).values["gaps"][0].id
    # The rest of the run is open-ended; swap in always-terminating responses.
    patch_all_agents(monkeypatch, quiet)
    graph.invoke(Command(resume={gap_id: {"text": "5000 t/an", "skip": False}}), config)

    after = entries(graph, config)
    assert len(after) > len(before)
    # Every earlier entry is still there, in order.
    assert [e.id for e in after[: len(before)]] == [e.id for e in before]
    assert any(e.decision_type == "answer_integrated" for e in after)


def test_a_user_answer_is_logged_as_an_authoritative_decision(monkeypatch, seeded_run):
    graph, config = seeded_run
    gap_id = graph.get_state(config).values["gaps"][0].id
    patch_all_agents(monkeypatch, quiet)

    graph.invoke(Command(resume={gap_id: {"text": "5000 t/an", "skip": False}}), config)

    integrated = next(e for e in entries(graph, config) if e.decision_type == "answer_integrated")
    assert integrated.agent == "integrate_answers"
    assert integrated.input_ids == [gap_id]
    assert integrated.confidence == 1.0
    # A human answered — no model produced it, so none is attributed.
    assert integrated.model is None


def test_write_decision_log_produces_one_json_object_per_line(seeded_run, fake_output_dir):
    graph, config = seeded_run

    path = write_decision_log(graph.get_state(config).values)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(entries(graph, config))
    parsed = [json.loads(line) for line in lines]
    assert {"id", "timestamp", "turn", "agent", "decision_type", "summary"} <= set(parsed[0])


def test_write_decision_log_on_an_empty_log_writes_an_empty_file(fake_output_dir):
    path = write_decision_log({})

    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
