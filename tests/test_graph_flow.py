"""Integration tests that run the actual compiled graph from build_graph().

The LLM (src.llm.call_structured) and RAG retrieval (src.rag.retrieve) are the
only things faked — everything else (node functions, conditional edges,
checkpointing, interrupts) is the real graph.

Each test seeds a full CDCState directly onto the checkpoint via
`graph.update_state(config, seed, as_node="initial_scan")` and then resumes
execution from there. This lets us start mid-pipeline (e.g. with a section
already "complete") without having to script an entire multi-turn run just
to get there.
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from src.agents.critic import CriticOutput
from src.agents.gap_filler import QuestionDraft
from src.agents.gap_finder import GapCandidate, GapFinderOutput
from src.agents.orchestrator import DedupVerdict
from src.agents.critic import ContradictionFinding
from src.graph import build_graph
from src.state import ContextItem, Gap, LoopSettings, SectionConfig, SectionStatus


# ---------------------------------------------------------------------------
# Scriptable fake for src.llm.call_structured
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Dispenses pre-set Pydantic responses, one queue per wire model type.

    Call `.add(SomeModel, response1, response2, ...)` to script the ordered
    responses for that model type. Each call to the fake pops the next
    response off that model's queue. Running out raises immediately with a
    clear error instead of returning something bogus.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[object]] = {}

    def add(self, model_cls: type, *responses: object) -> "ScriptedLLM":
        self._queues.setdefault(model_cls.__name__, []).extend(responses)
        return self

    def __call__(self, prompt: str, model: type, llm=None, max_retries: int = 2, *, prompt_id=None):
        key = model.__name__
        queue = self._queues.get(key)
        if not queue:
            raise AssertionError(
                f"ScriptedLLM ran out of responses for wire model {key!r}. "
                f"Prompt preview: {prompt[:200]!r}"
            )
        return queue.pop(0)


@pytest.fixture
def scripted_llm(monkeypatch) -> ScriptedLLM:
    fake = ScriptedLLM()
    # Each agent module imported `call_structured` by name at module load
    # time, so the patch target is the name inside each module, not
    # src.llm.call_structured itself.
    monkeypatch.setattr("src.agents.orchestrator.call_structured", fake)
    monkeypatch.setattr("src.agents.gap_finder.call_structured", fake)
    monkeypatch.setattr("src.agents.gap_filler.call_structured", fake)
    monkeypatch.setattr("src.agents.critic.call_structured", fake)
    return fake


@pytest.fixture
def no_rag_hits(monkeypatch):
    """gap_filler.retrieve() always returns no hits (forces the question path)."""
    monkeypatch.setattr("src.agents.gap_filler.retrieve", lambda query, top_k=None: [])


@pytest.fixture
def graph():
    return build_graph()


def make_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def make_sections() -> list[SectionConfig]:
    return [
        SectionConfig(id="sec_a", title="Section A", description="desc A", required=True, template_slot="a"),
        SectionConfig(id="sec_b", title="Section B", description="desc B", required=True, template_slot="b"),
    ]


def base_seed(sections: list[SectionConfig], statuses: dict[str, SectionStatus], **overrides) -> dict:
    seed = {
        "sections_config": sections,
        "context_items": [],
        "gaps": [],
        "section_statuses": statuses,
        "asked_questions": [],
        "loop_settings": LoopSettings(max_turns=15, max_questions_per_batch=3, max_questions_per_gap=2),
        "turn": 0,
        "pending_user_questions": [],
        "current_mode": "section",
        "current_section_id": None,
        "active_fresh_item_ids": [],
        "done": False,
        "stop_reason": None,
    }
    seed.update(overrides)
    return seed


# ---------------------------------------------------------------------------
# 1. A gap with no RAG hits reaches human_input as an interrupt.
# ---------------------------------------------------------------------------


def test_gap_with_no_rag_hits_reaches_human_input_interrupt(graph, scripted_llm, no_rag_hits):
    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="empty"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    seed = base_seed(sections, statuses)
    config = make_config()

    scripted_llm.add(
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
            section_complete=False,
        ),
    )
    scripted_llm.add(QuestionDraft, QuestionDraft(question_text="Quel est le volume de production maximal visé ?"))
    scripted_llm.add(DedupVerdict, DedupVerdict())  # nothing already answers it

    # as_node="initial_scan": these tests exercise the orchestrator loop, so
    # they seed past the initial per-section scan rather than scripting it.
    graph.update_state(config, seed, as_node="initial_scan")
    result = graph.invoke(None, config)

    interrupts = result.get("__interrupt__")
    assert interrupts, f"expected an interrupt, got result={result!r}"
    payload = interrupts[0].value
    assert payload["questions"] == [
        {
            "gap_id": result_gap_id(graph, config),
            "text": "Quel est le volume de production maximal visé ?",
        }
    ]
    assert result.get("done") is not True

    state = graph.get_state(config)
    assert state.next == ("human_input",)


def test_interrupt_payload_is_severity_ordered_and_capped(graph, scripted_llm, no_rag_hits):
    """End-to-end: what the user is actually shown is the top-N by severity.

    Seeds gaps that would have been found in earlier turns (so they are *not*
    the ones gap_finder returns this turn) and asserts they still win on
    severity, and that the form never exceeds max_questions_per_batch.
    """
    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="in_progress"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    leftovers = [
        Gap(id="gap_old_nice", section_ids=["sec_a"], category="scope",
            description="vieille lacune mineure", severity="nice_to_have", rag_attempted=True),
        Gap(id="gap_old_blocking", section_ids=["sec_a"], category="business_rule",
            description="vieille lacune bloquante", severity="blocking", rag_attempted=True),
    ]
    seed = base_seed(
        sections,
        statuses,
        gaps=leftovers,
        loop_settings=LoopSettings(max_turns=15, max_questions_per_batch=2, max_questions_per_gap=2),
    )
    config = make_config()

    # This turn's gap_finder contributes one merely-important gap.
    scripted_llm.add(
        GapFinderOutput,
        GapFinderOutput(
            new_gaps=[
                GapCandidate(section_ids=["sec_a"], category="nfr",
                             description="nouvelle lacune importante", severity="important")
            ],
            resolved_gap_ids=[],
            section_complete=False,
        ),
    )
    # Drafting echoes the gap description so we can identify which gaps won.
    scripted_llm.add(
        QuestionDraft,
        QuestionDraft(question_text="Q-bloquante"),
        QuestionDraft(question_text="Q-importante"),
    )
    scripted_llm.add(DedupVerdict, DedupVerdict(), DedupVerdict())

    graph.update_state(config, seed, as_node="initial_scan")
    result = graph.invoke(None, config)

    payload = result["__interrupt__"][0].value
    # Cap respected, blocking first, and the old nice_to_have is excluded
    # despite being the oldest gap in the pool.
    assert [q["text"] for q in payload["questions"]] == ["Q-bloquante", "Q-importante"]
    assert payload["questions"][0]["gap_id"] == "gap_old_blocking"

    gaps_by_id = {g.id: g for g in graph.get_state(config).values["gaps"]}
    assert gaps_by_id["gap_old_blocking"].questions_asked == 1
    assert gaps_by_id["gap_old_nice"].questions_asked == 0
    assert gaps_by_id["gap_old_nice"].status == "open"  # stays open, competes again next turn


def result_gap_id(graph, config) -> str:
    gaps = graph.get_state(config).values["gaps"]
    assert len(gaps) == 1
    return gaps[0].id


# ---------------------------------------------------------------------------
# 2. A critic-detected contradiction reopens a complete section and loops
#    back to orchestrator instead of terminating.
# ---------------------------------------------------------------------------


def test_critic_contradiction_reopens_complete_section_and_loops_back(graph, scripted_llm):
    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="complete"),
        "sec_b": SectionStatus(section_id="sec_b", status="in_progress"),
    }
    from src.state import ContextItem

    answer_item = ContextItem(
        id="ctx_answer1",
        content="Le volume prévu est de 5000 tonnes/an.",
        source="user_answer",
        section_ids=["sec_b"],
        linked_gap_id="gap_1",
        turn_added=1,
        fresh=False,  # already consumed by a prior orchestrator pass
    )

    seed = base_seed(
        sections,
        statuses,
        context_items=[answer_item],
        turn=1,
        active_fresh_item_ids=[answer_item.id],
    )
    config = make_config()

    scripted_llm.add(
        CriticOutput,
        CriticOutput(
            contradictions=[
                ContradictionFinding(
                    section_ids=["sec_a", "sec_b"],
                    description=(
                        "Le volume de 5000 tonnes/an indiqué en section B dépasse la "
                        "capacité de production maximale déjà validée en section A."
                    ),
                    severity="blocking",
                )
            ]
        ),
    )

    graph.update_state(config, seed, as_node="integrate_answers")
    result = graph.invoke(None, config, interrupt_after=["critic"])

    assert result.get("done") is not True

    state = graph.get_state(config)
    assert state.next == ("orchestrator",), "graph should loop back to orchestrator, not terminate"

    statuses_after = state.values["section_statuses"]
    assert statuses_after["sec_a"].status == "reopened"
    assert "5000 tonnes" in statuses_after["sec_a"].reopen_reason

    gaps_after = state.values["gaps"]
    assert any(g.category == "contradiction" and g.severity == "blocking" for g in gaps_after)


# ---------------------------------------------------------------------------
# 3. Hitting max_turns with an open blocking gap ends the run cleanly
#    (done=True, stop_reason set) instead of looping forever.
# ---------------------------------------------------------------------------


def test_max_turns_with_blocking_gap_stops_instead_of_looping(graph, scripted_llm):
    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="in_progress"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    blocking_gap = Gap(
        id="gap_blocking",
        section_ids=["sec_a"],
        category="business_rule",
        description="Le volume de production maximal n'est toujours pas précisé.",
        severity="blocking",
        status="open",
    )

    seed = base_seed(
        sections,
        statuses,
        gaps=[blocking_gap],
        loop_settings=LoopSettings(max_turns=3, max_questions_per_batch=3, max_questions_per_gap=2),
        turn=2,  # orchestrator will bump this to 3 == max_turns
    )
    config = make_config()

    # No responses scripted anywhere: apply_loop_limits is pure logic and
    # must short-circuit before any node makes an LLM call. If it doesn't,
    # ScriptedLLM raises immediately instead of the test hanging.
    # as_node="initial_scan": these tests exercise the orchestrator loop, so
    # they seed past the initial per-section scan rather than scripting it.
    graph.update_state(config, seed, as_node="initial_scan")
    result = graph.invoke(None, config)

    assert result.get("done") is True
    assert result.get("stop_reason")
    assert "gap_blocking" not in result["stop_reason"]  # message uses description, not raw id
    assert "volume de production maximal" in result["stop_reason"]

    state = graph.get_state(config)
    assert state.next == (), "graph must terminate, not keep looping"


# ---------------------------------------------------------------------------
# 4. max_turns is enforced even while fresh items keep arriving.
# ---------------------------------------------------------------------------


def test_max_turns_is_not_bypassed_by_fresh_items(graph, scripted_llm):
    """Regression: the fresh-item shortcut used to run before the limit check.

    Every answer produces a fresh context item, whose re-scan produces the next
    question — so in bench_20260806_100245 the orchestrator took the fresh-item
    early return on every single turn and never reached apply_loop_limits at all.
    max_turns was unenforceable in exactly the runaway case it exists for.
    """
    sections = make_sections()
    statuses = {
        "sec_a": SectionStatus(section_id="sec_a", status="in_progress"),
        "sec_b": SectionStatus(section_id="sec_b", status="empty"),
    }
    blocking_gap = Gap(
        id="gap_blocking",
        section_ids=["sec_a"],
        category="business_rule",
        description="Le volume de production maximal n'est toujours pas précisé.",
        severity="blocking",
        status="open",
    )
    seed = base_seed(
        sections,
        statuses,
        gaps=[blocking_gap],
        # A fresh item waiting to be re-scanned — the state the loop sat in.
        context_items=[
            ContextItem(
                id="ctx_fresh",
                content="30 jours",
                source="user_answer",
                section_ids=["sec_a"],
                turn_added=2,
                fresh=True,
            )
        ],
        loop_settings=LoopSettings(max_turns=3, max_questions_per_batch=3, max_questions_per_gap=2),
        turn=2,
    )
    config = make_config()

    # Again nothing is scripted: reaching any agent means the limit was skipped.
    graph.update_state(config, seed, as_node="initial_scan")
    result = graph.invoke(None, config)

    assert result.get("done") is True
    assert "volume de production maximal" in result["stop_reason"]
    assert graph.get_state(config).next == ()
