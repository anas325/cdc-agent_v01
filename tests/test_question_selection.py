"""Unit tests for fill_gaps() question selection.

Selection must always be driven by severity across the *whole* pool of open
gaps, not just the ones found in the current turn, and must never hand back
more than max_questions_per_batch questions.

src.rag.retrieve and the two call_structured entry points (gap_filler for
drafting, orchestrator for the dedup gate) are the only things faked.
"""

from __future__ import annotations

import pytest

from src.agents.gap_filler import QuestionDraft, fill_gaps
from src.agents.orchestrator import DedupVerdict
from src.state import AskedQuestion, CDCState, Gap, SectionConfig


@pytest.fixture
def no_rag_hits(monkeypatch):
    monkeypatch.setattr("src.agents.gap_filler.retrieve", lambda query, top_k=None: [])


@pytest.fixture
def draft_calls(monkeypatch) -> list[str]:
    """Records every drafted question; returns the gap description verbatim."""
    seen: list[str] = []

    def fake_draft(prompt: str, model: type, llm=None, max_retries: int = 2, *, prompt_id=None):
        seen.append(prompt)
        # The gap description is quoted in the drafting prompt; echo it back so
        # tests can map a question to the gap it came from.
        return QuestionDraft(question_text=prompt.split('Description : "')[1].split('"')[0])

    monkeypatch.setattr("src.agents.gap_filler.call_structured", fake_draft)
    return seen


@pytest.fixture
def dedup_passthrough(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: DedupVerdict(),
    )


def make_gap(gap_id: str, severity: str, **overrides) -> Gap:
    fields = {
        "id": gap_id,
        "section_ids": ["sec_a"],
        "category": "functional_ambiguity",
        "description": gap_id,
        "severity": severity,
        "status": "open",
        "rag_attempted": True,  # skip the RAG path unless a test opts back in
    }
    fields.update(overrides)
    return Gap(**fields)


def make_state(gaps: list[Gap]) -> CDCState:
    return {
        "sections_config": [
            SectionConfig(id="sec_a", title="Section A", description="desc A", required=True, template_slot="a")
        ],
        "context_items": [],
        "gaps": gaps,
        "asked_questions": [],
        "section_statuses": {},
    }


def test_older_blocking_gap_outranks_newer_nice_to_have(no_rag_hits, draft_calls, dedup_passthrough):
    """The pool is global: a leftover blocking gap beats a fresh trivial one."""
    state = make_state(
        [
            make_gap("old_blocking", "blocking"),
            make_gap("new_nice", "nice_to_have"),
        ]
    )

    result = fill_gaps(state, turn=5, max_batch=1, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["old_blocking"]
    # The losing candidate must not have consumed a drafting call.
    assert len(draft_calls) == 1


def test_questions_are_severity_ordered_and_capped(no_rag_hits, draft_calls, dedup_passthrough):
    state = make_state(
        [
            make_gap("n1", "nice_to_have"),
            make_gap("i1", "important"),
            make_gap("b1", "blocking"),
            make_gap("i2", "important"),
            make_gap("b2", "blocking"),
        ]
    )

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["b1", "b2", "i1"]
    assert len(draft_calls) == 3  # nothing drafted beyond the cap


def test_pool_ranks_by_section_then_severity_then_type(no_rag_hits, draft_calls, dedup_passthrough):
    """Section is the primary key; within a section, severity then gap type."""
    state = {
        "sections_config": [
            SectionConfig(id="sec_a", title="A", description="a", required=True, template_slot="a"),
            SectionConfig(id="sec_b", title="B", description="b", required=True, template_slot="b"),
        ],
        "context_items": [],
        "gaps": [
            # sec_b comes second even though it holds a blocking gap.
            make_gap("b_blocking", "blocking", section_ids=["sec_b"]),
            # Within sec_a: same severity, contradiction must precede edge_case.
            make_gap("a_edge", "important", section_ids=["sec_a"], category="edge_case"),
            make_gap("a_contra", "important", section_ids=["sec_a"], category="contradiction"),
        ],
        "asked_questions": [],
        "section_statuses": {},
    }

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["a_contra", "a_edge", "b_blocking"]


def test_skipped_section_gaps_sink_below_active_ones(no_rag_hits, draft_calls, dedup_passthrough):
    """A gap on a skipped section is ranked last, behind every active-section gap."""
    from src.state import SectionStatus

    state = {
        "sections_config": [
            SectionConfig(id="sec_a", title="A", description="a", required=True, template_slot="a"),
            SectionConfig(id="sec_b", title="B", description="b", required=True, template_slot="b"),
        ],
        "context_items": [],
        "gaps": [
            make_gap("a_blocking", "blocking", section_ids=["sec_a"]),
            make_gap("b_blocking", "blocking", section_ids=["sec_b"]),
        ],
        "asked_questions": [],
        "section_statuses": {"sec_a": SectionStatus(section_id="sec_a", status="skipped")},
    }

    result = fill_gaps(state, turn=1, max_batch=2, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["b_blocking", "a_blocking"]


def test_ties_keep_gap_creation_order(no_rag_hits, draft_calls, dedup_passthrough):
    state = make_state([make_gap("b1", "blocking"), make_gap("b2", "blocking"), make_gap("b3", "blocking")])

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["b1", "b2", "b3"]


def test_non_open_gaps_are_never_selected(no_rag_hits, draft_calls, dedup_passthrough):
    state = make_state(
        [
            make_gap("resolved", "blocking", status="resolved"),
            make_gap("deferred", "blocking", status="deferred"),
            make_gap("open", "nice_to_have"),
        ]
    )

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["open"]


def test_exhausted_gap_becomes_assumption_without_drafting(no_rag_hits, draft_calls, monkeypatch):
    """A gap at its question budget is settled with an assumption, not a question."""
    from src.agents.gap_filler import AssumptionDraft

    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: AssumptionDraft(assumption_text="ASSUMPTION: défaut."),
    )
    state = make_state([make_gap("spent", "blocking", questions_asked=2)])

    result = fill_gaps(state, turn=4, max_batch=3, max_per_gap=2)

    assert result.pending_questions == []
    assert result.gap_updates == {"spent": "assumed"}
    assert len(result.new_context_items) == 1
    assert result.new_context_items[0].source == "assumption"
    assert result.new_context_items[0].linked_gap_id == "spent"


def test_rag_is_attempted_once_per_gap(monkeypatch, dedup_passthrough, draft_calls):
    """A gap already RAG-graded is not re-retrieved on a later turn."""
    queries: list[str] = []
    monkeypatch.setattr(
        "src.agents.gap_filler.retrieve",
        lambda query, top_k=None: queries.append(query) or [],
    )
    state = make_state(
        [
            make_gap("fresh_gap", "blocking", rag_attempted=False),
            make_gap("already_graded", "blocking", rag_attempted=True),
        ]
    )

    result = fill_gaps(state, turn=2, max_batch=3, max_per_gap=2)

    assert queries == ["fresh_gap"]
    assert result.rag_attempted_gap_ids == ["fresh_gap"]
    assert [pq.gap_id for pq in result.pending_questions] == ["fresh_gap", "already_graded"]


def test_dedup_drop_advances_to_next_candidate(no_rag_hits, draft_calls, monkeypatch):
    """A question killed by the dedup gate must not leave the batch short."""
    verdicts = [
        DedupVerdict(already_resolved=True, resolved_by_item_id="ctx_1"),
        DedupVerdict(),
    ]
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: verdicts.pop(0),
    )
    state = make_state([make_gap("dropped", "blocking"), make_gap("kept", "important")])

    result = fill_gaps(state, turn=1, max_batch=1, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["kept"]
    assert result.gap_updates == {"dropped": "resolved"}
    assert result.resolved_by == {"dropped": "ctx_1"}


def test_dedup_rewrite_replaces_question_text(no_rag_hits, draft_calls, monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: DedupVerdict(
            partially_resolved=True, rewritten_question="Question reformulée ?"
        ),
    )
    state = make_state([make_gap("g1", "blocking")])

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.text for pq in result.pending_questions] == ["Question reformulée ?"]


# ---------------------------------------------------------------------------
# Repeat suppression
#
# The gate is an LLM judgement on "does the context already answer this?", which
# for a still-live contradiction is honestly "no" no matter how many times the
# question has gone out. These cover the deterministic check that sits in front
# of it — see tests/test_contradiction_loop.py for the structural half of the fix.
# ---------------------------------------------------------------------------

CART_QUESTION = (
    "Quelle durée de conservation du panier après une erreur de paiement doit être retenue, "
    "30 jours ou 24 heures ?"
)


def make_state_with_history(gaps: list[Gap], asked: list[AskedQuestion]) -> CDCState:
    state = make_state(gaps)
    state["asked_questions"] = asked
    return state


def test_a_question_already_asked_is_not_asked_again(no_rag_hits, dedup_passthrough, monkeypatch):
    """The exact loop from bench_20260806_100245: same question, new gap id."""
    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: QuestionDraft(
            question_text="Le panier doit-il être conservé 30 jours ou 24 heures après une "
            "erreur de paiement ?"
        ),
    )
    answered = make_gap("old_gap", "important", answer_item_ids=["ctx_answer"])
    state = make_state_with_history(
        [answered, make_gap("new_gap_same_question", "important")],
        [AskedQuestion(id="q1", gap_id="old_gap", text=CART_QUESTION, turn=4)],
    )

    result = fill_gaps(state, turn=5, max_batch=3, max_per_gap=2)

    assert result.pending_questions == []
    assert result.gap_updates["new_gap_same_question"] == "resolved"
    # Closed against the answer the earlier question already got.
    assert result.resolved_by["new_gap_same_question"] == "ctx_answer"


def test_a_vague_rewrite_of_an_asked_question_is_dropped(no_rag_hits, draft_calls, monkeypatch):
    """"Quelle durée doit être retenue ?" — the rewrite branch's endgame."""
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: DedupVerdict(
            partially_resolved=True, rewritten_question="Quelle durée doit être retenue ?"
        ),
    )
    state = make_state_with_history(
        [make_gap("g1", "blocking")],
        [AskedQuestion(id="q1", gap_id="g0", text=CART_QUESTION, turn=4)],
    )

    result = fill_gaps(state, turn=5, max_batch=3, max_per_gap=2)

    assert result.pending_questions == []
    assert result.gap_updates["g1"] == "resolved"


def test_a_genuinely_new_question_still_gets_through(no_rag_hits, dedup_passthrough, monkeypatch):
    """The guard must not swallow an unrelated question asked in the same run."""
    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: QuestionDraft(
            question_text="Quel est le budget maximal alloué à la phase 1 du projet ?"
        ),
    )
    state = make_state_with_history(
        [make_gap("g1", "blocking")],
        [AskedQuestion(id="q1", gap_id="g0", text=CART_QUESTION, turn=4)],
    )

    result = fill_gaps(state, turn=5, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["g1"]


def test_two_gaps_with_the_same_question_are_asked_once(no_rag_hits, dedup_passthrough, monkeypatch):
    """Duplicate gaps put the identical question in one batch (round 10 of the run)."""
    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: QuestionDraft(
            question_text=CART_QUESTION
        ),
    )
    state = make_state([make_gap("g1", "important"), make_gap("g2", "important")])

    result = fill_gaps(state, turn=5, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["g1"]
    assert result.gap_updates["g2"] == "resolved"


def test_gate_already_asked_verdict_drops_the_question(no_rag_hits, draft_calls, monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: DedupVerdict(
            already_asked=True
        ),
    )
    state = make_state_with_history(
        [make_gap("g1", "blocking")],
        [AskedQuestion(id="q1", gap_id="g0", text=CART_QUESTION, turn=4)],
    )

    result = fill_gaps(state, turn=5, max_batch=3, max_per_gap=2)

    assert result.pending_questions == []
    assert result.gap_updates["g1"] == "resolved"


def test_gate_already_asked_is_ignored_when_nothing_was_ever_asked(
    no_rag_hits, draft_calls, monkeypatch
):
    """A hallucinated verdict must not silence the very first question."""
    monkeypatch.setattr(
        "src.agents.orchestrator.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: DedupVerdict(
            already_asked=True
        ),
    )
    state = make_state([make_gap("g1", "blocking")])

    result = fill_gaps(state, turn=1, max_batch=3, max_per_gap=2)

    assert [pq.gap_id for pq in result.pending_questions] == ["g1"]
