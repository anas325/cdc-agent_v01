"""Regression tests for the contradiction question loop.

In bench_20260806_100245 / cdc_003_ecommerce the swarm asked the same question
("panier conservé 24 h ou 30 jours ?") eleven times across seven turns and never
reached synthesis. The loop was self-sustaining:

    assumption (24 h) contradicts an answer (30 j)
      -> critic raises a contradiction gap
      -> user answers "30 jours"  (a new, still-fresh context item)
      -> the 24 h assumption is still there, so the critic raises it again
         under a *new* gap id, because the id hashes a description quoting
         whichever ctx id happened to be fresh
      -> ...

Each test below pins one of the links in that chain.
"""

from __future__ import annotations

import pytest

from src.agents.critic import ContradictionFinding, CriticOutput, run_critic
from src.context_utils import coerce_section_ids, format_context_items, live_items
from src.graph import _merge_gaps, integrate_answers_node
from src.state import ContextItem, Gap, PendingQuestion, SectionConfig, SectionStatus


SECTIONS = [
    SectionConfig(id="technical", title="Spécifications techniques", description="", template_slot="t"),
    SectionConfig(id="constraints", title="Contraintes", description="", template_slot="c"),
]


def make_item(id_: str, content: str, source: str = "user_answer", **kw) -> ContextItem:
    fields = {
        "id": id_,
        "content": content,
        "source": source,
        "section_ids": ["technical"],
        "turn_added": 1,
    }
    fields.update(kw)
    return ContextItem(**fields)


def make_state(**overrides) -> dict:
    state = {
        "sections_config": SECTIONS,
        "context_items": [],
        "gaps": [],
        "asked_questions": [],
        "pending_user_questions": [],
        "section_statuses": {s.id: SectionStatus(section_id=s.id, status="empty") for s in SECTIONS},
        "turn": 5,
        "active_fresh_item_ids": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# section_ids validation — the defect that let everything else churn
# ---------------------------------------------------------------------------


def test_context_item_ids_are_never_accepted_as_section_ids():
    """The critic returned ['ctx_7f3e35b23286', 'ctx_004f1430d9b3'] as sections."""
    state = make_state()
    assert coerce_section_ids(state, ["ctx_7f3e35b23286", "ctx_004f1430d9b3"], ["technical"]) == [
        "technical"
    ]


def test_valid_section_ids_are_kept_and_deduplicated():
    state = make_state()
    assert coerce_section_ids(state, ["constraints", "technical", "constraints"], []) == [
        "constraints",
        "technical",
    ]


def test_fallback_is_filtered_too():
    state = make_state()
    assert coerce_section_ids(state, ["ctx_abc123456789"], ["ctx_def123456789"]) == []


def _run_critic_returning(monkeypatch, findings: list[ContradictionFinding], state: dict):
    monkeypatch.setattr(
        "src.agents.critic.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: CriticOutput(
            contradictions=findings
        ),
    )
    monkeypatch.setattr("src.agents.critic.current_model_name", lambda: "test-model")
    return run_critic(state, [it.id for it in state["context_items"] if it.fresh])


def test_critic_moves_misplaced_ctx_ids_into_conflicting_items(monkeypatch):
    state = make_state(
        context_items=[
            make_item("ctx_7f3e35b23286", "ASSUMPTION: panier conservé 24 h", source="assumption"),
            make_item("ctx_004f1430d9b3", "30 jours", fresh=True),
        ]
    )
    result = _run_critic_returning(
        monkeypatch,
        [
            ContradictionFinding(
                topic="conservation du panier apres erreur de paiement",
                section_ids=["ctx_7f3e35b23286", "ctx_004f1430d9b3"],
                description="24 h contre 30 jours",
            )
        ],
        state,
    )

    gap = result.new_gaps[0]
    assert gap.section_ids == ["technical"]  # recovered from the cited items
    assert sorted(gap.conflicting_item_ids) == ["ctx_004f1430d9b3", "ctx_7f3e35b23286"]


def test_the_same_contradiction_keeps_one_gap_id_across_turns(monkeypatch):
    """The description quotes a different fresh ctx id each turn; the id must not follow."""
    ids = []
    for turn, fresh_id in enumerate(["ctx_2fa48d99b2ee", "ctx_f55ecbe81501", "ctx_34494bc84d6b"]):
        state = make_state(
            context_items=[
                make_item("ctx_7f3e35b23286", "ASSUMPTION: 24 h", source="assumption"),
                make_item(fresh_id, "30 jours", fresh=True),
            ],
            turn=5 + turn,
        )
        result = _run_critic_returning(
            monkeypatch,
            [
                ContradictionFinding(
                    topic="conservation du panier apres erreur de paiement",
                    section_ids=["technical"],
                    conflicting_item_ids=["ctx_7f3e35b23286", fresh_id],
                    # The wording the model actually varied, turn to turn.
                    description=f"Le nouvel élément ({fresh_id}) indique 30 jours, "
                    f"alors que l'hypothèse (ctx_7f3e35b23286) indique 24 heures.",
                )
            ],
            state,
        )
        ids.append(result.new_gaps[0].id)

    assert len(set(ids)) == 1, f"one contradiction produced {len(set(ids))} gap ids: {ids}"


# ---------------------------------------------------------------------------
# gap merging
# ---------------------------------------------------------------------------


def test_merge_gaps_drops_a_gap_already_present():
    existing = [Gap(id="g1", category="contradiction", description="d", severity="important")]
    duplicate = Gap(id="g1", category="contradiction", description="d", severity="important")
    new = Gap(id="g2", category="edge_case", description="e", severity="important")

    assert [g.id for g in _merge_gaps(existing, [duplicate, new])] == ["g1", "g2"]


def test_merge_gaps_does_not_resurrect_a_settled_gap():
    existing = [
        Gap(id="g1", category="contradiction", description="d", severity="important", status="assumed")
    ]
    reported_again = Gap(id="g1", category="contradiction", description="d", severity="important")

    merged = _merge_gaps(existing, [reported_again])

    assert len(merged) == 1
    assert merged[0].status == "assumed"


def test_merge_gaps_collapses_duplicates_inside_one_batch():
    a = Gap(id="g1", category="contradiction", description="d", severity="important")
    b = Gap(id="g1", category="contradiction", description="d", severity="important")

    assert len(_merge_gaps([], [a, b])) == 1


# ---------------------------------------------------------------------------
# superseding — the rule that actually ends the loop
# ---------------------------------------------------------------------------


@pytest.fixture
def contradiction_state():
    assumption = make_item(
        "ctx_assumption", "ASSUMPTION: le panier est conservé 24 h", source="assumption",
        validation_status="needs_review",
    )
    earlier = make_item("ctx_earlier", "Le panier est conservé 30 jours")
    gap = Gap(
        id="gap_contradiction",
        section_ids=["technical"],
        category="contradiction",
        description="24 h contre 30 jours",
        severity="important",
        conflicting_item_ids=["ctx_assumption", "ctx_earlier"],
    )
    return make_state(
        context_items=[assumption, earlier],
        gaps=[gap],
        pending_user_questions=[PendingQuestion(gap_id="gap_contradiction", text="24 h ou 30 jours ?")],
        _raw_answers={"gap_contradiction": {"text": "30 jours"}},
    )


def test_user_answer_supersedes_the_contradicted_assumption(contradiction_state):
    updates = integrate_answers_node(contradiction_state)

    by_id = {it.id: it for it in updates["context_items"]}
    answer = next(it for it in updates["context_items"] if it.source == "user_answer" and it.fresh)

    assert by_id["ctx_assumption"].superseded_by == answer.id
    assert by_id["ctx_assumption"].validation_status == "rejected"
    # The other side of the contradiction was never an assumption: untouched.
    assert by_id["ctx_earlier"].superseded_by is None


def test_superseded_items_leave_the_live_context(contradiction_state):
    """What the critic sees next turn — the conflict is gone, not just outvoted."""
    updates = integrate_answers_node(contradiction_state)

    rendered = format_context_items(updates["context_items"])

    assert "24 h" not in rendered
    assert "30 jours" in rendered
    assert len(live_items(updates["context_items"])) == len(updates["context_items"]) - 1


def test_superseding_is_logged(contradiction_state):
    updates = integrate_answers_node(contradiction_state)

    entry = next(d for d in updates["decision_log"] if d.decision_type == "assumption_superseded")
    assert entry.input_ids == ["ctx_assumption"]
    assert entry.details["rule"] == "user_answer_beats_assumption"


def test_a_user_answer_never_supersedes_another_user_answer(contradiction_state):
    """Two humans disagreeing is an editorial call, not something to auto-resolve."""
    contradiction_state["context_items"] = [
        make_item("ctx_assumption", "Le panier est conservé 24 h"),  # user_answer, not assumption
        make_item("ctx_earlier", "Le panier est conservé 30 jours"),
    ]

    updates = integrate_answers_node(contradiction_state)

    assert all(it.superseded_by is None for it in updates["context_items"])
    assert not [d for d in updates["decision_log"] if d.decision_type == "assumption_superseded"]


def test_a_non_contradiction_gap_supersedes_nothing(contradiction_state):
    contradiction_state["gaps"] = [
        contradiction_state["gaps"][0].model_copy(update={"category": "business_rule"})
    ]

    updates = integrate_answers_node(contradiction_state)

    assert all(it.superseded_by is None for it in updates["context_items"])


def test_fresh_ids_are_deterministically_ordered(contradiction_state):
    """Checkpointed state must not depend on set iteration order."""
    contradiction_state["active_fresh_item_ids"] = ["ctx_zzz", "ctx_aaa"]

    ids = integrate_answers_node(contradiction_state)["active_fresh_item_ids"]

    assert ids == sorted(ids)
