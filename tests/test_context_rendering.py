"""Unit tests for context rendering helpers and prompt scoping.

Covers format_all_sections_context deduplication, the section-scoped
format_context_for_sections helper, and (via a prompt-capturing fake LLM)
that single-gap agents only receive context for the gap's sections.
"""

from __future__ import annotations

from src.context_utils import format_all_sections_context, format_context_for_sections
from src.state import ContextItem, Gap, SectionConfig


def make_section(id_: str, title: str | None = None) -> SectionConfig:
    return SectionConfig(id=id_, title=title or id_, description="", template_slot=id_)


def make_item(id_: str, content: str, section_ids: list[str]) -> ContextItem:
    return ContextItem(id=id_, content=content, source="initial_cdc", section_ids=section_ids, turn_added=0)


SECTIONS = [make_section("sec_a", "Section A"), make_section("sec_b", "Section B")]


def make_state(items: list[ContextItem]) -> dict:
    return {"sections_config": SECTIONS, "context_items": items}


# ---------------------------------------------------------------------------
# format_all_sections_context
# ---------------------------------------------------------------------------


def test_each_item_rendered_exactly_once():
    state = make_state([make_item("ctx_multi", "contenu multi-sections", ["sec_a", "sec_b"])])
    out = format_all_sections_context(state)
    assert out.count("ctx_multi") == 1
    assert out.count("contenu multi-sections") == 1


def test_single_section_item_only_in_its_block():
    state = make_state([make_item("ctx_a", "contenu A", ["sec_a"])])
    out = format_all_sections_context(state)
    block_a, rest = out.split("### Section: Section B", 1)
    assert "contenu A" in block_a
    assert "contenu A" not in rest


def test_untagged_item_in_shared_block():
    state = make_state([make_item("ctx_pre", "préambule non classé", [])])
    out = format_all_sections_context(state)
    assert out.count("préambule non classé") == 1
    shared = out.split("### Contexte transversal", 1)[1]
    assert "préambule non classé" in shared


def test_empty_sections_keep_placeholder():
    state = make_state([])
    out = format_all_sections_context(state)
    assert out.count("(aucun contexte pour l'instant)") == 3  # sec_a, sec_b, shared


# ---------------------------------------------------------------------------
# format_context_for_sections
# ---------------------------------------------------------------------------


def test_scoped_context_excludes_other_sections():
    state = make_state(
        [
            make_item("ctx_a", "contenu A", ["sec_a"]),
            make_item("ctx_b", "contenu B", ["sec_b"]),
            make_item("ctx_multi", "contenu partagé", ["sec_a", "sec_b"]),
            make_item("ctx_pre", "préambule", []),
        ]
    )
    out = format_context_for_sections(state, ["sec_a"])
    assert "contenu A" in out
    assert "contenu partagé" in out
    assert "préambule" in out
    assert "contenu B" not in out
    assert "### Section: Section B" not in out


def test_scoped_context_empty_ids_falls_back_to_full_view():
    state = make_state([make_item("ctx_b", "contenu B", ["sec_b"])])
    assert format_context_for_sections(state, []) == format_all_sections_context(state)


def test_scoped_context_ignores_unknown_section_id():
    state = make_state([make_item("ctx_a", "contenu A", ["sec_a"])])
    out = format_context_for_sections(state, ["sec_a", "hallucinated"])
    assert "contenu A" in out
    assert "hallucinated" not in out


def test_scoped_context_only_unknown_ids_falls_back_to_full_view():
    state = make_state([make_item("ctx_a", "contenu A", ["sec_a"])])
    assert format_context_for_sections(state, ["hallucinated"]) == format_all_sections_context(state)


# ---------------------------------------------------------------------------
# Prompt scoping in single-gap agents
# ---------------------------------------------------------------------------


def make_gap(section_ids: list[str]) -> Gap:
    return Gap(
        id="gap_1",
        section_ids=section_ids,
        category="functional_ambiguity",
        description="lacune de test",
        severity="important",
    )


def make_agent_state() -> dict:
    return {
        "sections_config": SECTIONS,
        "context_items": [
            make_item("ctx_a", "SECRET_A contenu A", ["sec_a"]),
            make_item("ctx_b", "SECRET_B contenu B", ["sec_b"]),
        ],
        "gaps": [],
        "asked_questions": [],
    }


class PromptCapture:
    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt, model, **kwargs):
        self.prompts.append(prompt)
        return self.response

    def assert_scoped_to_sec_a(self):
        assert len(self.prompts) == 1
        assert "SECRET_A" in self.prompts[0]
        assert "SECRET_B" not in self.prompts[0]


def test_draft_question_prompt_scoped_to_gap_sections(monkeypatch):
    from src.agents import gap_filler

    capture = PromptCapture(gap_filler.QuestionDraft(question_text="Q ?"))
    monkeypatch.setattr(gap_filler, "call_structured", capture)
    gap_filler._draft_question(make_agent_state(), make_gap(["sec_a"]))
    capture.assert_scoped_to_sec_a()


def test_build_assumption_prompt_scoped_to_gap_sections(monkeypatch):
    from src.agents import gap_filler

    capture = PromptCapture(gap_filler.AssumptionDraft(assumption_text="ASSUMPTION: x"))
    monkeypatch.setattr(gap_filler, "call_structured", capture)
    gap_filler.build_assumption(make_agent_state(), make_gap(["sec_a"]), turn=1)
    capture.assert_scoped_to_sec_a()


def test_dedup_gate_prompt_scoped_to_gap_sections(monkeypatch):
    from src.agents import orchestrator

    capture = PromptCapture(orchestrator.DedupVerdict())
    monkeypatch.setattr(orchestrator, "call_structured", capture)
    orchestrator.dedup_gate(make_agent_state(), make_gap(["sec_a"]), "Question candidate ?")
    capture.assert_scoped_to_sec_a()
