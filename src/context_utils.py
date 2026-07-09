"""Shared helpers for rendering state into LLM-readable text blocks."""

from __future__ import annotations

from src.state import CDCState, ContextItem, Gap, SectionConfig


def format_context_items(items: list[ContextItem], section_id: str | None = None) -> str:
    if section_id is not None:
        items = [it for it in items if section_id in it.section_ids or not it.section_ids]
    if not items:
        return "(aucun contexte pour l'instant)"
    lines = []
    for it in items:
        tag = f"[{it.source}]"
        if it.source == "assumption":
            tag = "[ASSUMPTION]"
        lines.append(f"- {tag} (id={it.id}, sections={it.section_ids}): {it.content}")
    return "\n".join(lines)


def format_all_sections_context(state: CDCState) -> str:
    sections: list[SectionConfig] = state["sections_config"]
    items = state["context_items"]
    blocks = []
    for sec in sections:
        blocks.append(f"### Section: {sec.title} (id={sec.id})\n{format_context_items(items, sec.id)}")
    return "\n\n".join(blocks)


def format_open_gaps(gaps: list[Gap], section_id: str | None = None) -> str:
    filtered = [g for g in gaps if g.status == "open"]
    if section_id is not None:
        filtered = [g for g in filtered if section_id in g.section_ids]
    if not filtered:
        return "(aucun gap ouvert)"
    lines = []
    for g in filtered:
        lines.append(
            f"- id={g.id} severity={g.severity} category={g.category} sections={g.section_ids}: {g.description}"
        )
    return "\n".join(lines)


def format_asked_questions(state: CDCState) -> str:
    aq = state.get("asked_questions", [])
    if not aq:
        return "(aucune question posée jusqu'ici)"
    return "\n".join(f"- (gap={a.gap_id}, turn={a.turn}): {a.text}" for a in aq)


def get_section(state: CDCState, section_id: str) -> SectionConfig:
    for sec in state["sections_config"]:
        if sec.id == section_id:
            return sec
    raise KeyError(section_id)
