"""Shared helpers for rendering state into LLM-readable text blocks."""

from __future__ import annotations

from src.state import CDCState, ContextItem, Gap, SectionConfig


def live_items(items: list[ContextItem]) -> list[ContextItem]:
    """Context items still in force: superseded ones are no longer part of the CDC.

    An item retired by a later, more authoritative answer (see
    ContextItem.superseded_by) must disappear from every prompt — otherwise the
    critic keeps re-detecting the contradiction that the answer just settled.
    """
    return [it for it in items if it.superseded_by is None]


def format_context_items(items: list[ContextItem], section_id: str | None = None) -> str:
    items = live_items(items)
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
    """Render every context item exactly once, grouped by section.

    Single-section items go under their section's block; multi-section and
    untagged items go under a final shared block (each line already prints
    its `sections=[...]` scope).
    """
    return _format_sections_blocks(state["sections_config"], state["context_items"])


def format_context_for_sections(state: CDCState, section_ids: list[str]) -> str:
    """Render only the context relevant to `section_ids` (plus shared items).

    Falls back to the full view when `section_ids` is empty; unknown ids are
    ignored (gap section_ids come from LLM output and may be invalid).
    """
    wanted = set(section_ids)
    sections = [s for s in state["sections_config"] if s.id in wanted]
    if not sections:
        return format_all_sections_context(state)
    items = [
        it
        for it in state["context_items"]
        if not it.section_ids or wanted & set(it.section_ids)
    ]
    return _format_sections_blocks(sections, items)


def _format_sections_blocks(sections: list[SectionConfig], items: list[ContextItem]) -> str:
    blocks = []
    for sec in sections:
        own = [it for it in items if it.section_ids == [sec.id]]
        blocks.append(f"### Section: {sec.title} (id={sec.id})\n{format_context_items(own)}")
    shared = [it for it in items if len(it.section_ids) != 1]
    blocks.append(
        f"### Contexte transversal / général (plusieurs sections ou non classé)\n{format_context_items(shared)}"
    )
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


def valid_section_ids(state: CDCState) -> set[str]:
    """The only section ids that exist — sections are config, never LLM output."""
    return {s.id for s in state.get("sections_config", [])}


def coerce_section_ids(state: CDCState, raw_ids: list[str], fallback: list[str]) -> list[str]:
    """Keep only real section ids, falling back when an agent returned none.

    Agents are asked for section ids but readily hand back *context item* ids
    instead — every context line in their prompt is rendered as `id=ctx_…`, so
    that is what they echo. Left unchecked those ids propagate: they land on the
    Gap, get copied onto the answer ContextItem built from it, and that answer
    then belongs to no section at all — invisible to section completion, and
    re-randomising the content-hashed gap id on every re-detection.

    `fallback` is used verbatim when nothing valid survives (it is filtered too).
    """
    known = valid_section_ids(state)
    kept = [sid for sid in dict.fromkeys(raw_ids) if sid in known]
    if kept:
        return kept
    return [sid for sid in dict.fromkeys(fallback) if sid in known]


def sections_of_items(state: CDCState, item_ids: list[str]) -> list[str]:
    """Union of the section ids carried by `item_ids` — the natural fallback."""
    wanted = set(item_ids)
    out: list[str] = []
    for it in state.get("context_items", []):
        if it.id in wanted:
            out.extend(it.section_ids)
    return [sid for sid in dict.fromkeys(out) if sid in valid_section_ids(state)]


def get_section(state: CDCState, section_id: str) -> SectionConfig:
    for sec in state["sections_config"]:
        if sec.id == section_id:
            return sec
    raise KeyError(section_id)
