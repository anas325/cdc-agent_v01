"""Gap-finder agent.

Works through the gap taxonomy systematically for a focused section (or a
set of freshly-added context items), using completion_hints as a checklist,
and flags cross-section contradictions.

Prompt wording lives in `gap_finder_prompts.py` — keep it out of this file so
the two can be iterated (and A/B tested) independently of the agent logic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.agents.gap_finder_prompts import (
    GAP_TAXONOMY,
    build_prompt_fresh_mode,
    build_prompt_section_mode,
)
from src.context_utils import coerce_section_ids, sections_of_items
from src.decisions import clamp_confidence, make_decision
from src.ids import stable_id
from src.llm import call_structured, current_model_name
from src.state import CDCState, DecisionLogEntry, Gap, GapCategory, GapSeverity

__all__ = [
    "GAP_TAXONOMY",
    "GapCandidate",
    "GapFinderOutput",
    "GapFinderResult",
    "compute_section_complete",
    "run_gap_finder",
]


class GapCandidate(BaseModel):
    section_ids: list[str] = Field(default_factory=list)
    category: GapCategory
    description: str
    severity: GapSeverity
    follow_up_of_gap_id: str | None = None
    # Optionnel et toléré absent : les petits modèles l'omettent souvent.
    confidence: float | None = None


class GapFinderOutput(BaseModel):
    new_gaps: list[GapCandidate] = Field(default_factory=list)
    resolved_gap_ids: list[str] = Field(default_factory=list)



class GapFinderResult(BaseModel):
    new_gaps: list[Gap]
    resolved_gap_ids: list[str]
    section_complete: bool | None
    decisions: list[DecisionLogEntry] = Field(default_factory=list)

def compute_section_complete(
    state: CDCState,
    section_id: str,
    new_gaps: list[Gap],
    resolved_gap_ids: list[str],
) -> bool:
    """
    A section is complete iff it has no open blocking/important gaps
    after applying the latest gap-finder results.
    """

    remaining_existing = [
        gap
        for gap in state["gaps"]
        if section_id in gap.section_ids
        and gap.id not in resolved_gap_ids
        and gap.status != "resolved"
    ]

    all_open_gaps = remaining_existing + [
        gap for gap in new_gaps if section_id in gap.section_ids
    ]

    return not any(
        gap.severity in {"blocking", "important"}
        for gap in all_open_gaps
    )


def run_gap_finder(
    state: CDCState,
    mode: Literal["section", "fresh"],
    section_id: str | None = None,
    fresh_item_ids: list[str] | None = None,
) -> GapFinderResult:
    if mode == "section":
        assert section_id is not None
        prompt = build_prompt_section_mode(state, section_id)
        prompt_id = "gap_finder.section"
    else:
        assert fresh_item_ids
        prompt = build_prompt_fresh_mode(state, fresh_item_ids)
        prompt_id = "gap_finder.fresh"

    output = call_structured(prompt, GapFinderOutput, prompt_id=prompt_id)

    new_gaps: list[Gap] = []
    decisions: list[DecisionLogEntry] = []
    # Le mode "fresh" n'a pas de section courante : on retombe sur les sections
    # des éléments évalués plutôt que de laisser passer des ids inventés.
    default_sections = [section_id] if section_id else sections_of_items(state, fresh_item_ids or [])
    for cand in output.new_gaps:
        gap = Gap(
            id=stable_id("gap", section_id or "", cand.category, cand.description),
            section_ids=coerce_section_ids(state, cand.section_ids, fallback=default_sections),
            category=cand.category,
            description=cand.description,
            severity=cand.severity,
        )
        new_gaps.append(gap)
        decisions.append(
            make_decision(
                state,
                agent="gap_finder",
                decision_type="gap_detected",
                summary=f"[{gap.severity}/{gap.category}] {gap.description}",
                prompt_id=prompt_id,
                input_ids=fresh_item_ids or ([section_id] if section_id else []),
                output_ids=[gap.id],
                confidence=clamp_confidence(cand.confidence),
                model=current_model_name(),
                mode=mode,
                section_ids=gap.section_ids,
                follow_up_of_gap_id=cand.follow_up_of_gap_id,
            )
        )
    section_complete = None

    if mode == "section":
        section_complete = compute_section_complete(
            state=state,
            section_id=section_id,
            new_gaps=new_gaps,
            resolved_gap_ids=output.resolved_gap_ids,
        )

    return GapFinderResult(
        new_gaps=new_gaps,
        resolved_gap_ids=output.resolved_gap_ids,
        section_complete=section_complete if mode == "section" else None,
        decisions=decisions,
    )
