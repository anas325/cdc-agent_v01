"""Pydantic state models for the CDC refinement agent graph.

Sections are never hard-coded here — they are loaded at runtime from
config/sections.yaml (see src/config.py) and referenced by id (str).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

ContextSource = Literal["initial_cdc", "user_answer", "rag", "assumption"]

GapCategory = Literal[
    "functional_ambiguity",
    "nfr",
    "data_model",
    "business_rule",
    "edge_case",
    "integration",
    "acceptance_criteria",
    "contradiction",
    "scope",
]

GapSeverity = Literal["blocking", "important", "nice_to_have"]

GapStatus = Literal[
    "open",
    "rag_answered",
    "user_answered",
    "assumed",
    "deferred",
    "resolved",
]

SectionStatusValue = Literal["empty", "in_progress", "complete", "reopened"]


class ContextItem(BaseModel):
    id: str
    content: str
    source: ContextSource
    section_ids: list[str] = Field(default_factory=list)
    linked_gap_id: str | None = None
    turn_added: int
    fresh: bool = False


class Gap(BaseModel):
    id: str
    section_ids: list[str] = Field(default_factory=list)
    category: GapCategory
    description: str
    severity: GapSeverity
    status: GapStatus = "open"
    question_text: str | None = None
    answer_item_ids: list[str] = Field(default_factory=list)
    questions_asked: int = 0


class SectionStatus(BaseModel):
    section_id: str
    status: SectionStatusValue = "empty"
    reopen_reason: str | None = None


class AskedQuestion(BaseModel):
    id: str
    gap_id: str
    text: str
    turn: int


class PendingQuestion(BaseModel):
    """A question queued for the human-in-the-loop batch this turn."""

    gap_id: str
    text: str


class TurnLogEntry(BaseModel):
    turn: int
    agent: str
    summary: str
    details: dict = Field(default_factory=dict)


class SectionConfig(BaseModel):
    id: str
    title: str
    description: str
    required: bool = True
    template_slot: str
    completion_hints: list[str] = Field(default_factory=list)


class LoopSettings(BaseModel):
    max_turns: int = 15
    max_questions_per_batch: int = 3
    max_questions_per_gap: int = 2


class CDCState(TypedDict, total=False):
    sections_config: list[SectionConfig]
    context_items: list[ContextItem]
    gaps: list[Gap]
    section_statuses: dict[str, SectionStatus]
    asked_questions: list[AskedQuestion]
    current_section_id: str | None
    turn: int
    pending_user_questions: list[PendingQuestion]
    done: bool
    loop_settings: LoopSettings
    turn_log: Annotated[list[TurnLogEntry], operator.add]
    initial_cdc_text: str
    stop_reason: str | None
    current_mode: Literal["fresh", "section"]
    active_fresh_item_ids: list[str]
    active_gap_ids: list[str]
    _raw_answers: dict
    _qmd_text: str
    _mapped_items: dict[str, list[str]]
