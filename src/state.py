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

# Qui a produit l'information : l'utilisateur, la recherche documentaire, un
# raisonnement du LLM, ou la mécanique du système (découpage du CDC initial).
CreatedBy = Literal["user", "rag", "llm", "system"]

# État de revue d'un élément de contexte. Voir les seuils dans src/decisions.py.
ValidationStatus = Literal["unreviewed", "accepted", "rejected", "needs_review"]

# Qualité des preuves documentaires derrière une réponse RAG.
EvidenceGrade = Literal["sufficient", "partial", "insufficient"]

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

SectionStatusValue = Literal["empty", "in_progress", "complete", "reopened", "skipped"]

# Décisions d'agent journalisées (voir src/decisions.py et docs/01).
DecisionType = Literal[
    "gap_detected",
    "rag_answer",
    "rag_rejected",
    "question_drafted",
    "question_deduped",
    "assumption_built",
    "assumption_superseded",
    "answer_integrated",
    "contradiction_found",
    "section_status_changed",
    "section_synthesized",
    "loop_limit_applied",
    "final_check",
]

EXCERPT_MAX_CHARS = 400


class Evidence(BaseModel):
    """Un extrait de document de référence qui étaye un ContextItem."""

    chunk_id: str  # id Chroma, ex. "stock_process.pdf::p14::3"
    document: str  # nom du fichier source
    page: int | None = None  # page PDF (1-based) ; None pour md/txt
    retrieval_score: float = 0.0  # 0–1, dérivé de la distance (voir src/rag.py)
    excerpt: str = ""  # texte du chunk, tronqué à EXCERPT_MAX_CHARS


class ContextItem(BaseModel):
    id: str
    content: str
    source: ContextSource
    section_ids: list[str] = Field(default_factory=list)
    linked_gap_id: str | None = None
    turn_added: int
    fresh: bool = False
    # --- Provenance (phase 1 « auditabilité ») ---------------------------
    # Tous ces champs ont une valeur par défaut : les checkpoints Postgres
    # écrits avant leur ajout doivent continuer à se désérialiser.
    created_by: CreatedBy = "system"
    timestamp: str | None = None  # ISO-8601 UTC
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_grade: EvidenceGrade | None = None
    confidence: float | None = None  # score opérationnel 0–1, pas une probabilité
    validation_status: ValidationStatus = "unreviewed"
    model: str | None = None  # modèle LLM ayant produit le contenu, le cas échéant
    prompt_version: str | None = None
    # Id de l'élément qui remplace celui-ci. Une hypothèse contredite puis
    # tranchée par une réponse utilisateur n'est pas supprimée (l'audit doit
    # pouvoir la relire) : elle est retirée du contexte vivant. Voir
    # context_utils.live_items et la règle dans graph.integrate_answers_node.
    superseded_by: str | None = None


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
    rag_attempted: bool = False
    # Pour une contradiction : les ContextItem qui s'opposent. C'est ce qui
    # permet de retirer le perdant une fois la contradiction tranchée, plutôt
    # que de la redétecter à chaque tour.
    conflicting_item_ids: list[str] = Field(default_factory=list)


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


class DecisionLogEntry(BaseModel):
    """Trace machine-lisible d'UNE décision d'agent.

    Complète turn_log (récit lisible pour l'UI) : ici on enregistre les
    identifiants en entrée/sortie, les preuves, la confiance et le couple
    modèle/version de prompt, pour pouvoir répondre après coup à « pourquoi le
    système a-t-il pris cette décision ? ».
    """

    id: str
    timestamp: str  # ISO-8601 UTC
    thread_id: str | None = None  # = run_id
    turn: int
    agent: str  # agent ayant décidé (gap_finder, critic, orchestrator, ...)
    decision_type: DecisionType
    summary: str
    input_ids: list[str] = Field(default_factory=list)  # gaps / items en entrée
    output_ids: list[str] = Field(default_factory=list)  # ids produits
    evidence_ids: list[str] = Field(default_factory=list)  # Evidence.chunk_id
    confidence: float | None = None
    model: str | None = None
    prompt_version: str | None = None
    details: dict = Field(default_factory=dict)


class SectionConfig(BaseModel):
    id: str
    title: str
    description: str
    required: bool = True
    template_slot: str
    completion_hints: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


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
    decision_log: Annotated[list[DecisionLogEntry], operator.add]
    initial_cdc_text: str
    # Mode « sans CDC initial » : l'auteur décrit lui-même chaque section au
    # lieu de déposer un document. section_id -> texte libre, consommé par
    # ingest_node au même titre que initial_cdc_text.
    initial_section_texts: dict[str, str]
    stop_reason: str | None
    current_mode: Literal["fresh", "section"]
    active_fresh_item_ids: list[str]
    _raw_answers: dict
    _qmd_text: str
    _mapped_items: dict[str, list[str]]
