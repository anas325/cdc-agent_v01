"""Gap-finder agent.

Works through the gap taxonomy systematically for a focused section (or a
set of freshly-added context items), using completion_hints as a checklist,
and flags cross-section contradictions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.context_utils import (
    coerce_section_ids,
    format_all_sections_context,
    format_open_gaps,
    get_section,
    sections_of_items,
)
from src.decisions import clamp_confidence, make_decision
from src.ids import stable_id
from src.llm import call_structured, current_model_name
from src.state import CDCState, DecisionLogEntry, Gap, GapCategory, GapSeverity

GAP_TAXONOMY = [
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


def _build_prompt_section_mode(state: CDCState, section_id: str) -> str:
    section = get_section(state, section_id)
    hints = "\n".join(f"- {h}" for h in section.completion_hints) or "(aucun critère spécifique)"
    return f"""Tu es un analyste qui audite un cahier des charges (CDC) pour en détecter les lacunes
et ambiguïtés, avant qu'il ne soit transmis à une équipe de développement.

CONTEXTE COMPLET (toutes sections, car les incohérences peuvent être transversales) :
{format_all_sections_context(state)}

GAPS DÉJÀ OUVERTS SUR CETTE SECTION (ne pas dupliquer) :
{format_open_gaps(state["gaps"], section_id)}

Concentre-toi maintenant sur la section : "{section.title}" (id={section.id})
Description de la section : {section.description}

Critères de complétude à vérifier un par un (checklist) :
{hints}

Analyse cette section de façon SYSTÉMATIQUE en passant par CHAQUE catégorie de lacune suivante,
une par une, et identifie les problèmes concrets (pas de généralités) :
{", ".join(GAP_TAXONOMY)}

Pour la catégorie "contradiction", compare explicitement le contenu de cette section avec les
AUTRES sections et signale toute incohérence transversale (severity au moins "important",
section_ids doit alors lister TOUTES les sections concernées).

Ne remonte QUE des lacunes concrètes et actionnables, citant les éléments ambigus du texte.
N'invente pas de lacune si le contexte répond déjà clairement au critère.
"""


def _build_prompt_fresh_mode(state: CDCState, fresh_item_ids: list[str]) -> str:
    items = [it for it in state["context_items"] if it.id in fresh_item_ids]
    items_text = "\n".join(
        f"- id={it.id} (source={it.source}, linked_gap_id={it.linked_gap_id}, sections={it.section_ids}): {it.content}"
        for it in items
    )
    gaps_text = "\n".join(
        f"- id={g.id} status={g.status} severity={g.severity}: {g.description}"
        for g in state["gaps"]
        if g.id in {it.linked_gap_id for it in items if it.linked_gap_id}
    )
    return f"""Tu es un analyste qui audite un cahier des charges (CDC). De nouvelles informations
viennent d'être ajoutées au contexte (réponses utilisateur ou résultats RAG). Tu dois vérifier
si elles résolvent réellement les lacunes auxquelles elles répondent.

CONTEXTE COMPLET (toutes sections) :
{format_all_sections_context(state)}

NOUVEAUX ÉLÉMENTS À ÉVALUER :
{items_text}

LACUNES ORIGINALES CONCERNÉES :
{gaps_text}

Pour chaque nouvel élément lié à une lacune (linked_gap_id) :
- Si la réponse résout complètement la lacune → ajoute son gap_id à resolved_gap_ids.
- Si la réponse est partielle, ambiguë, ou introduit une nouvelle incohérence avec le reste du
  contexte → NE PAS la marquer comme résolue, et crée un nouveau gap de suivi (follow_up_of_gap_id
  = id de la lacune originale) décrivant précisément ce qui manque encore.
- Si l'élément est une "ASSUMPTION" (hypothèse par défaut faute de réponse), vérifie aussi
  qu'elle ne contredit pas une autre section déjà validée ; si oui, crée un gap "contradiction".

Vérifie aussi si ces nouveaux éléments révèlent une incohérence transversale avec une section
déjà marquée comme complète ailleurs dans le contexte (catégorie "contradiction")."""


def run_gap_finder(
    state: CDCState,
    mode: Literal["section", "fresh"],
    section_id: str | None = None,
    fresh_item_ids: list[str] | None = None,
) -> GapFinderResult:
    if mode == "section":
        assert section_id is not None
        prompt = _build_prompt_section_mode(state, section_id)
        prompt_id = "gap_finder.section"
    else:
        assert fresh_item_ids
        prompt = _build_prompt_fresh_mode(state, fresh_item_ids)
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
