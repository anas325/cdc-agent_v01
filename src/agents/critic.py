"""Critic agent.

Runs globally (never per-section) after every integration of new answers.
Checks whether newly added context items contradict earlier context or a
section already marked complete. Never validates gaps it created itself —
it only inspects state produced by other agents.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.context_utils import (
    coerce_section_ids,
    format_all_sections_context,
    live_items,
    sections_of_items,
    valid_section_ids,
)
from src.decisions import make_decision
from src.ids import stable_id
from src.llm import call_structured, current_model_name
from src.state import CDCState, DecisionLogEntry, Gap
from src.utils.text_match import content_tokens


class ContradictionFinding(BaseModel):
    topic: str = Field(
        default="",
        description="Sujet court de la contradiction, ex. 'conservation du panier apres erreur de paiement'",
    )
    section_ids: list[str] = Field(description="All sections involved in the contradiction, at least 2 preferred")
    conflicting_item_ids: list[str] = Field(
        default_factory=list, description="ids (ctx_...) of the context items that conflict"
    )
    description: str
    severity: str = "important"


class CriticOutput(BaseModel):
    contradictions: list[ContradictionFinding] = Field(default_factory=list)


class CriticResult(BaseModel):
    new_gaps: list[Gap]
    reopened_sections: dict[str, str]  # section_id -> reopen_reason
    decisions: list[DecisionLogEntry] = Field(default_factory=list)


def _contradiction_key(finding: ContradictionFinding) -> str:
    """A stable identity for "this contradiction", independent of its wording.

    The description is free text that quotes whichever `ctx_…` id happened to be
    fresh this turn, so hashing it hands the same contradiction a new gap id on
    every re-detection — which resets `questions_asked` and defeats the per-gap
    question budget. Hash the normalized topic instead, falling back to the
    id-stripped, order-independent content words of the description.
    """
    topic = " ".join(sorted(content_tokens(finding.topic)))
    return topic or " ".join(sorted(content_tokens(finding.description)))


def run_critic(state: CDCState, fresh_item_ids: list[str]) -> CriticResult:
    fresh_items = [it for it in live_items(state["context_items"]) if it.id in fresh_item_ids]
    if not fresh_items:
        return CriticResult(new_gaps=[], reopened_sections={})

    fresh_text = "\n".join(f"- id={it.id} sections={it.section_ids}: {it.content}" for it in fresh_items)
    complete_sections = [
        sid for sid, ss in state["section_statuses"].items() if ss.status == "complete"
    ]
    legal_sections = sorted(valid_section_ids(state))

    prompt = f"""Tu es le contrôleur qualité (critic) d'un processus de rédaction de cahier des
charges (CDC). Des nouveaux éléments viennent d'être intégrés au contexte. Ton seul rôle est de
détecter des INCOHÉRENCES entre ces nouveaux éléments et le reste du contexte (pas de créer de
nouvelles lacunes de complétude, seulement des contradictions factuelles).

CONTEXTE COMPLET (toutes sections) :
{format_all_sections_context(state)}

NOUVEAUX ÉLÉMENTS INTÉGRÉS CE TOUR :
{fresh_text}

SECTIONS ACTUELLEMENT MARQUÉES "COMPLETE" : {complete_sections}

Cherche spécifiquement :
1. Un nouvel élément qui contredit un élément de contexte antérieur (même section ou non).
2. Un nouvel élément qui contredit le contenu d'une section déjà marquée "complete" ci-dessus.

Ne signale QUE des contradictions concrètes et vérifiables citant les deux affirmations en
conflit. Si aucune contradiction n'est trouvée, retourne une liste vide.

Pour CHAQUE contradiction, renseigne :
- section_ids : UNIQUEMENT des identifiants de section pris dans cette liste
  fermée : {legal_sections}. N'y mets JAMAIS un identifiant d'élément de
  contexte (ctx_...) : ce sont deux choses différentes.
- conflicting_item_ids : c'est ICI que vont les identifiants ctx_... des
  éléments qui s'opposent (au moins deux).
- topic : le sujet de la contradiction en 3 à 8 mots, sans identifiant ni
  chiffre, ex. "conservation du panier apres erreur de paiement". Formule-le de
  façon IDENTIQUE si la même contradiction réapparaît plus tard."""

    output = call_structured(prompt, CriticOutput, prompt_id="critic.contradiction")

    valid_severities = {"blocking", "important", "nice_to_have"}

    new_gaps: list[Gap] = []
    reopened: dict[str, str] = {}
    decisions: list[DecisionLogEntry] = []
    for finding in output.contradictions:
        severity = finding.severity if finding.severity in valid_severities else "important"
        # An id the model put in the wrong field is still information: a ctx id
        # landing in section_ids belongs with the conflicting items.
        conflicting = [
            cid
            for cid in dict.fromkeys(finding.conflicting_item_ids + finding.section_ids)
            if cid not in valid_section_ids(state)
        ]
        section_ids = coerce_section_ids(
            state,
            finding.section_ids,
            fallback=sections_of_items(state, conflicting or fresh_item_ids)
            or sections_of_items(state, fresh_item_ids),
        )
        gap = Gap(
            id=stable_id("gap", "critic", ",".join(sorted(section_ids)), _contradiction_key(finding)),
            section_ids=section_ids,
            category="contradiction",
            description=finding.description,
            severity=severity,  # type: ignore[arg-type]
            conflicting_item_ids=conflicting,
        )
        new_gaps.append(gap)
        reopened_here = [sid for sid in section_ids if sid in complete_sections]
        for sid in reopened_here:
            reopened[sid] = finding.description
        decisions.append(
            make_decision(
                state,
                agent="critic",
                decision_type="contradiction_found",
                summary=finding.description,
                prompt_id="critic.contradiction",
                input_ids=fresh_item_ids,
                output_ids=[gap.id],
                model=current_model_name(),
                severity=severity,
                section_ids=section_ids,
                conflicting_item_ids=conflicting,
                reopened_sections=reopened_here,
            )
        )

    return CriticResult(new_gaps=new_gaps, reopened_sections=reopened, decisions=decisions)
