"""Critic agent.

Runs globally (never per-section) after every integration of new answers.
Checks whether newly added context items contradict earlier context or a
section already marked complete. Never validates gaps it created itself —
it only inspects state produced by other agents.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.context_utils import format_all_sections_context
from src.ids import stable_id
from src.llm import call_structured
from src.state import CDCState, Gap


class ContradictionFinding(BaseModel):
    section_ids: list[str] = Field(description="All sections involved in the contradiction, at least 2 preferred")
    description: str
    severity: str = "important"


class CriticOutput(BaseModel):
    contradictions: list[ContradictionFinding] = Field(default_factory=list)


class CriticResult(BaseModel):
    new_gaps: list[Gap]
    reopened_sections: dict[str, str]  # section_id -> reopen_reason


def run_critic(state: CDCState, fresh_item_ids: list[str]) -> CriticResult:
    fresh_items = [it for it in state["context_items"] if it.id in fresh_item_ids]
    if not fresh_items:
        return CriticResult(new_gaps=[], reopened_sections={})

    fresh_text = "\n".join(f"- id={it.id} sections={it.section_ids}: {it.content}" for it in fresh_items)
    complete_sections = [
        sid for sid, ss in state["section_statuses"].items() if ss.status == "complete"
    ]

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
conflit. Si aucune contradiction n'est trouvée, retourne une liste vide."""

    output = call_structured(prompt, CriticOutput)

    valid_severities = {"blocking", "important", "nice_to_have"}

    new_gaps: list[Gap] = []
    reopened: dict[str, str] = {}
    for finding in output.contradictions:
        severity = finding.severity if finding.severity in valid_severities else "important"
        gap = Gap(
            id=stable_id("gap", "critic", ",".join(finding.section_ids), finding.description),
            section_ids=finding.section_ids,
            category="contradiction",
            description=finding.description,
            severity=severity,  # type: ignore[arg-type]
        )
        new_gaps.append(gap)
        for sid in finding.section_ids:
            if sid in complete_sections:
                reopened[sid] = finding.description

    return CriticResult(new_gaps=new_gaps, reopened_sections=reopened)
