"""Registry of prompt ids and their versions.

Every `call_structured` call passes a `prompt_id` from this table. The version
string it maps to is stamped onto the resulting ContextItem, the decision log
and the telemetry record, so an audit can tell which wording produced a given
answer. It is also folded into the LLM disk-cache key (src/llm_cache.py), so
bumping a version here correctly invalidates cached responses for that prompt
instead of silently replaying stale ones.

**When you edit a prompt's wording, bump its version here in the same commit.**
"""

from __future__ import annotations

PROMPT_VERSIONS: dict[str, str] = {
    # v3 : test d'ancrage (une double lecture ne compte que si les deux lectures
    # tiennent à du texte présent), guide de catégorisation par test décisif,
    # few-shots négatifs décrits par leur forme au lieu de citer le passage,
    # plafond ramené à 3, et matériel des few-shots renouvelé pour ne plus
    # recouper le jeu d'évaluation — cf. gap_finder_prompts.py.
    "gap_finder.section": "v3",
    "gap_finder.fresh": "v3",
    "gap_filler.rag_grade": "v1",
    "gap_filler.question": "v1",
    "gap_filler.assumption": "v1",
    # v2 : verdict "already_asked" ajouté (une question déjà posée et répondue
    # n'est pas la même chose qu'un contexte qui y répond).
    "orchestrator.dedup": "v2",
    # v2 : section_ids contraint à la liste fermée des sections, ids ctx_...
    # déplacés dans conflicting_item_ids, et topic stable ajouté.
    "critic.contradiction": "v2",
    "synthesizer.slot": "v1",
    "final_validator.contradictions": "v1",
    # Not a graph agent: the synthetic stakeholder that answers question batches
    # during batch evaluation (evals/simulator.py). Registered here so its calls
    # are versioned and cached like every other prompt.
    "simulator.answer": "v1",
    # Also not a graph agent: the optional LLM judge that rates question quality
    # during scoring (evals/scoring.py, `run_scoring.py --judge llm`). It never
    # feeds the run — it only grades one after the fact.
    "judge.question_quality": "v1",
}


class UnknownPromptError(KeyError):
    pass


def version(prompt_id: str | None) -> str | None:
    """Version string for `prompt_id`, or None if no id was supplied.

    Raises for an id that isn't registered: a typo would otherwise silently
    produce unversioned audit records.
    """
    if prompt_id is None:
        return None
    try:
        return PROMPT_VERSIONS[prompt_id]
    except KeyError as exc:
        raise UnknownPromptError(
            f"Unknown prompt_id {prompt_id!r}; register it in src/prompts.py"
        ) from exc
