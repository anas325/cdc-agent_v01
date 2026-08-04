"""Gap-filler agent.

Walks every open gap in severity order (most severe first) and, for each one,
tries RAG first, grading the retrieved answer with an LLM call. If insufficient,
formulates one precise, context-referencing question for the user, and stops once
the turn's question budget is full. Also builds ASSUMPTION context items when a
gap runs out of question budget or the user answers "I don't know".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents import orchestrator as orch
from src.context_utils import format_context_for_sections
from src.decisions import (
    clamp_confidence,
    evidence_from_hit,
    make_decision,
    now_iso,
    validation_for,
)
from src.ids import stable_id
from src.llm import call_structured, current_model_name
from src.prompts import version as prompt_version
from src.rag import retrieve
from src.state import (
    CDCState,
    ContextItem,
    DecisionLogEntry,
    EvidenceGrade,
    Gap,
    PendingQuestion,
)

_SEVERITY_ORDER = {"blocking": 0, "important": 1, "nice_to_have": 2}

# Gap-type tie-break within a (section, severity) bucket: contradictions first,
# then the remaining categories by descending business impact (mirrors the
# CATEGORY_WEIGHTS used for section scoring in the UI).
_CATEGORY_ORDER = {
    "contradiction": 0,
    "scope": 1,
    "functional_ambiguity": 2,
    "business_rule": 3,
    "acceptance_criteria": 4,
    "integration": 5,
    "data_model": 6,
    "nfr": 7,
    "edge_case": 8,
}


def _pool_sort_key(gap: Gap, section_rank: dict[str, int], n_sections: int):
    """Rank a gap in the candidate pool: section, then severity, then gap type.

    ``section_rank`` maps a section id to its position in sections_config, with
    skipped sections omitted so a gap left attached only to skipped sections (or
    to none) sinks to the bottom. A gap spanning several sections is ranked by
    its earliest still-active section.
    """
    ranks = [section_rank[sid] for sid in gap.section_ids if sid in section_rank]
    section = min(ranks) if ranks else n_sections
    return (
        section,
        _SEVERITY_ORDER[gap.severity],
        _CATEGORY_ORDER.get(gap.category, len(_CATEGORY_ORDER)),
    )


class RagGrade(BaseModel):
    sufficient: bool
    answer_summary: str = ""
    # Auto-évaluation du LLM, utilisée comme score opérationnel (pas une
    # probabilité) : elle pilote validation_status et l'affichage UI.
    confidence: float = 0.0
    evidence_grade: EvidenceGrade = "insufficient"


class QuestionDraft(BaseModel):
    question_text: str


class AssumptionDraft(BaseModel):
    assumption_text: str = Field(description="Must start with 'ASSUMPTION:' and state a concrete default.")
    confidence: float = 0.0


class FillResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    new_context_items: list[ContextItem] = Field(default_factory=list)
    pending_questions: list[PendingQuestion] = Field(default_factory=list)
    gap_updates: dict[str, str] = Field(default_factory=dict)  # gap_id -> new status
    rag_attempted_gap_ids: list[str] = Field(default_factory=list)
    resolved_by: dict[str, str] = Field(default_factory=dict)  # gap_id -> answering context item id
    question_texts: dict[str, str] = Field(default_factory=dict)  # gap_id -> question asked
    decisions: list[DecisionLogEntry] = Field(default_factory=list)


def _format_hit(hit: dict) -> str:
    """Cite a retrieved chunk with its document and page, so the grader can too."""
    page = hit.get("page")
    where = f"{hit.get('document', hit.get('source', 'inconnu'))}"
    if page is not None:
        where += f", p. {page}"
    return f"[{where}] {hit['content']}"


def _grade_rag_hits(gap: Gap, hits: list[dict]) -> RagGrade:
    hits_text = "\n\n".join(_format_hit(h) for h in hits)
    prompt = f"""Un cahier des charges présente la lacune suivante :
"{gap.description}" (catégorie={gap.category}, sévérité={gap.severity})

Voici des extraits de documents de référence récupérés par recherche sémantique :
{hits_text}

Ces extraits répondent-ils de façon SUFFISANTE et PRÉCISE à la lacune, sans ambiguïté restante ?
Si oui, résume la réponse concrète à retenir (answer_summary). Si les extraits sont hors-sujet,
partiels, ou n'apportent pas de réponse actionnable, réponds sufficient=false.

Indique aussi :
- evidence_grade : "sufficient" (les extraits répondent pleinement), "partial" (ils apportent
  un élément de réponse mais laissent une ambiguïté), "insufficient" (hors-sujet ou muets).
- confidence : un score entre 0.0 et 1.0 reflétant ta certitude que la réponse retenue est
  correcte et actionnable (≥ 0.80 = haute certitude, 0.50–0.79 = moyenne, < 0.50 = faible).
  Sois honnête : une confiance surévaluée fait accepter automatiquement une réponse fausse."""
    grade = call_structured(prompt, RagGrade, prompt_id="gap_filler.rag_grade")
    return grade.model_copy(update={"confidence": clamp_confidence(grade.confidence) or 0.0})


def _draft_question(state: CDCState, gap: Gap) -> str:
    prompt = f"""Un cahier des charges présente la lacune suivante, qui n'a pas pu être résolue
par la documentation existante :
Description : "{gap.description}"
Catégorie : {gap.category}
Sévérité : {gap.severity}
Sections concernées : {gap.section_ids}

CONTEXTE PERTINENT du CDC (sections concernées par la lacune, pour référencer précisément les éléments ambigus) :
{format_context_for_sections(state, gap.section_ids)}

Rédige UNE question précise et contextualisée à poser à l'utilisateur (rédacteur du CDC) pour
lever cette ambiguïté. La question DOIT :
- citer ou paraphraser l'élément ambigu concret du texte (pas de question générique type
  "pouvez-vous préciser le périmètre ?"),
- si c'est une contradiction, nommer explicitement les deux affirmations qui se contredisent,
- être formulée en français, courte, directe, à choix ouvert."""
    return call_structured(prompt, QuestionDraft, prompt_id="gap_filler.question").question_text


def fill_gaps(state: CDCState, turn: int, max_batch: int, max_per_gap: int) -> FillResult:
    """Selects the questions to ask this turn, section by section.

    The candidate pool is *every* open gap, not just the ones found this turn, so a
    gap left over from an earlier turn competes with fresh ones. Gaps are ranked by
    section (in sections_config order), then severity (most severe first), then gap
    type (contradictions first). The loop processes them lazily and stops as soon as
    ``max_batch`` questions are held, so nothing is drafted that won't be asked and
    there is no leftover queue to carry over: an un-asked gap stays ``open`` and
    competes again next turn.
    """
    sections = state.get("sections_config", [])
    statuses = state.get("section_statuses", {})
    section_rank = {
        sec.id: i
        for i, sec in enumerate(sections)
        if not (statuses.get(sec.id) and statuses[sec.id].status == "skipped")
    }

    open_gaps = [g for g in state["gaps"] if g.status == "open"]
    # Stable sort: full ties (same section/severity/type) keep creation order.
    open_gaps.sort(key=lambda g: _pool_sort_key(g, section_rank, len(sections)))

    result = FillResult()

    for gap in open_gaps:
        if len(result.pending_questions) >= max_batch:
            break

        # Gap has exhausted its question budget: settle it with a default assumption
        # rather than spending a drafting call on a question we can't ask.
        if gap.questions_asked >= max_per_gap:
            item = build_assumption(state, gap, turn)
            result.new_context_items.append(item)
            result.gap_updates[gap.id] = "assumed"
            result.decisions.append(
                make_decision(
                    state,
                    agent="gap_filler",
                    decision_type="assumption_built",
                    summary=f"Budget de questions épuisé : hypothèse par défaut retenue pour {gap.id}.",
                    prompt_id="gap_filler.assumption",
                    input_ids=[gap.id],
                    output_ids=[item.id],
                    confidence=item.confidence,
                    model=item.model,
                    reason="question_budget_exhausted",
                    questions_asked=gap.questions_asked,
                )
            )
            continue

        if not gap.rag_attempted:
            result.rag_attempted_gap_ids.append(gap.id)
            hits = retrieve(gap.description)
            if hits:
                grade = _grade_rag_hits(gap, hits)
                evidence = [evidence_from_hit(h) for h in hits]
                evidence_ids = [ev.chunk_id for ev in evidence]
                if grade.sufficient:
                    item = ContextItem(
                        id=stable_id("ctx", "rag", gap.id, grade.answer_summary),
                        content=f"[RAG] {grade.answer_summary}",
                        source="rag",
                        section_ids=gap.section_ids,
                        linked_gap_id=gap.id,
                        turn_added=turn,
                        fresh=True,
                        created_by="rag",
                        timestamp=now_iso(),
                        evidence=evidence,
                        evidence_grade=grade.evidence_grade,
                        confidence=grade.confidence,
                        validation_status=validation_for(grade.confidence),
                        model=current_model_name(),
                        prompt_version=prompt_version("gap_filler.rag_grade"),
                    )
                    result.new_context_items.append(item)
                    result.gap_updates[gap.id] = "rag_answered"
                    result.decisions.append(
                        make_decision(
                            state,
                            agent="gap_filler",
                            decision_type="rag_answer",
                            summary=f"Lacune {gap.id} résolue par la documentation : {grade.answer_summary}",
                            prompt_id="gap_filler.rag_grade",
                            input_ids=[gap.id],
                            output_ids=[item.id],
                            evidence_ids=evidence_ids,
                            confidence=grade.confidence,
                            model=current_model_name(),
                            evidence_grade=grade.evidence_grade,
                            validation_status=item.validation_status,
                        )
                    )
                    continue
                result.decisions.append(
                    make_decision(
                        state,
                        agent="gap_filler",
                        decision_type="rag_rejected",
                        summary=f"Extraits jugés insuffisants pour {gap.id} : passage à une question.",
                        prompt_id="gap_filler.rag_grade",
                        input_ids=[gap.id],
                        evidence_ids=evidence_ids,
                        confidence=grade.confidence,
                        model=current_model_name(),
                        evidence_grade=grade.evidence_grade,
                    )
                )

        question_text = _draft_question(state, gap)

        verdict = orch.dedup_gate(state, gap, question_text)
        if verdict.already_resolved and verdict.resolved_by_item_id:
            result.gap_updates[gap.id] = "resolved"
            result.resolved_by[gap.id] = verdict.resolved_by_item_id
            result.decisions.append(
                make_decision(
                    state,
                    agent="gap_filler",
                    decision_type="question_deduped",
                    summary=f"Question non posée pour {gap.id} : le contexte existant y répond déjà.",
                    prompt_id="orchestrator.dedup",
                    input_ids=[gap.id, verdict.resolved_by_item_id],
                    model=current_model_name(),
                    candidate_question=question_text,
                )
            )
            continue
        if verdict.partially_resolved and verdict.rewritten_question:
            result.decisions.append(
                make_decision(
                    state,
                    agent="gap_filler",
                    decision_type="question_deduped",
                    summary=f"Question de {gap.id} reformulée sur la seule partie non couverte.",
                    prompt_id="orchestrator.dedup",
                    input_ids=[gap.id],
                    model=current_model_name(),
                    original_question=question_text,
                    rewritten_question=verdict.rewritten_question,
                )
            )
            question_text = verdict.rewritten_question

        result.pending_questions.append(PendingQuestion(gap_id=gap.id, text=question_text))
        result.question_texts[gap.id] = question_text
        result.decisions.append(
            make_decision(
                state,
                agent="gap_filler",
                decision_type="question_drafted",
                summary=f"Question posée à l'utilisateur pour {gap.id} : {question_text}",
                prompt_id="gap_filler.question",
                input_ids=[gap.id],
                model=current_model_name(),
                severity=gap.severity,
                category=gap.category,
                rag_attempted=gap.rag_attempted or gap.id in result.rag_attempted_gap_ids,
            )
        )

    return result


def build_assumption(state: CDCState, gap: Gap, turn: int) -> ContextItem:
    prompt = f"""L'utilisateur n'a pas su répondre à la lacune suivante d'un cahier des charges :
Description : "{gap.description}"
Catégorie : {gap.category}
Sections concernées : {gap.section_ids}

CONTEXTE PERTINENT (sections concernées par la lacune) :
{format_context_for_sections(state, gap.section_ids)}

Propose une hypothèse par défaut raisonnable (pragmatique, standard du secteur) pour combler
cette lacune, afin que le développement puisse démarrer. Le texte DOIT commencer par
"ASSUMPTION:" suivi d'une phrase concrète et actionnable en français.

Indique aussi confidence : un score entre 0.0 et 1.0 reflétant à quel point cette valeur par
défaut est un standard sûr du secteur (proche de 1.0 si elle est quasi certaine, proche de 0.0
si elle est arbitraire et devra impérativement être confirmée)."""
    draft = call_structured(prompt, AssumptionDraft, prompt_id="gap_filler.assumption")
    text = draft.assumption_text.strip()
    if not text.upper().startswith("ASSUMPTION:"):
        text = f"ASSUMPTION: {text}"
    return ContextItem(
        id=stable_id("ctx", "assumption", gap.id, text),
        content=text,
        source="assumption",
        section_ids=gap.section_ids,
        linked_gap_id=gap.id,
        turn_added=turn,
        fresh=True,
        created_by="llm",
        timestamp=now_iso(),
        confidence=clamp_confidence(draft.confidence),
        # Une hypothèse est par définition non validée : quelle que soit la
        # confiance annoncée, elle doit être relue avant d'être tenue pour acquise.
        validation_status="needs_review",
        model=current_model_name(),
        prompt_version=prompt_version("gap_filler.assumption"),
    )
