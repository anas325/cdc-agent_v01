"""Gap-filler agent.

For each open gap, tries RAG first; grades the retrieved answer with an
LLM call. If insufficient, formulates one precise, context-referencing
question for the user. Also builds ASSUMPTION context items when the user
answers "I don't know".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.context_utils import format_context_for_sections
from src.ids import new_id
from src.llm import call_structured
from src.rag import retrieve
from src.state import CDCState, ContextItem, Gap, PendingQuestion

_SEVERITY_ORDER = {"blocking": 0, "important": 1, "nice_to_have": 2}


class RagGrade(BaseModel):
    sufficient: bool
    answer_summary: str = ""


class QuestionDraft(BaseModel):
    question_text: str


class AssumptionDraft(BaseModel):
    assumption_text: str = Field(description="Must start with 'ASSUMPTION:' and state a concrete default.")


class FillResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    new_context_items: list[ContextItem] = Field(default_factory=list)
    pending_questions: list[PendingQuestion] = Field(default_factory=list)
    gap_updates: dict[str, str] = Field(default_factory=dict)  # gap_id -> new status


def _grade_rag_hits(gap: Gap, hits: list[dict]) -> RagGrade:
    hits_text = "\n\n".join(f"[{h['source']}] {h['content']}" for h in hits)
    prompt = f"""Un cahier des charges présente la lacune suivante :
"{gap.description}" (catégorie={gap.category}, sévérité={gap.severity})

Voici des extraits de documents de référence récupérés par recherche sémantique :
{hits_text}

Ces extraits répondent-ils de façon SUFFISANTE et PRÉCISE à la lacune, sans ambiguïté restante ?
Si oui, résume la réponse concrète à retenir (answer_summary). Si les extraits sont hors-sujet,
partiels, ou n'apportent pas de réponse actionnable, réponds sufficient=false."""
    return call_structured(prompt, RagGrade)


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
    return call_structured(prompt, QuestionDraft).question_text


def fill_gaps(state: CDCState, gap_ids: list[str], turn: int) -> FillResult:
    gaps_by_id = {g.id: g for g in state["gaps"]}
    open_gaps = [gaps_by_id[gid] for gid in gap_ids if gaps_by_id.get(gid) and gaps_by_id[gid].status == "open"]
    open_gaps.sort(key=lambda g: _SEVERITY_ORDER[g.severity])

    result = FillResult()

    for gap in open_gaps:
        hits = retrieve(gap.description)
        if hits:
            grade = _grade_rag_hits(gap, hits)
            if grade.sufficient:
                item = ContextItem(
                    id=new_id("ctx"),
                    content=f"[RAG] {grade.answer_summary}",
                    source="rag",
                    section_ids=gap.section_ids,
                    linked_gap_id=gap.id,
                    turn_added=turn,
                    fresh=True,
                )
                result.new_context_items.append(item)
                result.gap_updates[gap.id] = "rag_answered"
                continue

        question_text = _draft_question(state, gap)
        result.pending_questions.append(PendingQuestion(gap_id=gap.id, text=question_text))

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
"ASSUMPTION:" suivi d'une phrase concrète et actionnable en français."""
    draft = call_structured(prompt, AssumptionDraft)
    text = draft.assumption_text.strip()
    if not text.upper().startswith("ASSUMPTION:"):
        text = f"ASSUMPTION: {text}"
    return ContextItem(
        id=new_id("ctx"),
        content=text,
        source="assumption",
        section_ids=gap.section_ids,
        linked_gap_id=gap.id,
        turn_added=turn,
        fresh=True,
    )
