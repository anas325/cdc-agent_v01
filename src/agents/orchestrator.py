"""Orchestrator agent.

Pure routing/bookkeeping logic (which section to work on next, loop-limit
enforcement) plus the question dedup gate, which is the one LLM call this
agent makes: before any question reaches the user, check whether existing
context already answers it (fully or partially).
"""

from __future__ import annotations

from pydantic import BaseModel

from src.context_utils import format_all_sections_context, format_asked_questions
from src.ids import new_id
from src.llm import call_structured
from src.state import AskedQuestion, CDCState, Gap, PendingQuestion, SectionStatus


def pick_next_section(state: CDCState) -> str | None:
    statuses = state["section_statuses"]
    sections = state["sections_config"]

    for target_status in ("reopened", "in_progress", "empty"):
        for sec in sections:
            if not sec.required:
                continue
            st = statuses.get(sec.id)
            if st and st.status == target_status:
                return sec.id
            if target_status == "empty" and st is None:
                return sec.id
    return None


def all_required_complete(state: CDCState) -> bool:
    statuses = state["section_statuses"]
    for sec in state["sections_config"]:
        if not sec.required:
            continue
        st = statuses.get(sec.id)
        if st is None or st.status != "complete":
            return False
    return True


class LimitCheckResult(BaseModel):
    hit_limit: bool = False
    blocking_stop: bool = False
    stop_message: str = ""
    downgraded_gap_ids: list[str] = []


def apply_loop_limits(state: CDCState) -> LimitCheckResult:
    max_turns = state["loop_settings"].max_turns
    if state.get("turn", 0) < max_turns:
        return LimitCheckResult()

    open_gaps: list[Gap] = [g for g in state["gaps"] if g.status == "open"]
    blocking_open = [g for g in open_gaps if g.severity == "blocking"]
    if blocking_open:
        names = "; ".join(g.description for g in blocking_open)
        return LimitCheckResult(
            hit_limit=True,
            blocking_stop=True,
            stop_message=(
                f"Limite de {max_turns} tours atteinte avec des lacunes bloquantes non résolues : {names}"
            ),
        )

    downgraded = [g.id for g in open_gaps]
    return LimitCheckResult(hit_limit=True, blocking_stop=False, downgraded_gap_ids=downgraded)


class DedupVerdict(BaseModel):
    already_resolved: bool = False
    partially_resolved: bool = False
    resolved_by_item_id: str | None = None
    rewritten_question: str = ""


def dedup_gate(state: CDCState, gap: Gap, candidate_question: str) -> DedupVerdict:
    prompt = f"""Une question est sur le point d'être posée à l'utilisateur pour combler une
lacune d'un cahier des charges. Avant de la poser, vérifie si le contexte déjà disponible ou une
question déjà posée n'y répond pas déjà.

LACUNE : {gap.description}

QUESTION CANDIDATE : "{candidate_question}"

CONTEXTE DÉJÀ DISPONIBLE :
{format_all_sections_context(state)}

QUESTIONS DÉJÀ POSÉES PRÉCÉDEMMENT :
{format_asked_questions(state)}

Réponds :
- already_resolved=true si le contexte répond déjà PLEINEMENT à la question (donne
  resolved_by_item_id = id de l'élément de contexte qui répond).
- partially_resolved=true si le contexte répond partiellement : dans ce cas fournis
  rewritten_question ne portant que sur la partie manquante.
- Sinon (rien ne répond), laisse already_resolved=false, partially_resolved=false,
  rewritten_question="" (la question candidate sera posée telle quelle)."""
    return call_structured(prompt, DedupVerdict)


def batch_questions(pending: list[PendingQuestion], max_batch: int) -> tuple[list[PendingQuestion], list[PendingQuestion]]:
    """Returns (this_turn_batch, remaining_for_later)."""
    return pending[:max_batch], pending[max_batch:]


def record_asked_questions(state: CDCState, batch: list[PendingQuestion], turn: int) -> list[AskedQuestion]:
    return [AskedQuestion(id=new_id("q"), gap_id=pq.gap_id, text=pq.text, turn=turn) for pq in batch]


def new_section_status_map(state: CDCState) -> dict[str, SectionStatus]:
    statuses = dict(state.get("section_statuses", {}))
    for sec in state["sections_config"]:
        if sec.id not in statuses:
            statuses[sec.id] = SectionStatus(section_id=sec.id, status="empty")
    return statuses
