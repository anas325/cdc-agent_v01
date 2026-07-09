"""LangGraph wiring for the CDC refinement agent swarm.

START -> ingest -> orchestrator -> (route)
    -> gap_finder -> gap_filler -> (route: pending questions?)
         -> human_input -> integrate_answers -> critic -> orchestrator
         -> critic -> orchestrator
    -> synthesizer -> final_validator -> END
    -> END (blocking gaps hit max_turns)
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents import critic as critic_agent
from src.agents import final_validator as final_validator_agent
from src.agents import gap_filler as gap_filler_agent
from src.agents import gap_finder as gap_finder_agent
from src.agents import orchestrator as orch
from src.agents import synthesizer as synthesizer_agent
from src.config import load_sections, load_settings
from src.ids import new_id
from src.rag import ingest_source_docs
from src.state import CDCState, ContextItem, PendingQuestion, SectionStatus, TurnLogEntry


def _log(state: CDCState, agent: str, summary: str, **details) -> TurnLogEntry:
    return TurnLogEntry(turn=state.get("turn", 0), agent=agent, summary=summary, details=details)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def ingest_node(state: CDCState) -> dict:
    settings = load_settings()
    sections = load_sections()
    ingest_source_docs()

    initial_text = state.get("initial_cdc_text", "") or ""
    context_items: list[ContextItem] = []
    if initial_text.strip():
        context_items.append(
            ContextItem(
                id=new_id("ctx"),
                content=initial_text.strip(),
                source="initial_cdc",
                section_ids=[s.id for s in sections],
                turn_added=0,
                fresh=False,
            )
        )

    section_statuses = {s.id: SectionStatus(section_id=s.id, status="empty") for s in sections}

    return {
        "sections_config": sections,
        "loop_settings": state.get("loop_settings") or settings.loop,
        "context_items": context_items,
        "gaps": [],
        "section_statuses": section_statuses,
        "asked_questions": [],
        "turn_log": [_log(state, "ingest", f"Chargement du CDC initial et ingestion RAG ({len(sections)} sections).")],
        "turn": 0,
        "pending_user_questions": [],
        "done": False,
        "stop_reason": None,
    }


def orchestrator_node(state: CDCState) -> dict:
    turn = state.get("turn", 0) + 1
    logs: list[TurnLogEntry] = []

    if state.get("pending_user_questions"):
        logs.append(_log(state, "orchestrator", "Questions en attente d'un tour précédent, envoi direct."))
        return {"turn": turn, "turn_log": logs}

    fresh_items = [it for it in state["context_items"] if it.fresh]
    if fresh_items:
        ids = [it.id for it in fresh_items]
        cleared = [it.model_copy(update={"fresh": False}) if it.id in ids else it for it in state["context_items"]]
        logs.append(_log(state, "orchestrator", f"Ré-évaluation de {len(ids)} élément(s) frais.", item_ids=ids))
        return {
            "turn": turn,
            "current_mode": "fresh",
            "active_fresh_item_ids": ids,
            "context_items": cleared,
            "turn_log": logs,
        }

    limits = orch.apply_loop_limits({**state, "turn": turn})
    if limits.hit_limit:
        if limits.blocking_stop:
            logs.append(_log(state, "orchestrator", "Arrêt : lacunes bloquantes non résolues à la limite de tours."))
            return {"turn": turn, "done": True, "stop_reason": limits.stop_message, "turn_log": logs}
        downgraded_ids = set(limits.downgraded_gap_ids)
        gaps = [
            g.model_copy(update={"status": "deferred"}) if g.id in downgraded_ids else g for g in state["gaps"]
        ]
        logs.append(
            _log(state, "orchestrator", f"Limite de tours atteinte : {len(downgraded_ids)} lacune(s) reportée(s).")
        )
        return {"turn": turn, "gaps": gaps, "current_mode": "section", "current_section_id": None, "turn_log": logs}

    next_section = orch.pick_next_section({**state, "turn": turn})
    if next_section is None:
        logs.append(_log(state, "orchestrator", "Toutes les sections requises sont complètes. Passage à la synthèse."))
        return {"turn": turn, "current_section_id": None, "turn_log": logs}

    logs.append(_log(state, "orchestrator", f"Section ciblée ce tour : {next_section}."))
    return {"turn": turn, "current_mode": "section", "current_section_id": next_section, "turn_log": logs}


def route_after_orchestrator(state: CDCState) -> str:
    if state.get("done"):
        return "end_stopped"
    if state.get("pending_user_questions"):
        return "human_input"
    if state.get("current_mode") == "fresh":
        return "gap_finder"
    if state.get("current_section_id") is None:
        return "synthesizer"
    return "gap_finder"


def gap_finder_node(state: CDCState) -> dict:
    mode = state.get("current_mode", "section")
    if mode == "fresh":
        result = gap_finder_agent.run_gap_finder(state, mode="fresh", fresh_item_ids=state.get("active_fresh_item_ids", []))
    else:
        section_id = state["current_section_id"]
        result = gap_finder_agent.run_gap_finder(state, mode="section", section_id=section_id)

    gaps = list(state["gaps"])
    resolved_ids = set(result.resolved_gap_ids)
    gaps = [g.model_copy(update={"status": "resolved"}) if g.id in resolved_ids else g for g in gaps]
    gaps.extend(result.new_gaps)

    updates: dict = {"gaps": gaps, "active_gap_ids": [g.id for g in result.new_gaps]}
    logs = [
        _log(
            state,
            "gap_finder",
            f"{len(result.new_gaps)} nouvelle(s) lacune(s), {len(resolved_ids)} résolue(s).",
            mode=mode,
        )
    ]

    if mode == "section" and result.section_complete is not None:
        section_id = state["current_section_id"]
        statuses = dict(state["section_statuses"])
        new_status = "complete" if result.section_complete else "in_progress"
        statuses[section_id] = SectionStatus(section_id=section_id, status=new_status)
        updates["section_statuses"] = statuses
        logs.append(_log(state, "gap_finder", f"Section {section_id} -> statut {new_status}."))

    updates["turn_log"] = logs
    return updates


def gap_filler_node(state: CDCState) -> dict:
    turn = state.get("turn", 0)
    gap_ids = state.get("active_gap_ids", [])
    result = gap_filler_agent.fill_gaps(state, gap_ids, turn)

    context_items = list(state["context_items"]) + result.new_context_items
    gaps = list(state["gaps"])
    gaps_by_id = {g.id: g for g in gaps}
    for gap_id, new_status in result.gap_updates.items():
        gaps_by_id[gap_id] = gaps_by_id[gap_id].model_copy(update={"status": new_status})
    gaps = list(gaps_by_id.values())

    max_per_gap = state["loop_settings"].max_questions_per_gap
    to_queue: list[PendingQuestion] = []
    auto_assumptions: list[ContextItem] = []

    for pq in result.pending_questions:
        gap = gaps_by_id[pq.gap_id]
        if gap.questions_asked >= max_per_gap:
            item = gap_filler_agent.build_assumption(state, gap, turn)
            auto_assumptions.append(item)
            gaps_by_id[gap.id] = gap.model_copy(update={"status": "assumed"})
            continue

        verdict = orch.dedup_gate(state, gap, pq.text)
        if verdict.already_resolved and verdict.resolved_by_item_id:
            gaps_by_id[gap.id] = gap.model_copy(
                update={"status": "resolved", "answer_item_ids": gap.answer_item_ids + [verdict.resolved_by_item_id]}
            )
            continue

        text = verdict.rewritten_question if verdict.partially_resolved and verdict.rewritten_question else pq.text
        to_queue.append(PendingQuestion(gap_id=gap.id, text=text))

    gaps = list(gaps_by_id.values())
    context_items = context_items + auto_assumptions

    max_batch = state["loop_settings"].max_questions_per_batch
    this_batch, remainder = orch.batch_questions(to_queue, max_batch)

    for pq in this_batch:
        g = gaps_by_id[pq.gap_id]
        gaps_by_id[pq.gap_id] = g.model_copy(update={"questions_asked": g.questions_asked + 1})
    gaps = list(gaps_by_id.values())

    rag_ids = [it.id for it in result.new_context_items]
    assumption_ids = [it.id for it in auto_assumptions]

    return {
        "context_items": context_items,
        "gaps": gaps,
        "pending_user_questions": this_batch + remainder,
        "active_fresh_item_ids": rag_ids + assumption_ids,
        "turn_log": [
            _log(
                state,
                "gap_filler",
                f"{len(result.new_context_items)} réponse(s) RAG, {len(this_batch)} question(s) envoyée(s), "
                f"{len(auto_assumptions)} hypothèse(s) auto (limite atteinte).",
            )
        ],
    }


def route_after_gap_filler(state: CDCState) -> str:
    return "human_input" if state.get("pending_user_questions") else "critic"


def human_input_node(state: CDCState) -> dict:
    batch = state["pending_user_questions"]
    to_ask = [{"gap_id": pq.gap_id, "text": pq.text} for pq in batch]
    answers: dict = interrupt({"questions": to_ask})
    return {"_raw_answers": answers}  # not part of typed state, consumed immediately by integrate_answers


def integrate_answers_node(state: CDCState) -> dict:
    turn = state.get("turn", 0)
    answers = state.get("_raw_answers", {}) or {}
    batch = state["pending_user_questions"]

    gaps_by_id = {g.id: g for g in state["gaps"]}
    new_items: list[ContextItem] = []
    asked_this_turn = []

    for pq in batch:
        raw = answers.get(pq.gap_id, {})
        skipped = bool(raw.get("skip")) if isinstance(raw, dict) else False
        text = (raw.get("text") if isinstance(raw, dict) else str(raw)) or ""
        gap = gaps_by_id[pq.gap_id]

        asked_this_turn.append(orch.record_asked_questions(state, [pq], turn)[0])

        if skipped or not text.strip():
            item = gap_filler_agent.build_assumption(state, gap, turn)
            gaps_by_id[gap.id] = gap.model_copy(update={"status": "assumed"})
        else:
            item = ContextItem(
                id=new_id("ctx"),
                content=text.strip(),
                source="user_answer",
                section_ids=gap.section_ids,
                linked_gap_id=gap.id,
                turn_added=turn,
                fresh=True,
            )
            gaps_by_id[gap.id] = gap.model_copy(
                update={"status": "user_answered", "answer_item_ids": gap.answer_item_ids + [item.id]}
            )
        new_items.append(item)

    return {
        "context_items": list(state["context_items"]) + new_items,
        "gaps": list(gaps_by_id.values()),
        "asked_questions": list(state["asked_questions"]) + asked_this_turn,
        "pending_user_questions": [],
        "active_fresh_item_ids": list(set(state.get("active_fresh_item_ids", []) + [it.id for it in new_items])),
        "turn_log": [_log(state, "integrate_answers", f"{len(new_items)} réponse(s) intégrée(s).")],
    }


def critic_node(state: CDCState) -> dict:
    fresh_ids = state.get("active_fresh_item_ids", [])
    result = critic_agent.run_critic(state, fresh_ids)

    gaps = list(state["gaps"]) + result.new_gaps
    statuses = dict(state["section_statuses"])
    for sid, reason in result.reopened_sections.items():
        statuses[sid] = SectionStatus(section_id=sid, status="reopened", reopen_reason=reason)

    logs = [_log(state, "critic", f"{len(result.new_gaps)} incohérence(s) détectée(s), {len(result.reopened_sections)} section(s) rouverte(s).")]
    return {"gaps": gaps, "section_statuses": statuses, "turn_log": logs}


def synthesizer_node(state: CDCState) -> dict:
    qmd_text, mapped = synthesizer_agent.run_synthesis(state)
    return {
        "_qmd_text": qmd_text,
        "_mapped_items": mapped,
        "turn_log": [_log(state, "synthesizer", "Document CDC final généré (.qmd).")],
    }


def final_validator_node(state: CDCState) -> dict:
    qmd_text = state.get("_qmd_text", "")
    mapped = state.get("_mapped_items", {})
    unmapped = final_validator_agent.find_unmapped_context(state, mapped)
    contradictions = final_validator_agent.find_final_contradictions(qmd_text) if qmd_text else []
    report_path = final_validator_agent.write_qa_report(state, unmapped, contradictions)
    return {
        "done": True,
        "turn_log": [
            _log(
                state,
                "final_validator",
                f"QA finale : {len(unmapped)} élément(s) non intégré(s), {len(contradictions)} incohérence(s). Rapport: {report_path}",
            )
        ],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(CDCState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("gap_finder", gap_finder_node)
    graph.add_node("gap_filler", gap_filler_node)
    graph.add_node("human_input", human_input_node)
    graph.add_node("integrate_answers", integrate_answers_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("final_validator", final_validator_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "gap_finder": "gap_finder",
            "human_input": "human_input",
            "synthesizer": "synthesizer",
            "end_stopped": END,
        },
    )

    graph.add_edge("gap_finder", "gap_filler")
    graph.add_conditional_edges(
        "gap_filler",
        route_after_gap_filler,
        {"human_input": "human_input", "critic": "critic"},
    )
    graph.add_edge("human_input", "integrate_answers")
    graph.add_edge("integrate_answers", "critic")
    graph.add_edge("critic", "orchestrator")

    graph.add_edge("synthesizer", "final_validator")
    graph.add_edge("final_validator", END)

    return graph.compile(checkpointer=MemorySaver())
