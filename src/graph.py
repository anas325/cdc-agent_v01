"""LangGraph wiring for the CDC refinement agent swarm.

START -> ingest -> initial_scan -> orchestrator -> (route)
    -> gap_finder -> gap_filler -> (route: pending questions?)
         -> human_input -> integrate_answers -> critic -> orchestrator
         -> critic -> orchestrator
    -> synthesizer -> final_validator -> END
    -> END (blocking gaps hit max_turns)
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents import critic as critic_agent
from src.agents import final_validator as final_validator_agent
from src.agents import gap_filler as gap_filler_agent
from src.agents import gap_finder as gap_finder_agent
from src.agents import orchestrator as orch
from src.agents import synthesizer as synthesizer_agent
from src import telemetry
from src.config import load_sections, load_settings
from src.llm import map_structured
from src.ids import stable_id
from src.rag import ingest_source_docs
from src.state import CDCState, ContextItem, SectionStatus, TurnLogEntry
from src.utils.cdc_sections import split_cdc_by_sections


def _log(state: CDCState, agent: str, summary: str, **details) -> TurnLogEntry:
    # Stamp how long the enclosing node had been running when it logged, so the
    # turn log carries timing without every node having to measure it.
    elapsed = telemetry.current_node_elapsed()
    if elapsed is not None:
        details = {**details, "_elapsed_s": round(elapsed, 3)}
    return TurnLogEntry(turn=state.get("turn", 0), agent=agent, summary=summary, details=details)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def ingest_node(state: CDCState) -> dict:
    """Load config, ingest RAG docs, and seed context from the initial CDC.

    The CDC is split per section (see split_cdc_by_sections) so each section
    gets its own context item — downstream prompts then only carry the slices
    they need instead of the whole document. Unmatched chunks (preamble,
    unrecognized headings) become untagged items (section_ids=[]): visible in
    every LLM context, but not synthesized into any section — the final
    validator reports them as unmapped. If no section heading is recognized,
    the whole document is kept as one item tagged with all sections.
    """
    settings = load_settings()
    sections = load_sections()
    ingest_source_docs()

    initial_text = state.get("initial_cdc_text", "") or ""
    context_items: list[ContextItem] = []
    ingest_details: dict = {}
    if initial_text.strip():
        split = split_cdc_by_sections(initial_text, sections)
        if split.matched_any:
            for sid, text in split.sections.items():
                context_items.append(
                    ContextItem(
                        id=stable_id("ctx", sid, text),
                        content=text,
                        source="initial_cdc",
                        section_ids=[sid],
                        turn_added=0,
                        fresh=False,
                    )
                )
            # The index keeps two byte-identical unmatched chunks distinct
            # while still producing the same ids on a re-run.
            for idx, chunk in enumerate(split.unmatched):
                context_items.append(
                    ContextItem(
                        id=stable_id("ctx", "unmatched", str(idx), chunk),
                        content=chunk,
                        source="initial_cdc",
                        section_ids=[],
                        turn_added=0,
                        fresh=False,
                    )
                )
            ingest_details = {
                "matched_sections": list(split.sections),
                "unmatched_chunks": len(split.unmatched),
            }
        else:
            context_items.append(
                ContextItem(
                    id=stable_id("ctx", "whole", initial_text.strip()),
                    content=initial_text.strip(),
                    source="initial_cdc",
                    section_ids=[s.id for s in sections],
                    turn_added=0,
                    fresh=False,
                )
            )
            ingest_details = {"matched_sections": [], "unmatched_chunks": 0}

    incoming_statuses = state.get("section_statuses") or {}
    section_statuses = {}
    for section in sections:
        existing = incoming_statuses.get(section.id)
        if existing is not None:
            section_statuses[section.id] = existing
        else:
            section_statuses[section.id] = SectionStatus(section_id=section.id, status="empty")

    return {
        "sections_config": sections,
        "loop_settings": state.get("loop_settings") or settings.loop,
        "context_items": context_items,
        "gaps": [],
        "section_statuses": section_statuses,
        "asked_questions": [],
        "turn_log": [
            _log(
                state,
                "ingest",
                f"Chargement du CDC initial et ingestion RAG ({len(sections)} sections).",
                **ingest_details,
            )
        ],
        "turn": 0,
        "pending_user_questions": [],
        "done": False,
        "stop_reason": None,
    }


def initial_scan_node(state: CDCState) -> dict:
    """Un passage de gap_finder par section, pour obtenir une note initiale partout.

    Lacunes uniquement : les section_statuses sont volontairement laissés
    intacts, pour que la boucle orchestrateur traite ensuite chaque section
    normalement.
    """
    statuses = state.get("section_statuses") or {}
    gaps = list(state.get("gaps") or [])
    logs: list[TurnLogEntry] = []

    # Ces appels par section sont indépendants : on les lance en parallèle (le
    # gros poste de temps du run). Contrairement à la boucle série d'origine,
    # chaque appel voit le même instantané de gaps (pré-scan), donc la consigne
    # "ne pas dupliquer" ne couvre plus les sections entre elles — compromis
    # accepté : le critic / dedup_gate en aval rattrapent les doublons.
    scan_sections = [
        sec
        for sec in state["sections_config"]
        if not (statuses.get(sec.id) is not None and statuses[sec.id].status == "skipped")
    ]
    base = {**state, "gaps": gaps}
    results = map_structured(
        [partial(gap_finder_agent.run_gap_finder, base, mode="section", section_id=sec.id) for sec in scan_sections]
    )
    for sec, result in zip(scan_sections, results):
        gaps.extend(result.new_gaps)
        logs.append(
            _log(
                state,
                "initial_scan",
                f"Scan initial {sec.id} : {len(result.new_gaps)} lacune(s).",
                section_id=sec.id,
            )
        )

    logs.append(_log(state, "initial_scan", f"Scan initial terminé : {len(gaps)} lacune(s) au total."))
    return {"gaps": gaps, "turn_log": logs}


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

    updates: dict = {"gaps": gaps}
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
    settings = state["loop_settings"]
    result = gap_filler_agent.fill_gaps(
        state, turn, settings.max_questions_per_batch, settings.max_questions_per_gap
    )

    gaps_by_id = {g.id: g for g in state["gaps"]}
    for gap_id, new_status in result.gap_updates.items():
        update: dict = {"status": new_status}
        answering_item_id = result.resolved_by.get(gap_id)
        if answering_item_id:
            update["answer_item_ids"] = gaps_by_id[gap_id].answer_item_ids + [answering_item_id]
        gaps_by_id[gap_id] = gaps_by_id[gap_id].model_copy(update=update)

    for gap_id in result.rag_attempted_gap_ids:
        gaps_by_id[gap_id] = gaps_by_id[gap_id].model_copy(update={"rag_attempted": True})

    # Every returned question is asked this turn — fill_gaps stops at the batch cap.
    for pq in result.pending_questions:
        g = gaps_by_id[pq.gap_id]
        gaps_by_id[pq.gap_id] = g.model_copy(update={"questions_asked": g.questions_asked + 1})

    return {
        "context_items": list(state["context_items"]) + result.new_context_items,
        "gaps": list(gaps_by_id.values()),
        "pending_user_questions": result.pending_questions,
        "active_fresh_item_ids": [it.id for it in result.new_context_items],
        "turn_log": [
            _log(
                state,
                "gap_filler",
                f"{len(result.new_context_items)} élément(s) auto-résolu(s) (RAG/hypothèse), "
                f"{len(result.pending_questions)} question(s) envoyée(s).",
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

    # The user clicked "skip section" instead of answering: mark that section
    # skipped and defer its still-open gaps (those left attached only to skipped
    # sections) so the orchestrator moves on to the next section next turn.
    skip_section_id = answers.get("__skip_section__") if isinstance(answers, dict) else None
    if skip_section_id:
        return _skip_section(state, skip_section_id)

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
                # Stable so that re-running with the same answers keeps
                # hitting the LLM cache past the first question batch.
                id=stable_id("ctx", "answer", gap.id, str(turn), text.strip()),
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


def _skip_section(state: CDCState, section_id: str) -> dict:
    statuses = dict(state["section_statuses"])
    statuses[section_id] = SectionStatus(section_id=section_id, status="skipped")
    skipped_ids = {sid for sid, ss in statuses.items() if ss.status == "skipped"}

    deferred = 0
    gaps = []
    for g in state["gaps"]:
        # Defer an open gap only if every section it still hangs off is skipped
        # (or it has none): a contradiction shared with an active section stays
        # open and gets handled when that section's turn comes.
        if g.status == "open" and section_id in g.section_ids and all(
            sid in skipped_ids for sid in g.section_ids
        ):
            gaps.append(g.model_copy(update={"status": "deferred"}))
            deferred += 1
        else:
            gaps.append(g)

    return {
        "gaps": gaps,
        "section_statuses": statuses,
        "pending_user_questions": [],
        "active_fresh_item_ids": [],
        "turn_log": [
            _log(
                state,
                "integrate_answers",
                f"Section {section_id} ignorée par l'utilisateur ; {deferred} lacune(s) reportée(s).",
                section_id=section_id,
            )
        ],
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


def _timed(name: str, fn):
    """Wrap a node so src.telemetry records its wall clock and nested LLM calls.

    Applied at registration rather than as a decorator on each node function so
    the nodes stay directly callable (and testable) without telemetry.
    """

    def wrapper(state: CDCState) -> dict:
        with telemetry.record_node(name):
            return fn(state)

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def build_graph():
    graph = StateGraph(CDCState)

    graph.add_node("ingest", _timed("ingest", ingest_node))
    graph.add_node("initial_scan", _timed("initial_scan", initial_scan_node))
    graph.add_node("orchestrator", _timed("orchestrator", orchestrator_node))
    graph.add_node("gap_finder", _timed("gap_finder", gap_finder_node))
    graph.add_node("gap_filler", _timed("gap_filler", gap_filler_node))
    graph.add_node("human_input", _timed("human_input", human_input_node))
    graph.add_node("integrate_answers", _timed("integrate_answers", integrate_answers_node))
    graph.add_node("critic", _timed("critic", critic_node))
    graph.add_node("synthesizer", _timed("synthesizer", synthesizer_node))
    graph.add_node("final_validator", _timed("final_validator", final_validator_node))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "initial_scan")
    graph.add_edge("initial_scan", "orchestrator")

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
