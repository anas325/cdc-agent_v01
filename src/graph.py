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
from src.decisions import make_decision, now_iso
from src.llm import current_model_name, map_structured
from src.ids import stable_id
from src.rag import ingest_source_docs
from src.state import CDCState, ContextItem, DecisionLogEntry, Gap, SectionStatus, TurnLogEntry
from src.utils.cdc_sections import split_cdc_by_sections


def _log(state: CDCState, agent: str, summary: str, **details) -> TurnLogEntry:
    # Stamp how long the enclosing node had been running when it logged, so the
    # turn log carries timing without every node having to measure it.
    elapsed = telemetry.current_node_elapsed()
    if elapsed is not None:
        details = {**details, "_elapsed_s": round(elapsed, 3)}
    return TurnLogEntry(turn=state.get("turn", 0), agent=agent, summary=summary, details=details)


def _merge_gaps(existing: list[Gap], new: list[Gap]) -> list[Gap]:
    """Append only genuinely new gaps, keyed by id.

    Gap ids are content hashes, so an agent re-reporting the same finding yields
    the same id. Appending blindly let a duplicate reach the candidate pool and
    be asked twice in one batch. Matching against *every* status also stops a
    gap that was already resolved or settled by an assumption from being
    resurrected the moment an agent mentions it again.
    """
    seen = {g.id for g in existing}
    out = list(existing)
    for gap in new:
        if gap.id in seen:
            continue
        seen.add(gap.id)
        out.append(gap)
    return out


def _decide(state: CDCState, agent: str, decision_type: str, summary: str, **kw) -> DecisionLogEntry:
    """Sibling of _log for the machine-readable audit trail.

    Nodes that get decisions back from a pure agent just pass them through under
    the `decision_log` key; this is for the ones that decide inline.
    """
    return make_decision(state, agent=agent, decision_type=decision_type, summary=summary, **kw)


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

    Deuxième point d'entrée — mode « sans CDC initial » : l'auteur n'a pas de
    document et décrit chaque section au clavier (`initial_section_texts`).
    Le découpage est alors déjà fait, donc pas de reconnaissance de titres :
    un ContextItem par section décrite, même statut qu'un CDC déposé (c'est
    le texte de l'auteur) mais attribué à l'utilisateur. Les deux entrées se
    cumulent : un brouillon partiel peut être complété à la main. Une section
    laissée vide reste « empty » et sera construite par les questions.
    """
    settings = load_settings()
    sections = load_sections()
    ingest_source_docs()

    initial_text = state.get("initial_cdc_text", "") or ""
    # Ordonné par la config, pas par le dict reçu : les ids sont des hachages de
    # contenu, mais l'ordre des items est checkpointé et ne doit pas dépendre de
    # l'ordre de saisie dans l'UI. Les ids inconnus sont ignorés (sections = config).
    raw_declared = state.get("initial_section_texts") or {}
    declared = {
        sec.id: (raw_declared.get(sec.id) or "").strip()
        for sec in sections
        if (raw_declared.get(sec.id) or "").strip()
    }
    context_items: list[ContextItem] = []
    ingest_details: dict = {}
    # The initial CDC is the author's own text: nothing was inferred, so it is
    # accepted as-is with full confidence and attributed to the system (the
    # splitter), not to an LLM — or to the user directly when they typed the
    # section themselves rather than uploading a document.
    initial_provenance = {
        "created_by": "system",
        "timestamp": now_iso(),
        "confidence": 1.0,
        "validation_status": "accepted",
    }
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
                        **initial_provenance,
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
                        **initial_provenance,
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
                    **initial_provenance,
                )
            )
            ingest_details = {"matched_sections": [], "unmatched_chunks": 0}

    for sid, text in declared.items():
        context_items.append(
            ContextItem(
                id=stable_id("ctx", "declared", sid, text),
                content=text,
                source="initial_cdc",
                section_ids=[sid],
                turn_added=0,
                fresh=False,
                **{**initial_provenance, "created_by": "user"},
            )
        )
    if declared:
        ingest_details["declared_sections"] = list(declared)

    if declared and not initial_text.strip():
        ingest_summary = (
            f"Démarrage sans CDC initial : {len(declared)} section(s) décrite(s) par l'auteur, "
            "ingestion RAG."
        )
    else:
        ingest_summary = f"Chargement du CDC initial et ingestion RAG ({len(sections)} sections)."

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
                ingest_summary,
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
    decisions: list[DecisionLogEntry] = []

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
        # Les passes voient toutes le même instantané pré-scan : deux sections
        # peuvent donc remonter la même incohérence transversale.
        before = len(gaps)
        gaps = _merge_gaps(gaps, result.new_gaps)
        decisions.extend(result.decisions)
        logs.append(
            _log(
                state,
                "initial_scan",
                f"Scan initial {sec.id} : {len(gaps) - before} lacune(s).",
                section_id=sec.id,
            )
        )

    logs.append(_log(state, "initial_scan", f"Scan initial terminé : {len(gaps)} lacune(s) au total."))
    return {"gaps": gaps, "turn_log": logs, "decision_log": decisions}


def orchestrator_node(state: CDCState) -> dict:
    turn = state.get("turn", 0) + 1
    logs: list[TurnLogEntry] = []

    # Checked FIRST, before the pending-questions and fresh-item shortcuts.
    # Those two both return early, so evaluating the limit after them meant a
    # self-sustaining loop (each answer creating a fresh item, whose re-scan
    # creates the next question) never reached the limit at all — max_turns was
    # unenforceable exactly in the case it exists for.
    limits = orch.apply_loop_limits({**state, "turn": turn})
    if limits.hit_limit:
        at_limit = {**state, "turn": turn}
        if limits.blocking_stop:
            logs.append(_log(state, "orchestrator", "Arrêt : lacunes bloquantes non résolues à la limite de tours."))
            return {
                "turn": turn,
                "done": True,
                "stop_reason": limits.stop_message,
                "turn_log": logs,
                "decision_log": [
                    _decide(
                        at_limit,
                        "orchestrator",
                        "loop_limit_applied",
                        limits.stop_message,
                        outcome="blocking_stop",
                        max_turns=state["loop_settings"].max_turns,
                    )
                ],
            }
        downgraded_ids = set(limits.downgraded_gap_ids)
        gaps = [
            g.model_copy(update={"status": "deferred"}) if g.id in downgraded_ids else g for g in state["gaps"]
        ]
        logs.append(
            _log(state, "orchestrator", f"Limite de tours atteinte : {len(downgraded_ids)} lacune(s) reportée(s).")
        )
        return {
            "turn": turn,
            "gaps": gaps,
            "current_mode": "section",
            "current_section_id": None,
            "turn_log": logs,
            "decision_log": [
                _decide(
                    at_limit,
                    "orchestrator",
                    "loop_limit_applied",
                    f"Limite de {state['loop_settings'].max_turns} tours atteinte : "
                    f"{len(downgraded_ids)} lacune(s) non bloquante(s) reportée(s).",
                    input_ids=sorted(downgraded_ids),
                    outcome="deferred_non_blocking",
                )
            ],
        }

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
    gaps = _merge_gaps(gaps, result.new_gaps)

    updates: dict = {"gaps": gaps, "decision_log": list(result.decisions)}
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
        updates["decision_log"].append(
            _decide(
                state,
                "gap_finder",
                "section_status_changed",
                f"Section {section_id} -> statut {new_status}.",
                input_ids=[section_id],
                # Statut dérivé de façon déterministe (aucune lacune bloquante ou
                # importante ouverte), pas d'un jugement du LLM.
                rule="no_open_blocking_or_important_gaps",
                new_status=new_status,
            )
        )

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
        gaps_by_id[pq.gap_id] = g.model_copy(
            update={
                "questions_asked": g.questions_asked + 1,
                # Keep the question on the gap itself, so a gap record is
                # self-contained for audit even without the asked_questions list.
                "question_text": result.question_texts.get(pq.gap_id, pq.text),
            }
        )

    return {
        "context_items": list(state["context_items"]) + result.new_context_items,
        "gaps": list(gaps_by_id.values()),
        "pending_user_questions": result.pending_questions,
        "active_fresh_item_ids": [it.id for it in result.new_context_items],
        "decision_log": list(result.decisions),
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
    decisions: list[DecisionLogEntry] = []
    # gap.conflicting_item_ids of contradictions the user has just settled ->
    # the answer that settles them. See _supersede_losing_assumptions.
    settled_by: dict[str, str] = {}

    for pq in batch:
        raw = answers.get(pq.gap_id, {})
        skipped = bool(raw.get("skip")) if isinstance(raw, dict) else False
        text = (raw.get("text") if isinstance(raw, dict) else str(raw)) or ""
        gap = gaps_by_id[pq.gap_id]

        asked_this_turn.append(orch.record_asked_questions(state, [pq], turn)[0])

        if skipped or not text.strip():
            item = gap_filler_agent.build_assumption(state, gap, turn)
            gaps_by_id[gap.id] = gap.model_copy(update={"status": "assumed"})
            decisions.append(
                _decide(
                    state,
                    "integrate_answers",
                    "assumption_built",
                    f"L'utilisateur n'a pas répondu à {gap.id} : hypothèse par défaut retenue.",
                    prompt_id="gap_filler.assumption",
                    input_ids=[gap.id],
                    output_ids=[item.id],
                    confidence=item.confidence,
                    model=item.model,
                    reason="user_skipped" if skipped else "empty_answer",
                )
            )
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
                # A human stakeholder answered directly: authoritative by
                # definition, so no LLM model/prompt is attributed to it.
                created_by="user",
                timestamp=now_iso(),
                confidence=1.0,
                validation_status="accepted",
            )
            gaps_by_id[gap.id] = gap.model_copy(
                update={"status": "user_answered", "answer_item_ids": gap.answer_item_ids + [item.id]}
            )
            if gap.category == "contradiction":
                for cid in gap.conflicting_item_ids:
                    settled_by.setdefault(cid, item.id)
            decisions.append(
                _decide(
                    state,
                    "integrate_answers",
                    "answer_integrated",
                    f"Réponse utilisateur intégrée pour {gap.id}.",
                    input_ids=[gap.id],
                    output_ids=[item.id],
                    confidence=1.0,
                    question=pq.text,
                )
            )
        new_items.append(item)

    context_items, supersede_decisions = _supersede_losing_assumptions(state, settled_by)
    decisions.extend(supersede_decisions)

    summary = f"{len(new_items)} réponse(s) intégrée(s)."
    if supersede_decisions:
        summary += f" {len(supersede_decisions)} hypothèse(s) contredite(s) retirée(s) du contexte."

    return {
        "context_items": context_items + new_items,
        "gaps": list(gaps_by_id.values()),
        "asked_questions": list(state["asked_questions"]) + asked_this_turn,
        "pending_user_questions": [],
        # Sorted, not set-ordered: this is checkpointed state and must not vary
        # between two otherwise identical runs.
        "active_fresh_item_ids": sorted(
            set(state.get("active_fresh_item_ids", []) + [it.id for it in new_items])
        ),
        "decision_log": decisions,
        "turn_log": [_log(state, "integrate_answers", summary)],
    }


def _supersede_losing_assumptions(
    state: CDCState, settled_by: dict[str, str]
) -> tuple[list[ContextItem], list[DecisionLogEntry]]:
    """Retire the assumptions a user answer has just overruled.

    Without this, answering a contradiction only *adds* the correct value beside
    the wrong one: the ASSUMPTION stays live context, the critic re-detects the
    same conflict against the new answer, and the swarm asks the same question
    every turn until it runs out of them.

    The rule is deterministic and needs no LLM call, because the ranking is
    already in the model: a user answer is confidence=1.0 / accepted, while an
    assumption is a stopgap that is always `needs_review`. So a user answer to a
    contradiction supersedes every *assumption* that contradiction cites. Nothing
    else is touched — two conflicting user answers, or a conflict with the
    initial CDC, is a real editorial decision and stays open for the author.
    """
    if not settled_by:
        return list(state["context_items"]), []

    items: list[ContextItem] = []
    decisions: list[DecisionLogEntry] = []
    for it in state["context_items"]:
        winner = settled_by.get(it.id)
        if winner is None or it.source != "assumption" or it.superseded_by is not None:
            items.append(it)
            continue
        # Kept in state, not deleted: the audit trail must still be able to show
        # what the system had assumed and what replaced it.
        items.append(it.model_copy(update={"superseded_by": winner, "validation_status": "rejected"}))
        decisions.append(
            _decide(
                state,
                "integrate_answers",
                "assumption_superseded",
                f"Hypothèse {it.id} retirée : la réponse utilisateur {winner} tranche la contradiction.",
                input_ids=[it.id],
                output_ids=[winner],
                rule="user_answer_beats_assumption",
            )
        )
    return items, decisions


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

    summary = f"Section {section_id} ignorée par l'utilisateur ; {deferred} lacune(s) reportée(s)."
    return {
        "gaps": gaps,
        "section_statuses": statuses,
        "pending_user_questions": [],
        "active_fresh_item_ids": [],
        "turn_log": [_log(state, "integrate_answers", summary, section_id=section_id)],
        "decision_log": [
            _decide(
                state,
                "integrate_answers",
                "section_status_changed",
                summary,
                input_ids=[section_id],
                new_status="skipped",
                deferred_gaps=deferred,
                decided_by="user",
            )
        ],
    }


def critic_node(state: CDCState) -> dict:
    fresh_ids = state.get("active_fresh_item_ids", [])
    result = critic_agent.run_critic(state, fresh_ids)

    gaps = _merge_gaps(list(state["gaps"]), result.new_gaps)
    statuses = dict(state["section_statuses"])
    for sid, reason in result.reopened_sections.items():
        statuses[sid] = SectionStatus(section_id=sid, status="reopened", reopen_reason=reason)

    logs = [_log(state, "critic", f"{len(result.new_gaps)} incohérence(s) détectée(s), {len(result.reopened_sections)} section(s) rouverte(s).")]
    return {
        "gaps": gaps,
        "section_statuses": statuses,
        "turn_log": logs,
        "decision_log": list(result.decisions),
    }


def synthesizer_node(state: CDCState) -> dict:
    qmd_text, mapped = synthesizer_agent.run_synthesis(state)
    return {
        "_qmd_text": qmd_text,
        "_mapped_items": mapped,
        "turn_log": [_log(state, "synthesizer", "Document CDC final généré (.qmd).")],
        "decision_log": [
            _decide(
                state,
                "synthesizer",
                "section_synthesized",
                f"Section {sid} rédigée à partir de {len(item_ids)} élément(s) de contexte.",
                prompt_id="synthesizer.slot",
                input_ids=item_ids,
                output_ids=[sid],
                model=current_model_name(),
            )
            for sid, item_ids in mapped.items()
        ],
    }


def final_validator_node(state: CDCState) -> dict:
    qmd_text = state.get("_qmd_text", "")
    mapped = state.get("_mapped_items", {})
    unmapped = final_validator_agent.find_unmapped_context(state, mapped)
    contradictions = final_validator_agent.find_final_contradictions(qmd_text) if qmd_text else []
    report_path = final_validator_agent.write_qa_report(state, unmapped, contradictions)

    decision = _decide(
        state,
        "final_validator",
        "final_check",
        f"QA finale : {len(unmapped)} élément(s) non intégré(s), {len(contradictions)} incohérence(s).",
        prompt_id="final_validator.contradictions" if qmd_text else None,
        input_ids=[it.id for it in unmapped],
        model=current_model_name() if qmd_text else None,
        contradictions=contradictions,
        qa_report=str(report_path),
    )
    # Written last, and with this node's own decision included, so the dump is
    # the complete trail rather than everything-but-the-final-check.
    log_path = final_validator_agent.write_decision_log(
        {**state, "decision_log": list(state.get("decision_log") or []) + [decision]}
    )
    return {
        "done": True,
        "decision_log": [decision],
        "turn_log": [
            _log(
                state,
                "final_validator",
                f"QA finale : {len(unmapped)} élément(s) non intégré(s), {len(contradictions)} incohérence(s). "
                f"Rapport: {report_path} · Journal de décisions: {log_path}",
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


def build_graph(checkpointer=None):
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

    return graph.compile(checkpointer=checkpointer or MemorySaver())
