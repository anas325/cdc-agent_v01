"""Streamlit test interface for the CDC refinement agent swarm.

Single-session, single-run-at-a-time debug tool: upload an initial CDC and/or
RAG source docs, start a run, answer batched questions as they come up, and
download the final document once the graph reaches END. All durable state
lives in the LangGraph checkpointer (keyed by thread_id) — every rerun
re-reads it instead of trusting in-memory globals.

Two tabs: "Pilotage" drives the run, "Sous le capot" exposes timings (from
src.telemetry), raw gaps, the turn log, and the checkpointer snapshot.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# transformers' optional vision submodules import torchvision (not installed);
# Streamlit's file watcher probes every loaded module and logs that as a warning.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
import streamlit_authenticator as stauth
from langgraph.types import Command

from src import db, telemetry
from src.config import load_sections, load_settings
from src.decisions import confidence_band, format_evidence
from src.graph import build_graph
from src.quality import score_section
from src.state import LoopSettings, SectionStatus
from src.telemetry import format_duration

SOURCE_DOCS_DIR = ROOT_DIR / "data" / "source_docs"

STATUS_ICONS = {
    "empty": "⚪",
    "in_progress": "🟡",
    "complete": "🟢",
    "reopened": "🔴",
    "skipped": "⚫",
}

SEVERITIES = ["blocking", "important", "nice_to_have"]
SEVERITY_LABELS = {"blocking": "bloquante", "important": "importante", "nice_to_have": "optionnelle"}
SEVERITY_BADGE = {"blocking": "red", "important": "orange", "nice_to_have": "gray"}
SEVERITY_ORDER = {"blocking": 0, "important": 1, "nice_to_have": 2}
# Gap-type ordering: contradictions first, then by descending impact. Mirrors
# _CATEGORY_ORDER in src.agents.gap_filler so the displayed order matches the
# order gaps are actually pulled from the pool.
CATEGORY_ORDER = {
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

# Gap statuses that mean "no longer a hole in the CDC".
CLOSED_STATUSES = {"rag_answered", "user_answered", "assumed", "resolved"}
STATUS_BADGE = {
    "open": "red",
    "deferred": "orange",
    "assumed": "violet",
    "rag_answered": "blue",
    "user_answered": "green",
    "resolved": "green",
}

# Provenance / audit trail (phase 1). Confidence bands come from
# src.decisions so the UI and the acceptance policy can never drift apart.
SOURCE_LABELS = {
    "initial_cdc": "CDC initial",
    "user_answer": "réponse utilisateur",
    "rag": "documentation",
    "assumption": "hypothèse",
}
SOURCE_BADGE = {
    "initial_cdc": "gray",
    "user_answer": "green",
    "rag": "blue",
    "assumption": "violet",
}
VALIDATION_LABELS = {
    "accepted": "acceptée",
    "needs_review": "à relire",
    "rejected": "rejetée",
    "unreviewed": "non revue",
}
VALIDATION_BADGE = {
    "accepted": "green",
    "needs_review": "orange",
    "rejected": "red",
    "unreviewed": "gray",
}
CONFIDENCE_BADGE = {"high": "green", "medium": "orange", "low": "red", "unknown": "gray"}

st.set_page_config(page_title="CDC Refinement Agent", layout="wide")


def _read_secret(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val


def _get_authenticator() -> stauth.Authenticate:
    """Build the authenticator from the closed, admin-managed user list.

    Credentials live in st.secrets ([credentials.usernames.*], bcrypt hashes).
    No registration widget is wired, so the user set stays closed. The JWT
    cookie keeps the login alive across reloads, new tabs, and restarts.

    Rebuilt each run (not cached): the Authenticate object owns a per-session
    cookie manager, so it must not be shared across sessions. Rebuilding is
    cheap since the passwords are already hashed.
    """
    key = _read_secret("COOKIE_KEY")
    try:
        usernames = dict(st.secrets["credentials"]["usernames"])
    except Exception:
        usernames = {}
    if not key or not usernames:
        st.error("Auth non configurée (credentials / COOKIE_KEY dans les secrets).")
        st.stop()
    credentials = {"usernames": {u: dict(v) for u, v in usernames.items()}}
    return stauth.Authenticate(credentials, "cdc_agent_auth", key, cookie_expiry_days=30)


authenticator = _get_authenticator()
authenticator.login(location="main")

if st.session_state.get("authentication_status") is not True:
    if st.session_state.get("authentication_status") is False:
        st.error("Identifiants incorrects.")
    st.stop()

USERNAME = st.session_state["username"]


@st.cache_resource
def get_graph():
    return build_graph(checkpointer=db.get_checkpointer())


def init_session_state() -> None:
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("run_active", False)
    st.session_state.setdefault("pending_questions", None)
    st.session_state.setdefault("finished", False)
    st.session_state.setdefault("last_step_seconds", None)
    st.session_state.setdefault("step_timeline", [])


def current_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ---------------------------------------------------------------------------
# Running the graph
# ---------------------------------------------------------------------------


def run_graph(payload) -> None:
    """Stream one graph step to the next interrupt or END, live-logging nodes.

    Uses stream_mode="updates" rather than invoke() so each node reports as it
    completes — the first step alone runs one LLM call per section inside
    initial_scan, which used to sit behind a single opaque spinner.
    """
    st.session_state.run_active = True
    graph = get_graph()
    step_started = time.perf_counter()
    seen_before = len(telemetry.node_runs())
    interrupts = None
    last_values: dict = {}

    with st.status("Exécution de l'agent swarm…", expanded=True) as status:
        try:
            for update in graph.stream(payload, current_config(), stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupts = update["__interrupt__"]
                    continue
                for node_name, node_values in update.items():
                    runs = telemetry.node_runs()
                    duration = runs[-1].duration_s if len(runs) > seen_before else None
                    suffix = f" — {format_duration(duration)}" if duration is not None else ""
                    st.write(f":material/check: `{node_name}`{suffix}")
                    seen_before = len(runs)
                    if isinstance(node_values, dict):
                        last_values = node_values
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, this is a debug tool
            st.session_state.run_active = False
            status.update(label=f"Échec après {format_duration(time.perf_counter() - step_started)}", state="error")
            st.exception(exc)
            return

        elapsed = time.perf_counter() - step_started
        status.update(label=f"Terminé en {format_duration(elapsed)}", state="complete", expanded=False)

    st.session_state.last_step_seconds = elapsed
    st.session_state.step_timeline = st.session_state.step_timeline + [elapsed]

    # interrupt() surfaces in the stream; fall back to the checkpointer's
    # pending tasks in case a future LangGraph version stops emitting it.
    if interrupts is None:
        tasks = getattr(graph.get_state(current_config()), "tasks", ()) or ()
        interrupts = [i for task in tasks for i in (getattr(task, "interrupts", ()) or ())] or None

    process_step_result({"__interrupt__": interrupts, **last_values})


def process_step_result(result: dict) -> None:
    interrupts = result.get("__interrupt__")
    if interrupts:
        st.session_state.pending_questions = interrupts[0].value.get("questions", [])
        st.session_state.finished = False
        st.session_state.run_active = False
        status = "awaiting_input"
    else:
        st.session_state.pending_questions = None
        st.session_state.finished = bool(result.get("done"))
        st.session_state.run_active = False
        status = "finished" if st.session_state.finished else "running"

    if st.session_state.thread_id:
        try:
            db.touch_run(st.session_state.thread_id, status)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort here
            st.warning(f"Impossible de mettre à jour l'historique du run : {exc}")


def get_state_values() -> dict:
    if not st.session_state.thread_id:
        return {}
    snapshot = get_graph().get_state(current_config())
    return snapshot.values or {}


def load_run_into_session(thread_id: str) -> None:
    """Reopen a persisted run: point the session at its thread and rebuild the
    UI flags from the durable checkpoint so it renders where it left off."""
    st.session_state.thread_id = thread_id
    st.session_state.run_active = False
    st.session_state.step_timeline = []
    st.session_state.last_step_seconds = None

    snapshot = get_graph().get_state(current_config())
    values = snapshot.values or {}
    tasks = getattr(snapshot, "tasks", ()) or ()
    interrupts = [i for task in tasks for i in (getattr(task, "interrupts", ()) or ())]

    if interrupts:
        st.session_state.pending_questions = interrupts[0].value.get("questions", [])
        st.session_state.finished = False
    elif values.get("pending_user_questions"):
        # Fallback if the interrupt isn't surfaced as a pending task.
        st.session_state.pending_questions = [
            {"gap_id": pq.gap_id, "text": pq.text}
            for pq in values["pending_user_questions"]
        ]
        st.session_state.finished = False
    else:
        st.session_state.pending_questions = None
        st.session_state.finished = bool(values.get("done"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_runs_history() -> None:
    """List the current user's runs; clicking one reopens it from Postgres."""
    st.sidebar.subheader("Mes runs")
    try:
        runs = db.list_runs(USERNAME)
    except Exception as exc:  # noqa: BLE001 - history is a convenience, not critical
        st.sidebar.warning(f"Historique indisponible : {exc}")
        return
    if not runs:
        st.sidebar.caption("Aucun run enregistré pour l'instant.")
        return
    status_icon = {"running": "🟡", "awaiting_input": "🟠", "finished": "🟢"}
    for run in runs:
        tid = str(run["thread_id"])
        icon = status_icon.get(run.get("status"), "⚪")
        label = f"{icon} {run.get('title') or tid[:8]}"
        active = tid == st.session_state.get("thread_id")
        if st.sidebar.button(
            label, key=f"open_run_{tid}", width="stretch",
            disabled=active, type="secondary",
        ):
            load_run_into_session(tid)
            st.rerun()


def render_sidebar(settings) -> tuple[str, LoopSettings, bool, dict[str, bool]]:
    st.sidebar.caption(f"Connecté : **{st.session_state.get('name', USERNAME)}**")
    authenticator.logout("Se déconnecter", location="sidebar")
    render_runs_history()
    st.sidebar.divider()
    st.sidebar.header("Configuration")

    cdc_file = st.sidebar.file_uploader("CDC initial (markdown/texte/PDF)", type=["md", "txt", "pdf"])
    cdc_text = ""
    if cdc_file is not None:
        st.session_state["_cdc_title"] = cdc_file.name
        if cdc_file.name.lower().endswith(".pdf"):
            from src.rag import extract_document_text

            SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = SOURCE_DOCS_DIR / cdc_file.name
            tmp_path.write_bytes(cdc_file.getvalue())
            cdc_text = extract_document_text(tmp_path)
        else:
            cdc_text = cdc_file.read().decode("utf-8", errors="ignore")

    st.sidebar.subheader("Documents source (RAG)")
    source_files = st.sidebar.file_uploader(
        "Ajouter des documents source", type=["md", "txt", "pdf"], accept_multiple_files=True
    )
    if source_files:
        SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in source_files:
            (SOURCE_DOCS_DIR / f.name).write_bytes(f.getvalue())
        st.sidebar.success(f"{len(source_files)} document(s) enregistré(s) — ingérés au prochain démarrage.")

    st.sidebar.subheader("Paramètres de boucle")
    max_turns = st.sidebar.number_input("max_turns", min_value=1, value=settings.loop.max_turns)
    max_batch = st.sidebar.number_input(
        "max_questions_per_batch", min_value=1, value=settings.loop.max_questions_per_batch
    )
    max_per_gap = st.sidebar.number_input(
        "max_questions_per_gap", min_value=1, value=settings.loop.max_questions_per_gap
    )

    st.sidebar.subheader("Sections à ignorer")
    sections = load_sections()
    skip_map = {}
    for sec in sections:
        if not sec.required:
            continue
        skip_map[sec.id] = st.sidebar.checkbox(
            f"Ignorer : {sec.title}", value=False, key=f"section_skip_{sec.id}"
        )

    start_clicked = st.sidebar.button(
        "Démarrer un nouveau run", disabled=st.session_state.run_active, type="primary", width="stretch"
    )

    if st.session_state.thread_id:
        st.sidebar.divider()
        st.sidebar.caption(f"thread_id : `{st.session_state.thread_id}`")
        if st.sidebar.button("Réinitialiser la télémétrie", width="stretch"):
            telemetry.reset(st.session_state.thread_id)
            st.session_state.step_timeline = []
            st.session_state.last_step_seconds = None
            st.rerun()

    loop_settings = LoopSettings(
        max_turns=int(max_turns),
        max_questions_per_batch=int(max_batch),
        max_questions_per_gap=int(max_per_gap),
    )
    return cdc_text, loop_settings, start_clicked, skip_map


def start_run(cdc_text: str, loop_settings: LoopSettings, skip_map: dict[str, bool]) -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_questions = None
    st.session_state.finished = False
    st.session_state.step_timeline = []
    telemetry.reset(st.session_state.thread_id)

    title = st.session_state.get("_cdc_title") or f"Run {st.session_state.thread_id[:8]}"
    try:
        db.create_run(st.session_state.thread_id, USERNAME, title)
    except Exception as exc:  # noqa: BLE001 - surfaced but non-fatal for the run
        st.warning(f"Impossible d'enregistrer le run dans l'historique : {exc}")

    sections = load_sections()
    section_statuses = {
        sec.id: SectionStatus(section_id=sec.id, status="skipped" if skip_map.get(sec.id) else "empty")
        for sec in sections
        if sec.required
    }
    input_state = {
        "initial_cdc_text": cdc_text,
        "loop_settings": loop_settings,
        "section_statuses": section_statuses,
    }
    run_graph(input_state)


# ---------------------------------------------------------------------------
# Tab: Pilotage
# ---------------------------------------------------------------------------


def render_run_metrics(values: dict) -> None:
    gaps = values.get("gaps", [])
    statuses = values.get("section_statuses", {})
    open_gaps = [g for g in gaps if g.status == "open"]
    blocking = sum(1 for g in open_gaps if g.severity == "blocking")
    complete = sum(1 for s in statuses.values() if s.status in ("complete", "skipped"))

    last = st.session_state.last_step_seconds
    total = sum(st.session_state.step_timeline)

    cols = st.columns(5)
    cols[0].metric("Dernier step", format_duration(last) if last is not None else "—")
    cols[1].metric("Temps cumulé", format_duration(total) if total else "—")
    cols[2].metric("Tour", values.get("turn", 0))
    cols[3].metric("Gaps ouverts", len(open_gaps), delta=f"{blocking} bloquants" if blocking else None,
                   delta_color="inverse" if blocking else "off")
    cols[4].metric("Sections traitées", f"{complete}/{len(statuses)}" if statuses else "—")


def render_status_table(values: dict) -> None:
    sections = values.get("sections_config", [])
    if not sections:
        st.info("Aucun run démarré pour l'instant.")
        return

    statuses = values.get("section_statuses", {})
    gaps = values.get("gaps", [])

    rows = []
    for sec in sections:
        status_obj = statuses.get(sec.id)
        status = status_obj.status if status_obj else "empty"
        open_gaps = [g for g in gaps if sec.id in g.section_ids and g.status == "open"]
        counts = {sev: sum(1 for g in open_gaps if g.severity == sev) for sev in SEVERITIES}
        score = score_section(
            [{"category": g.category, "severity": g.severity} for g in open_gaps]
        )
        rows.append(
            {
                "Section": sec.title,
                "Statut": f"{STATUS_ICONS.get(status, '')} {status}",
                "Score": round(score, 1),
                "Bloquantes": counts["blocking"],
                "Importantes": counts["important"],
                "Optionnelles": counts["nice_to_have"],
            }
        )

    st.subheader("État des sections")
    st.dataframe(rows, width="stretch", hide_index=True)


def _gap_sort_key(gap):
    return (
        gap.status not in ("open",),
        SEVERITY_ORDER.get(gap.severity, 9),
        CATEGORY_ORDER.get(gap.category, len(CATEGORY_ORDER)),
        gap.id,
    )


def render_provenance(item) -> None:
    """Show where one answer came from, how sure the system is, and its evidence.

    This is the visible half of the audit trail: origin badge, confidence band,
    review status, and — for RAG answers — the exact document and page cited.
    """
    badges = [f":{SOURCE_BADGE.get(item.source, 'gray')}-badge[{SOURCE_LABELS.get(item.source, item.source)}]"]
    if item.confidence is not None:
        band = confidence_band(item.confidence)
        badges.append(f":{CONFIDENCE_BADGE[band]}-badge[confiance {item.confidence:.0%}]")
    badges.append(
        f":{VALIDATION_BADGE.get(item.validation_status, 'gray')}-badge"
        f"[{VALIDATION_LABELS.get(item.validation_status, item.validation_status)}]"
    )
    st.markdown(f":material/reply: {' '.join(badges)}")
    st.markdown(item.content)

    if item.evidence:
        with st.popover(f":material/description: Preuves ({len(item.evidence)})"):
            for ev in item.evidence:
                st.markdown(f"**{format_evidence(ev)}** — score {ev.retrieval_score:.2f}")
                st.caption(f"`{ev.chunk_id}`")
                st.markdown(f"> {ev.excerpt}" if ev.excerpt else "_(extrait indisponible)_")

    trace = [f"origine : {item.created_by}"]
    if item.model:
        trace.append(f"modèle : {item.model}")
    if item.prompt_version:
        trace.append(f"prompt : {item.prompt_version}")
    if item.timestamp:
        trace.append(item.timestamp[:19].replace("T", " "))
    st.caption(" · ".join(trace))


def render_gap_card(gap, questions_by_gap: dict, answers_by_gap: dict, max_per_gap: int) -> None:
    with st.container(border=True):
        sev_badge = f":{SEVERITY_BADGE[gap.severity]}-badge[{SEVERITY_LABELS[gap.severity]}]"
        status_badge = f":{STATUS_BADGE.get(gap.status, 'gray')}-badge[{gap.status}]"
        st.markdown(
            f"{sev_badge} {status_badge} :blue-background[{gap.category}] &nbsp; :gray[:small[`{gap.id}`]]"
        )
        st.markdown(gap.description)

        asked = questions_by_gap.get(gap.id, [])
        answers = answers_by_gap.get(gap.id, [])
        if asked or answers:
            with st.expander(f"Échanges ({len(asked)} question(s), {len(answers)} réponse(s))"):
                for q in asked:
                    st.markdown(f":material/help: **Tour {q.turn} —** {q.text}")
                for item in answers:
                    render_provenance(item)

        st.caption(
            f"Questions posées : {gap.questions_asked}/{max_per_gap} · "
            f"sections : {', '.join(gap.section_ids) or '—'}"
        )


def render_gaps(values: dict) -> None:
    gaps = values.get("gaps", [])
    st.subheader("Gaps")
    if not gaps:
        st.caption("Aucun gap identifié pour l'instant.")
        return

    sections = values.get("sections_config", [])
    loop_settings = values.get("loop_settings")
    max_per_gap = loop_settings.max_questions_per_gap if loop_settings else 0

    questions_by_gap: dict[str, list] = {}
    for q in values.get("asked_questions", []):
        questions_by_gap.setdefault(q.gap_id, []).append(q)
    answers_by_gap: dict[str, list] = {}
    for item in values.get("context_items", []):
        if item.linked_gap_id:
            answers_by_gap.setdefault(item.linked_gap_id, []).append(item)

    show_closed = st.toggle("Afficher aussi les gaps résolus", value=False, key="gaps_show_closed")

    def visible(bucket):
        return [g for g in bucket if show_closed or g.status == "open"]

    for sec in sections:
        bucket = visible([g for g in gaps if sec.id in g.section_ids])
        if not bucket:
            continue
        bucket.sort(key=_gap_sort_key)
        open_count = sum(1 for g in bucket if g.status == "open")
        blocking = sum(1 for g in bucket if g.status == "open" and g.severity == "blocking")
        label = f"{sec.title} — {open_count} ouvert(s)"
        if blocking:
            label += f", {blocking} bloquant(s)"
        with st.expander(label, expanded=bool(blocking)):
            for gap in bucket:
                render_gap_card(gap, questions_by_gap, answers_by_gap, max_per_gap)

    orphans = visible([g for g in gaps if not g.section_ids])
    if orphans:
        orphans.sort(key=_gap_sort_key)
        with st.expander(f"Non rattachés — {len(orphans)}"):
            for gap in orphans:
                render_gap_card(gap, questions_by_gap, answers_by_gap, max_per_gap)


def render_question_form(questions: list[dict], values: dict) -> None:
    st.subheader("Questions en attente")
    gaps_by_id = {g.id: g for g in values.get("gaps", [])}

    # The section these questions belong to = the earliest one (in config order)
    # among the pending gaps. That is the section a "skip" button jumps past.
    sections = values.get("sections_config", [])
    section_order = {sec.id: i for i, sec in enumerate(sections)}
    section_title = {sec.id: sec.title for sec in sections}
    pending_section_ids = [
        sid for q in questions if (g := gaps_by_id.get(q["gap_id"])) for sid in g.section_ids
    ]
    skip_section_id = (
        min(pending_section_ids, key=lambda s: section_order.get(s, 1_000))
        if pending_section_ids
        else None
    )

    with st.form("answer_form"):
        for q in questions:
            gap = gaps_by_id.get(q["gap_id"])
            if gap is not None:
                st.markdown(
                    f":{SEVERITY_BADGE[gap.severity]}-badge[{SEVERITY_LABELS[gap.severity]}] "
                    f":blue-background[{gap.category}] &nbsp; :gray[:small[`{gap.id}`]]"
                )
            st.markdown(f"**{q['text']}**")
            col1, col2 = st.columns([4, 1], vertical_alignment="center")
            col1.text_input("Réponse", key=f"qa_answer_{q['gap_id']}", label_visibility="collapsed")
            col2.checkbox("Je ne sais pas", key=f"qa_skip_{q['gap_id']}")
        submit_col, skip_col = st.columns([1, 1])
        submitted = submit_col.form_submit_button("Envoyer les réponses", type="primary")
        skip_label = (
            f"⏭ Passer la section « {section_title.get(skip_section_id, skip_section_id)} »"
            if skip_section_id
            else "⏭ Passer cette section"
        )
        skipped_section = skip_col.form_submit_button(
            skip_label, disabled=skip_section_id is None
        )

    if skipped_section and skip_section_id:
        run_graph(Command(resume={"__skip_section__": skip_section_id}))
        st.rerun()
    elif submitted:
        answers = {
            q["gap_id"]: {
                "text": st.session_state.get(f"qa_answer_{q['gap_id']}", ""),
                "skip": st.session_state.get(f"qa_skip_{q['gap_id']}", False),
            }
            for q in questions
        }
        run_graph(Command(resume=answers))
        st.rerun()


def render_completion(values: dict, settings) -> None:
    st.success("Génération terminée.")
    if values.get("stop_reason"):
        st.warning(values["stop_reason"])

    output_dir = ROOT_DIR / settings.quarto.output_dir
    qmd_path = output_dir / f"{settings.quarto.output_basename}.qmd"
    docx_path = output_dir / f"{settings.quarto.output_basename}.{settings.quarto.target_format}"
    report_path = output_dir / "qa_report.md"
    decision_log_path = output_dir / "decision_log.jsonl"

    if report_path.exists():
        st.subheader("Rapport QA")
        st.markdown(report_path.read_text(encoding="utf-8"))

    col1, col2, col3 = st.columns(3)
    if qmd_path.exists():
        col1.download_button("Télécharger cdc_final.qmd", qmd_path.read_bytes(), file_name=qmd_path.name)
    if docx_path.exists():
        col2.download_button("Télécharger cdc_final.docx", docx_path.read_bytes(), file_name=docx_path.name)
    else:
        col2.info("Pas de .docx : Quarto n'est pas installé ou le rendu a échoué. Le .qmd reste disponible.")
    if decision_log_path.exists():
        col3.download_button(
            "Télécharger le journal de décisions",
            decision_log_path.read_bytes(),
            file_name=decision_log_path.name,
            help="Trace d'audit : une ligne JSON par décision prise par le système.",
        )


# ---------------------------------------------------------------------------
# Tab: Sous le capot
# ---------------------------------------------------------------------------


def render_timing_overview(summary: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Temps de calcul", format_duration(summary["compute_s"]))
    cols[1].metric("Dont LLM", format_duration(summary["llm_s"]),
                   delta=f"{summary['llm_share_of_compute']:.0%} du calcul", delta_color="off")
    cols[2].metric("Attente humaine", format_duration(summary["wait_s"]))
    cols[3].metric("Appels LLM", summary["llm_count"],
                   delta=f"{summary['retry_count']} avec retry" if summary["retry_count"] else None,
                   delta_color="inverse" if summary["retry_count"] else "off")
    cols[4].metric("Exécutions de nœuds", summary["node_count"],
                   delta=f"{summary['failure_count']} échec(s)" if summary["failure_count"] else None,
                   delta_color="inverse" if summary["failure_count"] else "off")


def render_node_timings(runs: list, summary: dict) -> None:
    st.subheader("Où passe le temps")
    by_node = summary["by_node"]
    if not by_node:
        st.caption("Pas encore de mesure.")
        return

    rows = sorted(
        (
            {
                "Nœud": name,
                "Exéc.": slot["count"],
                "Total (s)": round(slot["total_s"], 2),
                "Moyenne (s)": round(slot["mean_s"], 2),
                "Max (s)": round(slot["max_s"], 2),
                "Appels LLM": slot["llm_calls"],
                "Part du calcul": slot["share"],
            }
            for name, slot in by_node.items()
        ),
        key=lambda r: r["Total (s)"],
        reverse=True,
    )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={"Part du calcul": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
    )

    st.bar_chart(
        [{"Nœud": r["Nœud"], "Total (s)": r["Total (s)"]} for r in rows],
        x="Nœud",
        y="Total (s)",
        horizontal=True,
        height=max(180, 44 * len(rows)),
    )

    with st.expander("Chronologie détaillée (une ligne par exécution)"):
        origin = telemetry.run_started_at() or 0.0
        st.dataframe(
            [
                {
                    "#": i + 1,
                    "Nœud": run.label,
                    "Départ (s)": round(run.started_at - origin, 2),
                    "Durée (s)": round(run.duration_s, 2),
                    "LLM (s)": round(run.llm_seconds, 2),
                    "Appels": len(run.llm_calls),
                    "Erreur": run.error or "",
                }
                for i, run in enumerate(runs)
            ],
            width="stretch",
            hide_index=True,
        )


def render_llm_calls(calls: list, summary: dict) -> None:
    st.subheader("Appels LLM")
    if not calls:
        st.caption("Aucun appel LLM enregistré.")
        return

    if summary["cache_hit_count"]:
        st.info(
            f"{summary['cache_hit_count']} appel(s) servis depuis le cache disque "
            "(CDC_LLM_CACHE=1) — les durées affichées ne reflètent pas un vrai coût LLM."
        )

    if summary["retry_count"]:
        st.warning(
            f"{summary['retry_count']} appel(s) ont nécessité un retry de réparation JSON — "
            "coût invisible mais réel sur le temps total."
        )

    st.dataframe(
        [
            {
                "Nœud": c.node,
                "Schéma": c.schema,
                "Prompt": f"{c.prompt_id} ({c.prompt_version})" if c.prompt_id else "—",
                "Durée (s)": round(c.duration_s, 2),
                "Cache": "✓" if c.cache_hit else "",
                "Tentatives": c.attempts,
                "OK": c.ok,
                "Prompt (car.)": c.prompt_chars,
                "Réponse (car.)": c.response_chars,
                "Erreur": (c.error or "")[:200],
            }
            for c in calls
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Agrégé par schéma de sortie"):
        st.dataframe(
            sorted(
                (
                    {
                        "Schéma": name,
                        "Appels": slot["count"],
                        "Total (s)": round(slot["total_s"], 2),
                        "Moyenne (s)": round(slot["mean_s"], 2),
                        "Max (s)": round(slot["max_s"], 2),
                        "Retries": slot["retried"],
                        "Échecs": slot["failed"],
                    }
                    for name, slot in summary["by_schema"].items()
                ),
                key=lambda r: r["Total (s)"],
                reverse=True,
            ),
            width="stretch",
            hide_index=True,
        )


def render_raw_gaps(values: dict) -> None:
    st.subheader("Gaps (table brute)")
    gaps = values.get("gaps", [])
    if not gaps:
        st.caption("Aucun gap.")
        return
    st.dataframe(
        [
            {
                "id": g.id,
                "sections": ", ".join(g.section_ids),
                "catégorie": g.category,
                "sévérité": g.severity,
                "statut": g.status,
                "questions": g.questions_asked,
                "réponses": len(g.answer_item_ids),
                "description": g.description,
            }
            for g in gaps
        ],
        width="stretch",
        hide_index=True,
    )


def render_turn_log(values: dict) -> None:
    turn_log = values.get("turn_log", [])
    st.subheader("Journal des tours")
    if not turn_log:
        st.caption("Journal vide.")
        return
    for entry in reversed(turn_log):
        elapsed = entry.details.get("_elapsed_s") if entry.details else None
        suffix = f" — {format_duration(elapsed)}" if elapsed is not None else ""
        with st.expander(f"Tour {entry.turn} — {entry.agent} : {entry.summary}{suffix}"):
            if entry.details:
                st.json(entry.details)


def render_decision_log(values: dict) -> None:
    """The machine-readable audit trail: one row per AI decision.

    Answers "why did the system do that?" — which agent decided, on what
    inputs, with what confidence, and under which model/prompt version.
    """
    entries = values.get("decision_log", [])
    st.subheader("Journal des décisions")
    if not entries:
        st.caption("Aucune décision enregistrée.")
        return

    types = sorted({e.decision_type for e in entries})
    chosen = st.multiselect("Filtrer par type de décision", types, default=[], key="decision_types")
    shown = [e for e in entries if not chosen or e.decision_type in chosen]

    st.dataframe(
        [
            {
                "Horodatage": e.timestamp[11:19],
                "Tour": e.turn,
                "Agent": e.agent,
                "Décision": e.decision_type,
                "Résumé": e.summary,
                "Confiance": "—" if e.confidence is None else f"{e.confidence:.2f}",
                "Entrées": ", ".join(e.input_ids),
                "Sorties": ", ".join(e.output_ids),
                "Preuves": ", ".join(e.evidence_ids),
                "Modèle": e.model or "—",
                "Prompt": e.prompt_version or "—",
            }
            for e in reversed(shown)
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(f"{len(shown)} décision(s) affichée(s) sur {len(entries)}.")
    st.download_button(
        "Télécharger le journal (JSONL)",
        data="\n".join(e.model_dump_json() for e in entries),
        file_name="decision_log.jsonl",
        mime="application/jsonl",
        key="download_decision_log",
    )


def render_raw_state(values: dict) -> None:
    st.subheader("État brut du checkpointer")
    scalars = {
        "thread_id": st.session_state.thread_id,
        "turn": values.get("turn"),
        "current_mode": values.get("current_mode"),
        "current_section_id": values.get("current_section_id"),
        "active_fresh_item_ids": values.get("active_fresh_item_ids"),
        "done": values.get("done"),
        "stop_reason": values.get("stop_reason"),
    }
    st.json(scalars)

    with st.expander(f"context_items ({len(values.get('context_items', []))})"):
        st.json([item.model_dump() for item in values.get("context_items", [])])
    with st.expander(f"asked_questions ({len(values.get('asked_questions', []))})"):
        st.json([q.model_dump() for q in values.get("asked_questions", [])])
    with st.expander(f"sections_config ({len(values.get('sections_config', []))})"):
        st.json([s.model_dump() for s in values.get("sections_config", [])])


def render_debug_tab(values: dict) -> None:
    summary = telemetry.summary()
    render_timing_overview(summary)
    st.divider()
    render_node_timings(telemetry.node_runs(), summary)
    st.divider()
    render_llm_calls(telemetry.llm_calls(), summary)
    st.divider()
    render_raw_gaps(values)
    st.divider()
    render_turn_log(values)
    st.divider()
    render_decision_log(values)
    st.divider()
    render_raw_state(values)


# ---------------------------------------------------------------------------


def _autoload_last_run() -> None:
    """On a fresh session (e.g. after reload), reopen the user's most recent
    run so a page refresh lands back on the active run without re-selecting."""
    if st.session_state.get("_autoload_done"):
        return
    st.session_state["_autoload_done"] = True
    if st.session_state.get("thread_id"):
        return
    try:
        runs = db.list_runs(USERNAME)
    except Exception:  # noqa: BLE001 - autoload is best-effort
        return
    if runs:
        load_run_into_session(str(runs[0]["thread_id"]))


def main() -> None:
    init_session_state()
    _autoload_last_run()
    settings = load_settings()

    st.title("Assistant CDC IA")
    
    st.subheader("Système Multi-Agents Intelligent pour l’Analyse et le Raffinement des Cahiers des Charges")
    st.write("Un assistant intelligent qui aide à améliorer et compléter un cahier des charges."
"Il analyse le document pour identifier les informations manquantes, les ambiguïtés et les incohérences, puis propose des"
" améliorations ou pose des questions ciblées pour clarifier les besoins. À la fin, il génère un cahier des charges structuré"
" et prêt à être transmis à une équipe de développement, accompagné d'un rapport de vérification.")
    cdc_text, loop_settings, start_clicked, skip_map = render_sidebar(settings)

    tab_run, tab_debug = st.tabs([":material/play_arrow: Pilotage", ":material/build: Sous le capot"])

    # The run must execute inside the Pilotage tab so its st.status timeline
    # renders there rather than at the top of the page.
    with tab_run:
        if start_clicked:
            start_run(cdc_text, loop_settings, skip_map)

        if not st.session_state.thread_id:
            st.info("Charge un CDC initial (optionnel) et clique sur Démarrer dans la barre latérale.")
            values = {}
        else:
            values = get_state_values()
            render_run_metrics(values)
            render_status_table(values)

            if st.session_state.pending_questions:
                render_question_form(st.session_state.pending_questions, values)
            elif st.session_state.finished:
                render_completion(values, settings)

            render_gaps(values)

    with tab_debug:
        render_debug_tab(values)


if __name__ == "__main__":
    main()
