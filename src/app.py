"""Streamlit test interface for the CDC refinement agent swarm.

Single-session, single-run-at-a-time tool: upload an initial CDC and/or RAG
source docs, start a run, answer batched questions as they come up, and
download the final document once the graph reaches END. All durable state
lives in the LangGraph checkpointer (keyed by thread_id) — every rerun
re-reads it instead of trusting in-memory globals.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# transformers' optional vision submodules import torchvision (not installed);
# Streamlit's file watcher probes every loaded module and logs that as a warning.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from langgraph.types import Command

from src.config import load_settings
from src.graph import build_graph
from src.state import LoopSettings
SOURCE_DOCS_DIR = ROOT_DIR / "data" / "source_docs"

STATUS_ICONS = {
    "empty": "⚪",
    "in_progress": "🟡",
    "complete": "🟢",
    "reopened": "🔴",
}

SEVERITIES = ["blocking", "important", "nice_to_have"]

st.set_page_config(page_title="CDC Refinement Agent", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


def init_session_state() -> None:
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("run_active", False)
    st.session_state.setdefault("pending_questions", None)
    st.session_state.setdefault("finished", False)


def current_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def process_step_result(result: dict) -> None:
    interrupts = result.get("__interrupt__")
    if interrupts:
        st.session_state.pending_questions = interrupts[0].value.get("questions", [])
        st.session_state.finished = False
        st.session_state.run_active = False
    else:
        st.session_state.pending_questions = None
        st.session_state.finished = bool(result.get("done"))
        st.session_state.run_active = False


def get_state_values() -> dict:
    if not st.session_state.thread_id:
        return {}
    snapshot = get_graph().get_state(current_config())
    return snapshot.values or {}


def render_sidebar(settings) -> tuple[str, LoopSettings, bool]:
    st.sidebar.header("Configuration")

    cdc_file = st.sidebar.file_uploader("CDC initial (markdown/texte)", type=["md", "txt"])
    cdc_text = cdc_file.read().decode("utf-8", errors="ignore") if cdc_file is not None else ""

    st.sidebar.subheader("Documents source (RAG)")
    source_files = st.sidebar.file_uploader(
        "Ajouter des documents source", type=["md", "txt"], accept_multiple_files=True
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

    start_clicked = st.sidebar.button(
        "Démarrer / Reprendre un nouveau run", disabled=st.session_state.run_active, type="primary"
    )

    loop_settings = LoopSettings(
        max_turns=int(max_turns),
        max_questions_per_batch=int(max_batch),
        max_questions_per_gap=int(max_per_gap),
    )
    return cdc_text, loop_settings, start_clicked


def start_run(cdc_text: str, loop_settings: LoopSettings) -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.run_active = True
    st.session_state.pending_questions = None
    st.session_state.finished = False
    input_state = {"initial_cdc_text": cdc_text, "loop_settings": loop_settings}
    with st.spinner("Exécution de l'agent swarm..."):
        result = get_graph().invoke(input_state, current_config())
    process_step_result(result)


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
        rows.append(
            {
                "Section": sec.title,
                "Statut": f"{STATUS_ICONS.get(status, '')} {status}",
                "Bloquantes": counts["blocking"],
                "Importantes": counts["important"],
                "Optionnelles": counts["nice_to_have"],
            }
        )

    st.subheader("État des sections")
    st.dataframe(rows, width='stretch', hide_index=True)


def render_question_form(questions: list[dict], values: dict) -> None:
    st.subheader("Questions en attente")
    gaps = values.get("gaps", [])
    
    with st.form("answer_form"):
        for q in questions:
            st.write(q['gap_id'])
            st.markdown(f"**{q['text']}**")
            col1, col2 = st.columns([4, 1])
            col1.text_input("Réponse", key=f"answer_{q['gap_id']}", label_visibility="collapsed")
            col2.checkbox("Je ne sais pas", key=f"skip_{q['gap_id']}")
        submitted = st.form_submit_button("Envoyer les réponses")

    if submitted:
        answers = {
            q["gap_id"]: {
                "text": st.session_state.get(f"answer_{q['gap_id']}", ""),
                "skip": st.session_state.get(f"skip_{q['gap_id']}", False),
            }
            for q in questions
        }
        with st.spinner("Reprise de l'exécution..."):
            result = get_graph().invoke(Command(resume=answers), current_config())
        process_step_result(result)
        st.rerun()


def render_turn_log(values: dict) -> None:
    turn_log = values.get("turn_log", [])
    if not turn_log:
        return
    st.subheader("Journal des tours")
    for entry in reversed(turn_log):
        with st.expander(f"Tour {entry.turn} — {entry.agent} : {entry.summary}"):
            if entry.details:
                st.json(entry.details)


def render_completion(values: dict, settings) -> None:
    st.success("Génération terminée.")
    if values.get("stop_reason"):
        st.warning(values["stop_reason"])

    output_dir = ROOT_DIR / settings.quarto.output_dir
    qmd_path = output_dir / f"{settings.quarto.output_basename}.qmd"
    docx_path = output_dir / f"{settings.quarto.output_basename}.{settings.quarto.target_format}"
    report_path = output_dir / "qa_report.md"

    if report_path.exists():
        st.subheader("Rapport QA")
        st.markdown(report_path.read_text(encoding="utf-8"))

    col1, col2 = st.columns(2)
    if qmd_path.exists():
        col1.download_button("Télécharger cdc_final.qmd", qmd_path.read_bytes(), file_name=qmd_path.name)
    if docx_path.exists():
        col2.download_button("Télécharger cdc_final.docx", docx_path.read_bytes(), file_name=docx_path.name)
    else:
        col2.info("Pas de .docx : Quarto n'est pas installé ou le rendu a échoué. Le .qmd reste disponible.")


def main() -> None:
    init_session_state()
    settings = load_settings()

    st.title("CDC Refinement Agent Swarm")

    cdc_text, loop_settings, start_clicked = render_sidebar(settings)
    if start_clicked:
        start_run(cdc_text, loop_settings)

    if not st.session_state.thread_id:
        st.info("Charge un CDC initial (optionnel) et clique sur Démarrer dans la barre latérale.")
        return

    values = get_state_values()
    render_status_table(values)

    if st.session_state.pending_questions:
        render_question_form(st.session_state.pending_questions, values)
    elif st.session_state.finished:
        render_completion(values, settings)

    render_turn_log(values)


if __name__ == "__main__":
    main()
