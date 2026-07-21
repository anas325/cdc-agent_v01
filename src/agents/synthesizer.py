"""Synthesizer agent.

Renders each section's context items into coherent French CDC prose,
fills the Quarto template's slots, writes output/cdc_final.qmd, and shells
out to `quarto render` targeting DOCX (non-fatal if Quarto isn't installed).
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
from functools import partial
from pathlib import Path

from pydantic import BaseModel

from src.config import load_settings
from src.context_utils import format_context_items, get_section
from src.llm import call_structured, map_structured
from src.state import CDCState, ContextItem

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class SlotDraft(BaseModel):
    prose: str


def render_slot(state: CDCState, section_id: str, items: list[ContextItem]) -> str:
    section = get_section(state, section_id)
    if not items:
        return "_Aucune information disponible pour cette section._"

    assumptions = [it for it in items if it.source == "assumption"]
    normal_items = [it for it in items if it.source != "assumption"]

    prompt = f"""Tu rédiges la section "{section.title}" d'un cahier des charges (CDC)
professionnel en français, à partir des éléments de contexte validés ci-dessous. Le texte doit
être fluide, structuré (utilise des sous-titres Markdown ### et des listes si utile), au registre
professionnel d'un CDC, sans reformuler en questions, sans méta-commentaire sur le processus de
rédaction.

Description de la section : {section.description}

ÉLÉMENTS DE CONTEXTE VALIDÉS (hors hypothèses) :
{format_context_items(normal_items)}

Rédige uniquement le corps de la section (pas de titre h1, le titre est déjà géré ailleurs)."""
    draft = call_structured(prompt, SlotDraft)
    prose = draft.prose.strip()

    if assumptions:
        callouts = "\n\n".join(
            f"::: {{.callout-warning}}\n## Hypothèse retenue\n{it.content}\n:::" for it in assumptions
        )
        prose = f"{prose}\n\n{callouts}"

    return prose


def build_section_render_map(state: CDCState) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Returns (slot_id -> rendered prose, section_id -> mapped context item ids).

    Each section renders independently (no rolling state), so the per-section
    LLM calls are fanned out concurrently; results are zipped back in order.
    """
    sections = state["sections_config"]
    items_per = [[it for it in state["context_items"] if sec.id in it.section_ids] for sec in sections]
    proses = map_structured(
        [partial(render_slot, state, sec.id, items) for sec, items in zip(sections, items_per)]
    )
    slots = {sec.template_slot: prose for sec, prose in zip(sections, proses)}
    mapped = {sec.id: [it.id for it in items] for sec, items in zip(sections, items_per)}
    return slots, mapped


def fill_template(template_text: str, slots: dict[str, str], project_subtitle: str) -> str:
    text = template_text
    text = text.replace("{{ project_subtitle }}", project_subtitle)
    text = text.replace("{{ generation_date }}", datetime.date.today().isoformat())
    for slot_id, content in slots.items():
        text = text.replace(f"{{{{ {slot_id} }}}}", content)
    return text


def run_synthesis(state: CDCState) -> tuple[str, dict[str, list[str]]]:
    """Writes output/cdc_final.qmd, attempts quarto render. Returns (qmd_text, mapped_items)."""
    settings = load_settings()
    template_path = ROOT_DIR / settings.quarto.template_path
    output_dir = ROOT_DIR / settings.quarto.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    template_text = template_path.read_text(encoding="utf-8")
    slots, mapped = build_section_render_map(state)
    qmd_text = fill_template(template_text, slots, project_subtitle="Généré par CDC Refinement Agent Swarm")

    qmd_path = output_dir / f"{settings.quarto.output_basename}.qmd"
    qmd_path.write_text(qmd_text, encoding="utf-8")

    if shutil.which("quarto"):
        try:
            subprocess.run(
                ["quarto", "render", str(qmd_path), "--to", settings.quarto.target_format],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(output_dir),
            )
        except subprocess.CalledProcessError:
            pass  # non-fatal; UI surfaces qmd-only fallback

    return qmd_text, mapped
