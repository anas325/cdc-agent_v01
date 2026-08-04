"""Final validator: last QA pass after synthesis, before END.

Checks for unmapped context items and section-to-section contradictions
that weren't visible during the Q&A loop, then writes output/qa_report.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_settings
from src.decisions import confidence_band, format_evidence
from src.llm import call_structured
from src.state import CDCState, ContextItem

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class FinalContradiction(BaseModel):
    description: str


class FinalCheckOutput(BaseModel):
    contradictions: list[FinalContradiction] = Field(default_factory=list)


def find_unmapped_context(state: CDCState, mapped: dict[str, list[str]]) -> list[ContextItem]:
    mapped_ids = {iid for ids in mapped.values() for iid in ids}
    return [it for it in state["context_items"] if it.id not in mapped_ids]


def find_final_contradictions(qmd_text: str) -> list[str]:
    prompt = f"""Voici un cahier des charges (CDC) assemblé, au format Quarto Markdown. Relis-le
dans son ensemble et identifie toute incohérence entre sections qui n'aurait pas été visible
lorsque chaque section était rédigée isolément (ex: chiffres différents pour la même métrique,
fonctionnalité mentionnée dans une section mais explicitement exclue dans une autre).

DOCUMENT :
{qmd_text}

Ne signale que des incohérences concrètes et vérifiables. Liste vide si le document est cohérent."""
    output = call_structured(prompt, FinalCheckOutput, prompt_id="final_validator.contradictions")
    return [c.description for c in output.contradictions]


_CONFIDENCE_LABELS = {"high": "haute", "medium": "moyenne", "low": "faible", "unknown": "n/c"}


def _provenance_line(item: ContextItem) -> str:
    """One audit line: origin, confidence, review status, and cited evidence."""
    bits = [f"**{item.source}**", item.content]
    meta = [f"origine={item.created_by}", f"validation={item.validation_status}"]
    if item.confidence is not None:
        meta.append(
            f"confiance={item.confidence:.2f} ({_CONFIDENCE_LABELS[confidence_band(item.confidence)]})"
        )
    if item.model:
        meta.append(f"modèle={item.model}")
    if item.prompt_version:
        meta.append(f"prompt={item.prompt_version}")
    line = f"- {' — '.join(bits)}\n  - _{', '.join(meta)}_"
    if item.evidence:
        citations = "; ".join(
            f"{format_evidence(ev)} (score {ev.retrieval_score:.2f}, `{ev.chunk_id}`)"
            for ev in item.evidence
        )
        line += f"\n  - Preuves : {citations}"
    return line


def write_decision_log(state: CDCState) -> Path:
    """Dump the run's decision log as JSONL, one entry per line.

    JSONL rather than a single JSON array so an evaluation harness can stream it
    and so appending runs stays trivial.
    """
    settings = load_settings()
    output_dir = ROOT_DIR / settings.quarto.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "decision_log.jsonl"
    entries = state.get("decision_log") or []
    path.write_text(
        "\n".join(entry.model_dump_json() for entry in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )
    return path


def write_qa_report(
    state: CDCState,
    unmapped: list[ContextItem],
    final_contradictions: list[str],
) -> Path:
    settings = load_settings()
    output_dir = ROOT_DIR / settings.quarto.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qa_report.md"

    gaps = state["gaps"]
    resolved = [g for g in gaps if g.status in ("resolved", "rag_answered", "user_answered")]
    assumed = [g for g in gaps if g.status == "assumed"]
    deferred = [g for g in gaps if g.status == "deferred"]

    lines = ["# Rapport QA — Cahier des charges", ""]

    lines.append(f"## Lacunes résolues ({len(resolved)})")
    lines += [f"- [{g.severity}] {g.description}" for g in resolved] or ["_Aucune._"]
    lines.append("")

    lines.append(f"## Hypothèses retenues faute de réponse ({len(assumed)})")
    lines += [f"- [{g.severity}] {g.description}" for g in assumed] or ["_Aucune._"]
    lines.append("")

    lines.append(f"## Lacunes reportées (limite de tours atteinte) ({len(deferred)})")
    lines += [f"- [{g.severity}] {g.description}" for g in deferred] or ["_Aucune._"]
    lines.append("")

    # Traçabilité : tout ce que le système a produit lui-même (RAG ou hypothèse)
    # doit pouvoir être remonté à sa source et à son niveau de validation.
    machine_items = [it for it in state["context_items"] if it.source in ("rag", "assumption")]
    needs_review = [it for it in machine_items if it.validation_status != "accepted"]
    lines.append(f"## Traçabilité des informations produites par le système ({len(machine_items)})")
    lines.append(
        f"Dont {len(needs_review)} à faire relire avant validation du CDC."
        if machine_items
        else ""
    )
    lines += [_provenance_line(it) for it in machine_items] or ["_Aucune._"]
    lines.append("")

    lines.append(f"## Éléments de contexte non intégrés au document ({len(unmapped)})")
    lines += [f"- ({it.source}) {it.content}" for it in unmapped] or ["_Aucun._"]
    lines.append("")

    lines.append(f"## Incohérences détectées lors de la relecture finale ({len(final_contradictions)})")
    lines += [f"- {c}" for c in final_contradictions] or ["_Aucune._"]
    lines.append("")

    if state.get("stop_reason"):
        lines.append("## Motif d'arrêt")
        lines.append(state["stop_reason"])
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
