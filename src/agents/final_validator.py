"""Final validator: last QA pass after synthesis, before END.

Checks for unmapped context items and section-to-section contradictions
that weren't visible during the Q&A loop, then writes output/qa_report.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_settings
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
    output = call_structured(prompt, FinalCheckOutput)
    return [c.description for c in output.contradictions]


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
