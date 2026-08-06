"""Helpers for building audit records: decision-log entries and provenance.

Agents stay pure — they *build* DecisionLogEntry objects and return them in
their result models; only graph.py node wrappers write them into
``state["decision_log"]`` (whose ``operator.add`` reducer appends them).

Confidence is an operational score, not a probability (roadmap §24): it drives
the acceptance policy in `validation_for`, and nothing else.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.ids import stable_id
from src.prompts import version as prompt_version_for
from src.state import (
    CDCState,
    DecisionLogEntry,
    DecisionType,
    Evidence,
    ValidationStatus,
    EXCERPT_MAX_CHARS,
)

# Bandes de confiance (roadmap §24) : haute -> acceptation automatique,
# moyenne/basse -> revue humaine.
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def confidence_band(confidence: float | None) -> str:
    """'high' | 'medium' | 'low' | 'unknown' — the label shown in the UI."""
    if confidence is None:
        return "unknown"
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def validation_for(confidence: float | None) -> ValidationStatus:
    """Acceptance policy for machine-produced context.

    High confidence is accepted automatically; anything else is flagged for
    human review rather than silently trusted.
    """
    return "accepted" if confidence_band(confidence) == "high" else "needs_review"


def clamp_confidence(value: float | None) -> float | None:
    """Keep an LLM-reported confidence inside 0–1 (models overshoot, e.g. 95)."""
    if value is None:
        return None
    if value > 1.0:
        value = value / 100.0 if value <= 100.0 else 1.0
    return max(0.0, min(1.0, float(value)))


def evidence_from_hit(hit: dict) -> Evidence:
    """Convert one src.rag.retrieve() hit into an Evidence record."""
    excerpt = (hit.get("content") or "").strip()
    if len(excerpt) > EXCERPT_MAX_CHARS:
        excerpt = excerpt[:EXCERPT_MAX_CHARS].rstrip() + "…"
    return Evidence(
        chunk_id=hit.get("chunk_id") or f"{hit.get('document', 'unknown')}::?",
        document=hit.get("document") or hit.get("source") or "unknown",
        page=hit.get("page"),
        retrieval_score=float(hit.get("score") or 0.0),
        excerpt=excerpt,
    )


def format_evidence(ev: Evidence) -> str:
    """Human-readable citation, e.g. 'stock_process.pdf, p. 14'."""
    return f"{ev.document}, p. {ev.page}" if ev.page is not None else ev.document


def make_decision(
    state: CDCState,
    *,
    agent: str,
    decision_type: DecisionType,
    summary: str,
    prompt_id: str | None = None,
    input_ids: list[str] | None = None,
    output_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    confidence: float | None = None,
    model: str | None = None,
    **details,
) -> DecisionLogEntry:
    """Build one decision-log entry, stamping turn/thread/timestamp from context.

    The id is content-derived (like every other id here, see src/ids.py) so a
    replayed run produces the same decision ids.
    """
    from src import telemetry  # local import: telemetry imports nothing from us

    turn = state.get("turn", 0)
    input_ids = input_ids or []
    return DecisionLogEntry(
        id=stable_id("dec", str(turn), agent, decision_type, summary, *input_ids),
        timestamp=now_iso(),
        thread_id=telemetry.thread_id(),
        turn=turn,
        agent=agent,
        decision_type=decision_type,
        summary=summary,
        input_ids=input_ids,
        output_ids=output_ids or [],
        evidence_ids=evidence_ids or [],
        confidence=confidence,
        model=model,
        prompt_version=prompt_version_for(prompt_id),
        details=details,
    )
