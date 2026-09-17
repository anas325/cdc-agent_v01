"""A deliberately simple reference system to compare the swarm against.

The graph in `src/graph.py` is a lot of machinery — per-section fan-out, an
orchestrator that picks what to work on, RAG with an LLM grading its own hits, a
question-dedup gate, a critic, a loop. The benchmark says how well that machinery
does. It cannot say whether the machinery is *worth it*, because there is nothing
underneath it to compare to: a gap precision of 0.55 is a good number or a bad one
depending entirely on what one LLM call would have scored.

So this is that one LLM call, wired to the same dataset, the same synthetic
stakeholder and the same scorer:

1. **One pass.** The whole CDC and the whole section list go into a single
   `call_structured`, which returns every gap it can see, each with a question
   already attached. No sections, no turns, no orchestrator.
2. **Retrieval, instrumented but not trusted.** One top-k query per gap, logged
   with its ranks — so Recall@K is measured on the same index the graph uses —
   but the gap goes to the human anyway. Deciding that a chunk *answers* a gap
   is what `gap_filler`'s grading call does; a baseline that did it too would be
   measuring the swarm's component, not standing in for it. The cost of not
   doing it is the number this makes visible: every question RAG could have
   saved, the baseline still asks.

   `--rag-closes-gaps` switches on the other naive option — trust the retriever's
   own similarity score above a floor. On this dataset it is not a real
   alternative and the run shows why: scores land in a narrow band (~0.55–0.65
   on the ecommerce case), so any floor either closes every gap or none.
3. **One question batch.** Every open gap is asked at once, with no dedup and no
   budget. Answers come from `evals/simulator.py` — the same stakeholder the
   graph faces.
4. **Naive integration.** An answer closes its gap (`user_answered`), a "je ne
   sais pas" becomes an assumption (`assumed`). Nothing is checked against
   anything else, so no contradiction is ever found after the first pass.

What it shares with the real system is deliberate: `call_structured`, the same
provider/model config, the same RAG index, the same `Gap`/`ContextItem`/
`DecisionLogEntry` records, the same content-hashed ids. The comparison then
isolates the *architecture* rather than the model or the plumbing.

Pure functions only — `run_baseline.py` owns the CLI, the per-case isolation and
the files on disk, mirroring the `scoring.py` / `run_scoring.py` split.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.gap_finder_prompts import GAP_TAXONOMY
from src.decisions import clamp_confidence, evidence_from_hit, make_decision, now_iso
from src.ids import stable_id
from src.llm import call_structured, current_model_name
from src.state import (
    AskedQuestion,
    ContextItem,
    DecisionLogEntry,
    Gap,
    GapCategory,
    GapSeverity,
    SectionConfig,
    SectionStatus,
)

PROMPT_ID = "baseline.oneshot"

AGENT = "baseline"

# Only consulted under `--rag-closes-gaps`. `src/rag.py::retrieve` returns
# 1/(1+distance), and on this dataset the returned scores sit in a narrow band
# well above any value that would reject a hit — which is the point of having the
# option at all: it demonstrates that similarity alone carries no usable signal
# about whether a chunk answers a gap.
RAG_SCORE_FLOOR = 0.35

# How many chunks the pasted-in answer keeps. Enough to read as an answer, capped
# so one gap doesn't carry half the corpus into the record.
RAG_CHUNKS_KEPT = 2


# ---------------------------------------------------------------------------
# The single LLM call
# ---------------------------------------------------------------------------


class BaselineGap(BaseModel):
    """One finding, with its question — the graph needs two agents for this."""

    section_ids: list[str] = Field(default_factory=list)
    category: GapCategory
    description: str
    severity: GapSeverity
    question: str = ""


class BaselineOutput(BaseModel):
    gaps: list[BaselineGap] = Field(default_factory=list)


def build_prompt(cdc_text: str, sections: list[SectionConfig]) -> str:
    """Everything the baseline ever sees, in one string.

    Intentionally close to a competent prompt-only solution: the taxonomy, the
    section list and a severity rule, with no per-section framing, no context
    from earlier turns and no few-shot examples.
    """
    catalogue = "\n".join(f"- {s.id} — {s.title} : {s.description}" for s in sections)
    fence = '"""'
    return f"""Tu es un analyste chargé de relire un cahier des charges (CDC) avant de le
transmettre à une équipe de développement.

SECTIONS ATTENDUES DANS UN CDC (utilise ces identifiants, et eux seuls, dans section_ids) :
{catalogue}

CATÉGORIES DE LACUNES (champ category, valeurs exactes) :
{", ".join(GAP_TAXONOMY)}

SÉVÉRITÉ (champ severity) :
- "blocking" : deux équipes qui codent honnêtement ce point obtiendraient des résultats
  différents ; le choix ne peut pas être tranché par la technique seule.
- "important" : l'implémentation est possible sous hypothèse, mais cette hypothèse change
  un comportement visible et doit être confirmée par le métier.
- "nice_to_have" : imprécision de vocabulaire qui n'empêche ni le chiffrage ni le
  développement.

CAHIER DES CHARGES :
{fence}
{cdc_text}
{fence}

Relève TOUTES les lacunes, ambiguïtés et contradictions qui empêchent de développer à
partir de ce texte. Pour chacune :
- `description` : ce qui manque ou ce qui est ambigu, en citant le passage concerné ;
- `category` et `severity` dans les listes ci-dessus ;
- `section_ids` : la ou les sections concernées ;
- `question` : UNE question fermée à poser à l'auteur du CDC pour la combler, qui rappelle
  le passage concerné.

Pour une contradiction, `description` doit énoncer les DEUX affirmations qui s'opposent.
N'invente pas de lacune sur un point que le texte traite déjà clairement."""


def analyze(
    cdc_text: str, sections: list[SectionConfig]
) -> tuple[list[Gap], list[AskedQuestion], list[DecisionLogEntry]]:
    """The baseline's whole analysis: one call in, gaps and questions out.

    Questions are built here rather than in a later step because that *is* the
    baseline — the real system drafts a question only for the gap the
    orchestrator selected, after RAG failed and after the dedup gate.
    """
    result = call_structured(
        build_prompt(cdc_text, sections), BaselineOutput, prompt_id=PROMPT_ID
    )

    valid_ids = {s.id for s in sections}
    state = {"turn": 0}
    model = current_model_name()

    gaps: list[Gap] = []
    questions: list[AskedQuestion] = []
    decisions: list[DecisionLogEntry] = []

    for index, candidate in enumerate(result.gaps):
        # A hallucinated section id would make every downstream section lookup
        # (and the scorer's section bonus) meaningless, so drop it rather than
        # carry it; a gap left with no valid section survives, untagged.
        section_ids = [sid for sid in candidate.section_ids if sid in valid_ids]
        # The index keeps two identically-worded gaps from collapsing onto one id.
        gap_id = stable_id("gap", str(index), candidate.description)
        text = (candidate.question or "").strip()

        gaps.append(
            Gap(
                id=gap_id,
                section_ids=section_ids,
                category=candidate.category,
                description=candidate.description,
                severity=candidate.severity,
                status="open",
                question_text=text or None,
            )
        )
        decisions.append(
            make_decision(
                state,
                agent=AGENT,
                decision_type="gap_detected",
                summary=candidate.description[:200],
                prompt_id=PROMPT_ID,
                output_ids=[gap_id],
                model=model,
                section_ids=section_ids,
                category=candidate.category,
                severity=candidate.severity,
            )
        )

        if not text:
            continue
        question = AskedQuestion(
            id=stable_id("q", gap_id, text), gap_id=gap_id, text=text, turn=1
        )
        questions.append(question)
        decisions.append(
            make_decision(
                state,
                agent=AGENT,
                decision_type="question_drafted",
                summary=text[:200],
                prompt_id=PROMPT_ID,
                input_ids=[gap_id],
                output_ids=[question.id],
                model=model,
            )
        )

    return gaps, questions, decisions


# ---------------------------------------------------------------------------
# Naive retrieval
# ---------------------------------------------------------------------------


def fill_from_rag(
    gaps: list[Gap],
    retrieve,
    *,
    close_gaps: bool = False,
    score_floor: float = RAG_SCORE_FLOOR,
) -> tuple[list[ContextItem], list[DecisionLogEntry]]:
    """One query per gap, always logged; only `close_gaps` lets it close one.

    Every outcome lands in the decision log with its `evidence_ids`, including
    the rejections — a rejected retrieval leaves no ContextItem behind, so the
    log is the only place its rank order survives for `scoring.score_retrieval`.
    That is what makes Recall@K comparable between the two systems even though
    the baseline does not act on what it retrieves.

    With `close_gaps`, the retriever's own similarity score above `score_floor`
    is taken as an answer, with no model reading the chunks — the failure
    `gap_filler`'s grading step exists to prevent.

    `retrieve` is injected rather than imported so a test can drive this without
    a Chroma index. It mutates `gaps` in place — status and `answer_item_ids` —
    which is the one place this module is not pure, and is confined to objects
    the caller handed it.
    """
    items: list[ContextItem] = []
    decisions: list[DecisionLogEntry] = []
    state = {"turn": 1}

    for gap in gaps:
        query = f"{gap.description} {gap.question_text or ''}".strip()
        try:
            hits = retrieve(query)
        except Exception as exc:  # an empty or broken index must not kill the case
            decisions.append(
                make_decision(
                    state,
                    agent=AGENT,
                    decision_type="rag_rejected",
                    summary=f"échec de la récupération : {exc}",
                    input_ids=[gap.id],
                )
            )
            continue

        gap.rag_attempted = True
        evidence = [evidence_from_hit(hit) for hit in hits]
        evidence_ids = [ev.chunk_id for ev in evidence]
        best = max((hit.get("score") or 0.0 for hit in hits), default=0.0)

        if not close_gaps or not hits or best < score_floor:
            summary = (
                f"meilleur score {best:.2f} < {score_floor}"
                if close_gaps
                else "récupération mesurée mais non exploitée (la lacune part en question)"
            )
            decisions.append(
                make_decision(
                    state,
                    agent=AGENT,
                    decision_type="rag_rejected",
                    summary=summary,
                    input_ids=[gap.id],
                    evidence_ids=evidence_ids,
                    confidence=clamp_confidence(best),
                )
            )
            continue

        # No model reads these chunks: the baseline pastes them in as the answer.
        content = "\n\n".join(hit["content"] for hit in hits[:RAG_CHUNKS_KEPT])
        item = ContextItem(
            id=stable_id("ctx", gap.id, "rag", content),
            content=content,
            source="rag",
            section_ids=list(gap.section_ids),
            linked_gap_id=gap.id,
            turn_added=1,
            created_by="rag",
            timestamp=now_iso(),
            evidence=evidence[:RAG_CHUNKS_KEPT],
            confidence=clamp_confidence(best),
        )
        items.append(item)
        gap.status = "rag_answered"
        gap.answer_item_ids = [item.id]
        decisions.append(
            make_decision(
                state,
                agent=AGENT,
                decision_type="rag_answer",
                summary=f"lacune close sur le seul score de récupération ({best:.2f})",
                input_ids=[gap.id],
                output_ids=[item.id],
                evidence_ids=evidence_ids,
                confidence=clamp_confidence(best),
            )
        )

    return items, decisions


# ---------------------------------------------------------------------------
# Naive integration
# ---------------------------------------------------------------------------


def integrate(
    gaps: list[Gap], replies: list[dict]
) -> tuple[list[ContextItem], list[DecisionLogEntry]]:
    """Answers in, context items out — with nothing checked against anything.

    The graph runs the critic at this point, which is why a contradiction between
    two *answers* can only ever be found by the real system. Skips become
    assumptions, the way the Streamlit « Je ne sais pas » button does, so the
    baseline does not silently drop an unanswered question either.
    """
    by_id = {gap.id: gap for gap in gaps}
    items: list[ContextItem] = []
    decisions: list[DecisionLogEntry] = []
    state = {"turn": 1}

    for reply in replies:
        gap = by_id.get(reply["gap_id"])
        if gap is None:
            continue
        gap.questions_asked += 1
        skipped = bool(reply.get("skip")) or not (reply.get("text") or "").strip()

        if skipped:
            content = (
                f"Hypothèse (l'auteur n'a pas su répondre) : {gap.description} "
                "— à valider avant développement."
            )
            item = ContextItem(
                id=stable_id("ctx", gap.id, "assumption", content),
                content=content,
                source="assumption",
                section_ids=list(gap.section_ids),
                linked_gap_id=gap.id,
                turn_added=1,
                created_by="llm",
                timestamp=now_iso(),
                validation_status="needs_review",
            )
            gap.status = "assumed"
            decision_type, summary = "assumption_built", "réponse « je ne sais pas »"
        else:
            content = reply["text"].strip()
            item = ContextItem(
                id=stable_id("ctx", gap.id, "user", content),
                content=content,
                source="user_answer",
                section_ids=list(gap.section_ids),
                linked_gap_id=gap.id,
                turn_added=1,
                created_by="user",
                timestamp=now_iso(),
                validation_status="accepted",
            )
            gap.status = "user_answered"
            decision_type, summary = "answer_integrated", content[:200]

        items.append(item)
        gap.answer_item_ids = [*gap.answer_item_ids, item.id]
        decisions.append(
            make_decision(
                state,
                agent=AGENT,
                decision_type=decision_type,
                summary=summary,
                input_ids=[gap.id],
                output_ids=[item.id],
            )
        )

    return items, decisions


# ---------------------------------------------------------------------------
# Section bookkeeping
# ---------------------------------------------------------------------------

OPEN_STATUSES = ("open", "deferred")


def section_statuses(
    gaps: list[Gap], sections: list[SectionConfig]
) -> dict[str, SectionStatus]:
    """Same rule as `gap_finder.compute_section_complete`: a section is complete
    when nothing blocking or important is still open in it.

    Only required sections are tracked, because that is the set the graph starts
    from (`run_benchmark.run_case`) and therefore the denominator the scorer's
    `sections_complete_ratio` uses on the other side of the comparison.
    """
    statuses: dict[str, SectionStatus] = {}
    for section in sections:
        if not section.required:
            continue
        in_section = [gap for gap in gaps if section.id in gap.section_ids]
        outstanding = [
            gap
            for gap in in_section
            if gap.status in OPEN_STATUSES and gap.severity in ("blocking", "important")
        ]
        if outstanding:
            status = "in_progress"
        else:
            status = "complete" if in_section else "empty"
        statuses[section.id] = SectionStatus(section_id=section.id, status=status)
    return statuses
