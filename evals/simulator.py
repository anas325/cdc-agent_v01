"""Synthetic stakeholders that answer the graph's question batches (roadmap Phase 3).

`human_input_node` blocks on `interrupt({"questions": [{gap_id, text}, ...]})` and
resumes on `Command(resume={gap_id: {"text": ..., "skip": ...}})`. A simulator is
anything that turns the former into the latter, so a whole run can happen without
a human.

Two implementations, matching roadmap §15's two evaluation modes:

* `OracleSimulator` (mode A) — deterministic, no LLM. Replays the ground-truth
  `expected_answers` for whichever annotated gap the question is about. Measures
  whether the system integrates *correct* information correctly.
* `StakeholderSimulator` (mode B) — an LLM constrained to a stakeholder profile.
  It may only answer from the profile's knowledge; anything else comes back as
  "je ne sais pas". Measures behaviour under realistic interaction.

The "je ne sais pas" path is not a special case: returning skip=True is exactly
what the Streamlit user's *Je ne sais pas* button does, so `integrate_answers_node`
builds an assumption from it (src/graph.py). Contradictory answers likewise flow
into the critic untouched.

**Why the constraint on the LLM stakeholder matters**: a simulator allowed to
invent answers would resolve gaps the reference documents can't actually close,
and every downstream metric would be inflated. The prompt forbids it explicitly
and the wire model carries a `knows` flag so refusal is a first-class outcome.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from evals.dataset import Behavior, GroundTruth, StakeholderProfile
from src.llm import call_structured
from src.state import CDCState, Gap

# A question matches an annotated gap when this share of the annotation's
# keywords appear in the gap description + question text. Low enough to survive
# the agent's paraphrasing, high enough that unrelated gaps don't collide.
MATCH_THRESHOLD = 0.5

# In realistic mode, the per-question chance of answering with one of the
# profile's contradictory statements instead of the truthful one. Drawn from the
# seeded RNG, so a given --seed always produces the same interaction.
CONTRADICTION_RATE = 0.35

DONT_KNOW = "Je ne sais pas."


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SimulatedReply:
    gap_id: str
    text: str
    skip: bool
    # Why this reply came out the way it did — carried into the run transcript
    # so a surprising result can be traced back to the simulator, not the agent.
    reason: str
    matched_gap_ref: str | None = None
    confidence: float | None = None


@dataclass
class AnswerBatch:
    replies: list[SimulatedReply] = field(default_factory=list)

    @property
    def resume_payload(self) -> dict[str, dict]:
        """Exactly the shape `Command(resume=...)` expects — nothing extra."""
        return {r.gap_id: {"text": r.text, "skip": r.skip} for r in self.replies}

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.replies if r.skip)

    def as_records(self) -> list[dict]:
        return [
            {
                "gap_id": r.gap_id,
                "text": r.text,
                "skip": r.skip,
                "reason": r.reason,
                "matched_gap_ref": r.matched_gap_ref,
                "confidence": r.confidence,
            }
            for r in self.replies
        ]


class Simulator(Protocol):
    mode: str

    def answer(self, questions: list[dict], state: CDCState) -> AnswerBatch: ...

    # Whatever a resumed process must restore to keep answering as if it had
    # never stopped. JSON-serialisable; see run_benchmark.py's simulator.json.
    def get_state(self) -> dict: ...

    def set_state(self, state: dict) -> None: ...


# ---------------------------------------------------------------------------
# Text matching (shared by the oracle and, later, the Phase 4 scorer)
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, accent-folded text — French annotations shouldn't miss a match
    because the agent wrote "delai" where the annotator wrote "délai"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def keyword_score(keywords: list[str], haystack: str) -> float:
    """Share of `keywords` (substring match, phrases allowed) present in `haystack`."""
    if not keywords:
        return 0.0
    hay = normalize(haystack)
    hits = sum(1 for kw in keywords if normalize(kw) in hay)
    return hits / len(keywords)


def _gap_haystack(gap: Gap | None, question_text: str) -> str:
    if gap is None:
        return question_text
    return " ".join([gap.description, gap.category, *gap.section_ids, question_text])


# ---------------------------------------------------------------------------
# Mode A — oracle
# ---------------------------------------------------------------------------


class OracleSimulator:
    """Answers from ground truth, or admits ignorance.

    Runtime gap ids are content hashes computed during the run, so they can't be
    annotated in advance: a question is matched to an annotated gap by content
    (its keywords), not by id.
    """

    mode = "oracle"

    def __init__(self, ground_truth: GroundTruth, threshold: float = MATCH_THRESHOLD):
        self.ground_truth = ground_truth
        self.threshold = threshold

    def get_state(self) -> dict:
        """Nothing to carry: the oracle is a pure function of the question."""
        return {"mode": self.mode}

    def set_state(self, state: dict) -> None:
        return None

    def answer(self, questions: list[dict], state: CDCState) -> AnswerBatch:
        gaps_by_id = {g.id: g for g in state.get("gaps", [])}
        batch = AnswerBatch()

        for q in questions:
            gap_id = q["gap_id"]
            gap = gaps_by_id.get(gap_id)
            haystack = _gap_haystack(gap, q.get("text", ""))

            best = self._best_match(haystack)
            if best is None:
                # Nothing annotated covers this gap: an oracle that guessed here
                # would be crediting the system with information the benchmark
                # never promised.
                batch.replies.append(
                    SimulatedReply(gap_id=gap_id, text="", skip=True, reason="no_ground_truth")
                )
                continue

            expected, score = best
            batch.replies.append(
                SimulatedReply(
                    gap_id=gap_id,
                    text=expected.answer,
                    skip=False,
                    reason="matched",
                    matched_gap_ref=expected.gap_ref,
                    confidence=round(score, 3),
                )
            )
        return batch

    def _best_match(self, haystack: str):
        scored = []
        for expected in self.ground_truth.expected_answers:
            gt_gap = self.ground_truth.gap_by_id(expected.gap_ref)
            keywords = list(expected.keywords) + (list(gt_gap.keywords) if gt_gap else [])
            score = keyword_score(keywords, haystack)
            if score >= self.threshold:
                scored.append((expected, score))
        if not scored:
            return None
        # gap_ref breaks ties so an equally-good match is always resolved the
        # same way across reruns.
        scored.sort(key=lambda pair: (-pair[1], pair[0].gap_ref))
        return scored[0]


# ---------------------------------------------------------------------------
# Mode B — LLM stakeholder
# ---------------------------------------------------------------------------


class SimulatedAnswer(BaseModel):
    """Wire model for one simulated stakeholder reply."""

    knows: bool = Field(description="true only if the answer comes from the listed knowledge")
    answer: str = Field(default="", description="the reply, empty when knows is false")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def build_simulator_prompt(
    profile: StakeholderProfile,
    behavior: Behavior,
    question_text: str,
    gap: Gap | None,
) -> str:
    knowledge = "\n".join(f"- {k}" for k in profile.knowledge) or "- (aucune)"
    unknown = "\n".join(f"- {u}" for u in profile.unknown) or "- (aucun)"
    style = (
        "Réponds de façon brève et approximative, sans chiffrer précisément."
        if behavior.style == "vague"
        else "Réponds de façon précise et concrète, en reprenant les chiffres et les rôles exacts."
    )
    gap_context = ""
    if gap is not None:
        gap_context = (
            f"\nContexte de la lacune ciblée :\n"
            f"- catégorie : {gap.category}\n"
            f"- sévérité : {gap.severity}\n"
            f"- description : {gap.description}\n"
        )

    return f"""Tu simules une partie prenante interrogée pendant la rédaction d'un cahier des charges.

Profil :
- nom : {profile.name}
- rôle : {profile.role}

Connaissances dont tu disposes (ta SEULE source d'information autorisée) :
{knowledge}

Sujets que tu ne maîtrises pas :
{unknown}

RÈGLE ABSOLUE : tu ne peux répondre qu'à partir des connaissances listées ci-dessus.
Si l'information demandée n'y figure pas, même partiellement, mets `knows` à false et
laisse `answer` vide. N'invente jamais un chiffre, un rôle, un délai ou une règle.
Une réponse inventée fausserait l'évaluation du système.

{style}
{gap_context}
Question posée :
"{question_text}"

Renvoie :
- knows : true uniquement si la réponse provient des connaissances listées
- answer : ta réponse en français (vide si knows vaut false)
- confidence : ta confiance dans cette réponse, entre 0 et 1
"""


class StakeholderSimulator:
    """Profile-driven LLM stakeholder (roadmap §14).

    `behavior` decides how it answers once it has decided *whether* it can:
    a vague style, a chance of hedging even on known facts, and — when the
    profile lists them — occasional contradictory statements that give the
    critic something real to catch.
    """

    def __init__(
        self,
        profile: StakeholderProfile,
        *,
        behavior: Behavior | None = None,
        seed: int = 0,
        mode: str = "stakeholder",
    ):
        self.profile = profile
        self.behavior = behavior or profile.behavior
        self.mode = mode
        self._rng = random.Random(seed)
        self._contradictions = list(profile.contradictions)
        self._contradictions_used = 0

    def get_state(self) -> dict:
        """The two things that make this simulator stateful.

        A resumed run that restarted the RNG at the seed would replay dice already
        spent on earlier rounds, so a case answered across two processes would
        diverge from the same case answered in one.
        """
        return {
            "mode": self.mode,
            "rng": self._rng.getstate(),
            "contradictions_used": self._contradictions_used,
        }

    def set_state(self, state: dict) -> None:
        rng = state.get("rng")
        if rng:
            # A JSON round-trip turns random.getstate()'s nested tuples into lists,
            # and setstate() insists on tuples.
            version, internal, gauss = rng
            self._rng.setstate((version, tuple(internal), gauss))
        self._contradictions_used = state.get("contradictions_used", 0)

    def answer(self, questions: list[dict], state: CDCState) -> AnswerBatch:
        gaps_by_id = {g.id: g for g in state.get("gaps", [])}
        batch = AnswerBatch()

        for q in questions:
            gap_id = q["gap_id"]
            gap = gaps_by_id.get(gap_id)
            question_text = q.get("text", "")

            # Draw both dice up front and always in the same order, so a given
            # seed replays identically regardless of what the LLM answers.
            hedges = self._rng.random() < self.behavior.unknown_rate
            contradicts = (
                self.behavior.allow_contradictions
                and self._contradictions_used < len(self._contradictions)
                and self._rng.random() < CONTRADICTION_RATE
            )

            if hedges:
                batch.replies.append(
                    SimulatedReply(gap_id=gap_id, text="", skip=True, reason="hedged")
                )
                continue

            if contradicts:
                statement = self._contradictions[self._contradictions_used]
                self._contradictions_used += 1
                batch.replies.append(
                    SimulatedReply(
                        gap_id=gap_id,
                        text=statement,
                        skip=False,
                        reason="contradiction",
                        confidence=1.0,
                    )
                )
                continue

            prompt = build_simulator_prompt(self.profile, self.behavior, question_text, gap)
            try:
                result = call_structured(prompt, SimulatedAnswer, prompt_id="simulator.answer")
            except Exception as exc:  # a simulator failure must not kill the case
                batch.replies.append(
                    SimulatedReply(
                        gap_id=gap_id, text="", skip=True, reason=f"simulator_error: {exc}"
                    )
                )
                continue

            if not result.knows or not result.answer.strip():
                batch.replies.append(
                    SimulatedReply(
                        gap_id=gap_id,
                        text="",
                        skip=True,
                        reason="unknown",
                        confidence=result.confidence,
                    )
                )
            else:
                batch.replies.append(
                    SimulatedReply(
                        gap_id=gap_id,
                        text=result.answer.strip(),
                        skip=False,
                        reason="answered",
                        confidence=result.confidence,
                    )
                )
        return batch


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODES = ("oracle", "stakeholder", "realistic")

# The adversarial floor applied by --mode realistic, on top of whatever the
# profile authored: vague phrasing, a real chance of "je ne sais pas", and
# contradictions enabled. This is the mode that stress-tests the assumption and
# contradiction machinery rather than the happy path.
REALISTIC_UNKNOWN_RATE = 0.2


def make_simulator(
    mode: str, *, ground_truth: GroundTruth, profile: StakeholderProfile, seed: int = 0
) -> Simulator:
    if mode == "oracle":
        return OracleSimulator(ground_truth)
    if mode == "stakeholder":
        return StakeholderSimulator(profile, seed=seed, mode="stakeholder")
    if mode == "realistic":
        behavior = Behavior(
            style="vague",
            unknown_rate=max(profile.behavior.unknown_rate, REALISTIC_UNKNOWN_RATE),
            allow_contradictions=True,
        )
        return StakeholderSimulator(profile, behavior=behavior, seed=seed, mode="realistic")
    raise ValueError(f"Unknown simulator mode {mode!r}; expected one of {MODES}")
