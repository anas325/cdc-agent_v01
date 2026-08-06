"""Deterministic French text matching, used to recognise a question we already asked.

The dedup gate is an LLM judgement and can only answer "le contexte répond-il
déjà ?" — for a live contradiction the honest answer is "non", so the gate alone
cannot notice that the *same question* has already gone out five turns in a row.
These helpers give the gap-filler a cheap, deterministic second opinion that does
not depend on the model's mood.

Comparison is by **containment**, not Jaccard: a repeat is typically a narrowed
restatement ("Quelle durée doit être retenue ?" after the full question), so the
shorter side is nearly a subset of the longer one and Jaccard would score it low.
"""

from __future__ import annotations

import re
import unicodedata

# Entity ids are rendered into prompts (see src/context_utils.py), so drafted
# questions quote them. They churn every turn and would make two restatements of
# the same question look different — strip them before comparing.
_ENTITY_ID_RE = re.compile(r"\b(?:ctx|gap|dec|q)_[0-9a-f]{6,}\b")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Function words carry no topic signal; keeping them would inflate every score.
_STOPWORDS = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur
    lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que
    qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y
    est sont etre ete soit etait sera doit doivent devra devrait peut peuvent
    quel quelle quels quelles quoi comment pourquoi ou dont
    vous-avez avez as ai avoir avons ont
    ci la-bas cet cette celui celle
    plus moins tres bien alors donc car si comme
    preciser precisez pouvez pourriez souhaitez indiquer indiquez definir
    """.split()
)

# Below this many shared content words, "same question" is indistinguishable
# from "same topic", so containment is not trustworthy and we require equality.
_MIN_DISTINGUISHING_TOKENS = 4

DEFAULT_THRESHOLD = 0.75


def normalize(text: str) -> str:
    """Lowercase, accent-folded text — "delai" must match "délai"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def strip_entity_ids(text: str) -> str:
    """Drop `ctx_…` / `gap_…` tokens, which change every turn."""
    return _ENTITY_ID_RE.sub(" ", text)


def content_tokens(text: str) -> frozenset[str]:
    """Topic-bearing words of `text`: normalized, id-free, stopword-free."""
    words = _WORD_RE.findall(normalize(strip_entity_ids(text)))
    return frozenset(w for w in words if len(w) > 1 and w not in _STOPWORDS)


def containment(a: str, b: str) -> float:
    """Share of the *shorter* question's content words present in the other."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_near_duplicate(a: str, b: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    ta, tb = content_tokens(a), content_tokens(b)
    smaller = min(len(ta), len(tb)) if ta and tb else 0
    if smaller < 2:
        # A single shared content word names a topic, not a question. There is
        # nothing here to tell two questions apart, so decline to judge rather
        # than suppress a question that was never asked.
        return False
    if smaller < _MIN_DISTINGUISHING_TOKENS:
        # Too few words for a ratio to mean anything: require an exact match.
        return ta == tb
    return len(ta & tb) / smaller >= threshold


def is_too_vague(text: str) -> bool:
    """A question with almost no content words ("Quelle durée doit être retenue ?").

    Worth catching on its own: such a question is unanswerable out of context and
    is what the dedup gate's "rewrite on the missing part only" branch degenerates
    into when it keeps narrowing the same question.
    """
    return len(content_tokens(text)) < _MIN_DISTINGUISHING_TOKENS


def first_near_duplicate(
    candidate: str, previous: list[str], threshold: float = DEFAULT_THRESHOLD
) -> int | None:
    """Index of the first entry of `previous` that `candidate` restates, if any."""
    for i, earlier in enumerate(previous):
        if is_near_duplicate(candidate, earlier, threshold):
            return i
    return None
