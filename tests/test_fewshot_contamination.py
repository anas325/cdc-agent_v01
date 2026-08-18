"""The gap-finder few-shots must not quote the gap-finder eval set.

`gap_finder_prompts.py` and `evals/datasets/gap_finder.jsonl` were once built
from the same two reference CDCs in the same commit, and ended up sharing whole
sentences: nine of the ten cases had their excerpt — and, for the fresh cases,
their expected answer — written verbatim into the prompt. The eval then measured
recall of shown text rather than generalisation, in both directions at once: the
contaminated recall cases passed while the two clean ones failed, and the
negative examples primed the model onto exactly the passages they forbade.

A run of five shared words between an annotated case and a few-shot block is not
a coincidence at this length, so that is the tripwire. If this test fails, the
fix is to re-cut the few-shot on material the dataset does not use — never to
loosen the threshold.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from src.agents import gap_finder_prompts as p

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "datasets" / "gap_finder.jsonl"

# Every block that reaches the model, not just the few-shots: a rule phrased by
# quoting a case contaminates it just as effectively as an example does.
PROMPT_BLOCKS = [
    p.FEWSHOT_SECTION,
    p.FEWSHOT_FRESH,
    p.SEVERITY_RUBRIC,
    p.GROUNDING_TEST,
    p.SCOPE_GUARD,
    p.CATEGORY_GUIDE,
    p.GAP_QUALITY_RULES,
    p.SELF_CHECK,
]

NGRAM = 5

# Keys carrying annotator commentary rather than text the model ever sees.
_META_KEYS = {"case_id", "_intent", "_revision", "mode", "reason"}


def _words(text: str) -> list[str]:
    """Lowercased, accent-stripped word list — paraphrase-blind on purpose.

    Only near-verbatim reuse is contamination; discussing the same *topic* in a
    few-shot is legitimate and must not trip the test.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", stripped)


def _ngrams(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k not in _META_KEYS for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_no_eval_case_text_appears_in_the_prompts() -> None:
    prompt_ngrams = set().union(*(_ngrams(block) for block in PROMPT_BLOCKS))

    leaks: list[str] = []
    for case in _cases():
        for text in _strings(case):
            shared = _ngrams(text) & prompt_ngrams
            if shared:
                quoted = ", ".join(" ".join(g) for g in sorted(shared)[:3])
                leaks.append(f"{case['case_id']}: {text[:80]!r} shares [{quoted}]")

    assert not leaks, (
        f"{len(leaks)} eval-case string(s) reused in the gap-finder prompts — "
        "re-cut the few-shot on unused material:\n  " + "\n  ".join(leaks)
    )


def test_the_tripwire_would_actually_fire() -> None:
    """Guard the guard: a planted quote must be detected."""
    planted = max((s for c in _cases() for s in _strings(c)), key=len)
    assert _ngrams(planted) & _ngrams(f"Exemple à ne pas remonter : « {planted} »")
