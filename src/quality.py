"""Quality scoring of a CDC section from the gaps still open on it.

Lives outside src/app.py because two very different callers need it: the
Streamlit UI, which shows a per-section score, and the benchmark scorer
(evals/scoring.py), which measures how much the score improves between the
initial CDC and the final one. Importing src/app.py from a CLI drags in
Streamlit and its page config — 40+ seconds and a wall of "missing
ScriptRunContext" warnings — so the shared part sits here instead.
"""

from __future__ import annotations

# How much each kind of gap costs a section: a contradiction hurts more than a
# missing edge case. Mirrored by _CATEGORY_ORDER in src/agents/gap_filler.py,
# which asks about the expensive ones first.
CATEGORY_WEIGHTS = {
    "contradiction": 2.0,
    "scope": 1.5,
    "functional_ambiguity": 1.4,
    "business_rule": 1.3,
    "acceptance_criteria": 1.2,
    "integration": 1.2,
    "data_model": 1.2,
    "nfr": 1.0,
    "edge_case": 0.8,
}
SEVERITY_WEIGHTS = {
    "blocking": 10,
    "important": 5,
    "nice_to_have": 1,
}


def score_section(gaps: list[dict]) -> float:
    """0–100 for one section: 100 minus a weighted penalty per outstanding gap.

    Each gap is a dict with "category" and "severity" keys (a Gap dumped, or a
    hand-built pair). Clamped at zero — a badly specified section bottoms out
    rather than going negative.
    """
    penalty = 0.0

    for gap in gaps:
        penalty += SEVERITY_WEIGHTS[gap["severity"]] * CATEGORY_WEIGHTS[gap["category"]]

    return max(0.0, 100 - penalty)
