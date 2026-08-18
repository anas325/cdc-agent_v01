"""Minimal offline eval harness for the gap_finder and critic agents.

Calls run_gap_finder() / run_critic() directly (no LangGraph, no graph
state machinery) against hand-labeled JSONL cases in evals/datasets/, and
reports crude precision/recall/F1 plus a raw expected-vs-actual diff per
case for manual inspection.

The gap_finder dataset carries two kinds of case, told apart by `mode`:

- "section" — one section's context. Recall is scored on `expected_gaps`;
  precision on `forbidden_gaps` (categories this section must never produce,
  each with a `reason`) and `max_gaps` (ceiling on how many gaps are tolerated
  at all). An `expected_gaps` entry may add `must_quote` (a string the
  description has to contain) or `expected_section_ids` (a cross-section gap
  must list every section it spans).
- "fresh" — one open gap plus the answer that came back. Scored on
  `expected.resolved_gap_ids`, `expected.expected_new_gaps` (follow-ups that
  must be raised) and `expected.forbidden_new_gaps` (follow-ups that must not).

Usage:
    python evals/run_evals.py                       # both datasets, config/settings.yaml provider
    python evals/run_evals.py --provider anthropic   # override llm.provider for this run
    python evals/run_evals.py --provider ollama
    python evals/run_evals.py --dataset critic       # only one dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals.harness import override_llm_provider  # noqa: E402
from src.agents import critic as critic_module  # noqa: E402
from src.agents import gap_finder as gap_finder_module  # noqa: E402
from src.state import ContextItem, Gap, SectionConfig, SectionStatus  # noqa: E402

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


# ---------------------------------------------------------------------------
# State builders — turn a JSONL case into a minimal CDCState-shaped dict
# ---------------------------------------------------------------------------


def _section_config(d: dict) -> SectionConfig:
    return SectionConfig(
        id=d["id"],
        title=d.get("title", d["id"]),
        description=d.get("description", ""),
        required=d.get("required", True),
        template_slot=d.get("template_slot", d["id"]),
        completion_hints=d.get("completion_hints", []),
    )


def _context_item(d: dict, idx: int, fresh: bool = False) -> ContextItem:
    return ContextItem(
        id=d.get("id", f"ctx_{idx}"),
        content=d["content"],
        # The dataset writes the prompt's display form ("USER_ANSWER"); the
        # model's Literal is lowercase.
        source=d.get("source", "initial_cdc").lower(),
        section_ids=d.get("section_ids", []),
        linked_gap_id=d.get("linked_gap_id"),
        turn_added=d.get("turn_added", 0),
        fresh=fresh,
    )


def _implied_sections(case: dict, *extra: SectionConfig) -> list[SectionConfig]:
    """Every section the case mentions, declared ones first.

    Fresh cases declare no sections at all, they only tag items with ids.
    coerce_section_ids() drops any id missing from sections_config, so without
    these stubs every predicted gap would come back section-less.
    """
    sections = {s.id: s for s in extra}
    for d in case.get("sections", []):
        sections.setdefault(d["id"], _section_config(d))
    for d in [*case.get("context_items", []), *case.get("fresh_items", [])]:
        for sid in d.get("section_ids", []):
            sections.setdefault(sid, _section_config({"id": sid}))
    return list(sections.values())


def build_gap_finder_state(case: dict) -> tuple[dict, str]:
    section = _section_config(case["section"])
    other_sections = [_section_config(s) for s in case.get("other_sections", [])]
    sections_config = _implied_sections(case, section, *other_sections)

    context_items = [_context_item(d, i) for i, d in enumerate(case.get("context_items", []))]

    gaps = [
        Gap(
            id=f"existing_gap_{i}",
            section_ids=g.get("section_ids", [section.id]),
            category=g["category"],
            description=g["description"],
            severity=g.get("severity", "important"),
            status="open",
        )
        for i, g in enumerate(case.get("existing_gaps", []))
    ]

    state = {"sections_config": sections_config, "context_items": context_items, "gaps": gaps}
    return state, section.id


def build_fresh_state(case: dict) -> tuple[dict, list[str]]:
    existing_items = [_context_item(d, i) for i, d in enumerate(case.get("context_items", []))]
    fresh_items = [_context_item(d, 1000 + i, fresh=True) for i, d in enumerate(case.get("fresh_items", []))]
    sections_config = _implied_sections(case)

    gaps = []
    for g in case.get("gaps", []):
        # A fresh case pins the gap by id, not by section; take its scope from
        # the answers pointing at it.
        linked = [it for it in fresh_items if it.linked_gap_id == g["id"]]
        section_ids = g.get("section_ids") or list(
            dict.fromkeys(sid for it in linked for sid in it.section_ids)
        )
        gaps.append(
            Gap(
                id=g["id"],
                section_ids=section_ids,
                category=g["category"],
                description=g["description"],
                severity=g.get("severity", "important"),
                status=g.get("status", "open"),
            )
        )

    state = {
        "sections_config": sections_config,
        "context_items": [*existing_items, *fresh_items],
        "gaps": gaps,
    }
    return state, [it.id for it in fresh_items]


def build_critic_state(case: dict) -> tuple[dict, list[str]]:
    sections_config = [_section_config(s) for s in case["sections"]]
    complete_ids = set(case.get("complete_section_ids", []))
    section_statuses = {
        s.id: SectionStatus(section_id=s.id, status="complete" if s.id in complete_ids else "in_progress")
        for s in sections_config
    }

    existing_items = [_context_item(d, i) for i, d in enumerate(case.get("context_items", []))]
    fresh_items = [_context_item(d, 1000 + i, fresh=True) for i, d in enumerate(case.get("fresh_items", []))]

    state = {
        "sections_config": sections_config,
        "context_items": [*existing_items, *fresh_items],
        "section_statuses": section_statuses,
    }
    return state, [it.id for it in fresh_items]


# ---------------------------------------------------------------------------
# Predictions — flatten a GapFinderResult into plain dicts
# ---------------------------------------------------------------------------


def predicted_gaps(result) -> list[dict]:
    """The result's gaps, with follow_up_of_gap_id recovered from the decisions.

    Gap doesn't carry follow_up_of_gap_id — the agent only records it on the
    matching decision-log entry (output_ids = [gap.id]).
    """
    follow_ups: dict[str, str | None] = {}
    for dec in result.decisions:
        for gid in dec.output_ids:
            follow_ups[gid] = dec.details.get("follow_up_of_gap_id")
    return [
        {
            "id": g.id,
            "category": g.category,
            "severity": g.severity,
            "description": g.description,
            "section_ids": g.section_ids,
            "follow_up_of_gap_id": follow_ups.get(g.id),
        }
        for g in result.new_gaps
    ]


def format_gap(g: dict) -> str:
    tail = f" follow_up_of={g['follow_up_of_gap_id']}" if g.get("follow_up_of_gap_id") else ""
    return f"[{g.get('category')}/{g.get('severity')}] sections={g.get('section_ids', [])}{tail}: {g['description']}"


# ---------------------------------------------------------------------------
# Matching (deliberately simple, iterate later)
# ---------------------------------------------------------------------------


def matches(spec: dict, pred: dict) -> bool:
    """Does `pred` satisfy the annotation `spec`?

    Every field of the spec is optional and only constrains when present, so
    the same predicate serves both expectations and prohibitions. `must_quote`
    is the one hard filter: a gap that doesn't name the offending term is not
    the gap that was annotated. A prediction with no follow_up_of_gap_id at all
    still matches — small models routinely omit the field.
    """
    if spec.get("category") and spec["category"] != pred["category"]:
        return False
    if spec.get("must_quote") and spec["must_quote"] not in pred["description"]:
        return False
    want_follow_up = spec.get("follow_up_of_gap_id")
    if want_follow_up and pred.get("follow_up_of_gap_id") not in (None, want_follow_up):
        return False
    return True


def match_gaps(expected: list[dict], actual: list[dict]) -> tuple[list[tuple[dict, dict | None]], list[dict]]:
    """Greedily pair each expectation with one unused prediction.

    Returns the pairs (prediction None when missed) and the predictions left
    over — the latter are candidate false positives, always printed for review.
    """
    remaining = list(actual)
    pairs: list[tuple[dict, dict | None]] = []
    for exp in expected:
        hit = next((a for a in remaining if matches(exp, a)), None)
        if hit is not None:
            remaining.remove(hit)
        pairs.append((exp, hit))
    return pairs, remaining


def aggregate(totals: dict) -> dict:
    hits, expected, actual = totals["hits"], totals["expected"], totals["actual"]
    precision = hits / actual if actual else 1.0
    recall = hits / expected if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def print_totals(name: str, totals: dict) -> None:
    metrics = aggregate(totals)
    print(
        f"\n{name} TOTALS: precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} "
        f"f1={metrics['f1']:.2f} (hits={totals['hits']}, expected={totals['expected']}, actual={totals['actual']})"
    )
    extras = [f"{k}={totals[k]}" for k in ("forbidden", "over_cap", "wrong_sections") if totals.get(k)]
    if extras:
        print("  violations: " + ", ".join(extras))


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------


def run_section_case(case: dict, totals: dict) -> None:
    state, section_id = build_gap_finder_state(case)
    result = gap_finder_module.run_gap_finder(state, mode="section", section_id=section_id)
    actual = predicted_gaps(result)
    expected = case.get("expected_gaps", [])

    pairs, unmatched = match_gaps(expected, actual)
    hits = sum(1 for _, a in pairs if a is not None)

    forbidden = [f for f in case.get("forbidden_gaps", [])]
    violations = [(f, a) for f in forbidden for a in unmatched if matches(f, a)]

    max_gaps = case.get("max_gaps")
    over_cap = max(0, len(actual) - max_gaps) if max_gaps is not None else 0

    wrong_sections = [
        (e, a)
        for e, a in pairs
        if a is not None and e.get("expected_section_ids")
        and not set(e["expected_section_ids"]).issubset(a["section_ids"])
    ]

    totals["hits"] += hits
    totals["expected"] += len(expected)
    totals["actual"] += len(actual)
    totals["forbidden"] += len(violations)
    totals["over_cap"] += 1 if over_cap else 0
    totals["wrong_sections"] += len(wrong_sections)
    totals["cases"] += 1

    print(f"\n-- {case['case_id']} -- (section mode, section={section_id})")
    print(f"expected ({len(expected)}):")
    for exp, act in pairs:
        mark = "HIT " if act is not None else "MISS"
        print(f"  [{mark}] [{exp.get('category')}/{exp.get('severity', '?')}] {exp['description']}")
    if not expected:
        print("  (none — precision control case)")
    print(f"actual ({len(actual)}):")
    for a in actual:
        print(f"  {format_gap(a)}")
    if not actual:
        print("  (none)")
    for spec, a in violations:
        print(f"  !! forbidden category '{spec['category']}' ({spec.get('reason', '')}): {a['description']}")
    if over_cap:
        print(f"  !! cap exceeded: {len(actual)} gaps > max_gaps={max_gaps}")
    for exp, a in wrong_sections:
        print(f"  !! sections {a['section_ids']} miss expected {exp['expected_section_ids']}")
    print(f"case hits={hits}/{len(expected)} (actual={len(actual)}, unmatched={len(unmatched)})")


def run_fresh_case(case: dict, totals: dict) -> None:
    state, fresh_item_ids = build_fresh_state(case)
    result = gap_finder_module.run_gap_finder(state, mode="fresh", fresh_item_ids=fresh_item_ids)
    actual = predicted_gaps(result)

    expected = case.get("expected", {})
    exp_resolved = set(expected.get("resolved_gap_ids", []))
    act_resolved = set(result.resolved_gap_ids)
    resolved_hits = len(exp_resolved & act_resolved)
    spurious_resolved = act_resolved - exp_resolved

    exp_new = expected.get("expected_new_gaps", [])
    pairs, unmatched = match_gaps(exp_new, actual)
    new_hits = sum(1 for _, a in pairs if a is not None)

    forbidden = expected.get("forbidden_new_gaps", [])
    violations = [(f, a) for f in forbidden for a in unmatched if matches(f, a)]

    totals["resolved_hits"] += resolved_hits
    totals["resolved_expected"] += len(exp_resolved)
    totals["resolved_spurious"] += len(spurious_resolved)
    totals["hits"] += new_hits
    totals["expected"] += len(exp_new)
    totals["actual"] += len(actual)
    totals["forbidden"] += len(violations)
    totals["cases"] += 1

    print(f"\n-- {case['case_id']} -- (fresh mode)")
    print(f"resolved: expected={sorted(exp_resolved)} actual={sorted(act_resolved)}")
    for gid in sorted(exp_resolved - act_resolved):
        print(f"  !! not resolved: {gid}")
    for gid in sorted(spurious_resolved):
        print(f"  !! resolved but shouldn't be: {gid}")
    print(f"expected follow-up gaps ({len(exp_new)}):")
    for exp, act in pairs:
        mark = "HIT " if act is not None else "MISS"
        print(f"  [{mark}] {exp.get('description', exp)}")
    if not exp_new:
        print("  (none)")
    print(f"actual ({len(actual)}):")
    for a in actual:
        print(f"  {format_gap(a)}")
    if not actual:
        print("  (none)")
    for spec, a in violations:
        print(f"  !! forbidden follow-up ({spec.get('reason', '')}): {a['description']}")
    print(
        f"case resolved={resolved_hits}/{len(exp_resolved)} "
        f"follow_ups={new_hits}/{len(exp_new)} (actual={len(actual)}, unmatched={len(unmatched)})"
    )


def print_fresh_totals(name: str, totals: dict) -> None:
    metrics = aggregate(totals)
    print(
        f"\n{name} TOTALS: resolutions={totals['resolved_hits']}/{totals['resolved_expected']} "
        f"(spurious={totals['resolved_spurious']}) | follow-ups precision={metrics['precision']:.2f} "
        f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f} "
        f"(hits={totals['hits']}, expected={totals['expected']}, actual={totals['actual']})"
    )
    if totals["forbidden"]:
        print(f"  violations: forbidden={totals['forbidden']}")


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_gap_finder_dataset(path: Path) -> None:
    print(f"\n=== gap_finder: {path.name} ===")
    section_totals = {
        "hits": 0, "expected": 0, "actual": 0,
        "forbidden": 0, "over_cap": 0, "wrong_sections": 0, "cases": 0,
    }
    fresh_totals = {
        "hits": 0, "expected": 0, "actual": 0, "forbidden": 0, "cases": 0,
        "resolved_hits": 0, "resolved_expected": 0, "resolved_spurious": 0,
    }

    for case in load_jsonl(path):
        if case.get("mode", "section") == "fresh":
            run_fresh_case(case, fresh_totals)
        else:
            run_section_case(case, section_totals)

    if section_totals["cases"]:
        print_totals(f"{path.name} [section]", section_totals)
    if fresh_totals["cases"]:
        print_fresh_totals(f"{path.name} [fresh]", fresh_totals)


def run_critic_dataset(path: Path) -> None:
    print(f"\n=== critic: {path.name} ===")
    totals = {"hits": 0, "expected": 0, "actual": 0, "cases": 0}
    for case in load_jsonl(path):
        state, fresh_item_ids = build_critic_state(case)
        result = critic_module.run_critic(state, fresh_item_ids=fresh_item_ids)
        actual = [{"category": "contradiction", "description": g.description} for g in result.new_gaps]
        expected = [{"category": "contradiction", "description": e["description"]} for e in case["expected_contradictions"]]

        pairs, unmatched = match_gaps(expected, actual)
        hits = sum(1 for _, a in pairs if a is not None)
        totals["hits"] += hits
        totals["expected"] += len(expected)
        totals["actual"] += len(actual)
        totals["cases"] += 1

        print(f"\n-- {case['case_id']} --")
        print("expected:")
        for e in expected:
            print(f"  {e['description']}")
        if not expected:
            print("  (none — precision control case)")
        print("actual:")
        for a in actual:
            print(f"  {a['description']}")
        if not actual:
            print("  (none)")
        print(f"case hits={hits}/{len(expected)} (actual={len(actual)}, unmatched={len(unmatched)})")

    print_totals(path.name, totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["ollama", "anthropic"], help="override llm.provider for this run")
    parser.add_argument(
        "--dataset", choices=["gap_finder", "critic", "all"], default="all", help="which dataset(s) to run"
    )
    args = parser.parse_args()

    if args.provider:
        override_llm_provider(args.provider)

    if args.dataset in ("gap_finder", "all"):
        run_gap_finder_dataset(DATASETS_DIR / "gap_finder.jsonl")
    if args.dataset in ("critic", "all"):
        run_critic_dataset(DATASETS_DIR / "critic.jsonl")


if __name__ == "__main__":
    main()
