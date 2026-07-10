"""Minimal offline eval harness for the gap_finder and critic agents.

Calls run_gap_finder() / run_critic() directly (no LangGraph, no graph
state machinery) against hand-labeled JSONL cases in evals/datasets/, and
reports crude precision/recall/F1 plus a raw expected-vs-actual diff per
case for manual inspection.

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

from src import config as config_module  # noqa: E402
from src import llm as llm_module  # noqa: E402
from src.agents import critic as critic_module  # noqa: E402
from src.agents import gap_finder as gap_finder_module  # noqa: E402
from src.state import ContextItem, Gap, SectionConfig, SectionStatus  # noqa: E402

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# Only the model needs to change when forcing a provider; base_url/temperature
# stay whatever config/settings.yaml already has.
PROVIDER_DEFAULT_MODEL = {
    "ollama": "gpt-oss:20b",
    "anthropic": "claude-sonnet-5",
}


def override_llm_provider(provider: str) -> None:
    """Force every call_structured() call in the agent modules onto `provider`.

    Agent modules do `from src.llm import call_structured` at import time, so
    the name lives in each agent module's namespace (same pattern used by
    tests/test_graph_flow.py's ScriptedLLM) — patch it there, not on src.llm.
    """
    settings = config_module.load_settings()
    cfg = settings.llm.model_copy(update={"provider": provider, "model": PROVIDER_DEFAULT_MODEL[provider]})
    fixed_llm = llm_module._build_llm(cfg, json_mode=(provider == "ollama"))

    def call_structured_fixed(prompt, model, llm=None, max_retries=2):
        return llm_module.call_structured(prompt, model, llm=fixed_llm, max_retries=max_retries)

    gap_finder_module.call_structured = call_structured_fixed
    critic_module.call_structured = call_structured_fixed


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
        source=d.get("source", "initial_cdc"),
        section_ids=d.get("section_ids", []),
        linked_gap_id=d.get("linked_gap_id"),
        turn_added=d.get("turn_added", 0),
        fresh=fresh,
    )


def build_gap_finder_state(case: dict) -> tuple[dict, str]:
    section = _section_config(case["section"])
    other_sections = [_section_config(s) for s in case.get("other_sections", [])]
    sections_config = [section, *other_sections]

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
# Crude category-only matching (deliberately simple, iterate later)
# ---------------------------------------------------------------------------


def score_case(expected: list[dict], actual: list[dict]) -> dict:
    """Count a hit per expected item matched to an unused actual item of the same category.

    Ignores description text entirely — this is a first pass to catch gross
    regressions (e.g. a category the agent stops finding at all), not a
    precise scorer. Sharpen later if it proves too loose.
    """
    remaining_actual = list(actual)
    hits = 0
    for exp in expected:
        for i, act in enumerate(remaining_actual):
            if act["category"] == exp["category"]:
                hits += 1
                remaining_actual.pop(i)
                break
    return {"hits": hits, "expected": len(expected), "actual": len(actual)}


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
    totals = {"hits": 0, "expected": 0, "actual": 0}
    for case in load_jsonl(path):
        state, section_id = build_gap_finder_state(case)
        result = gap_finder_module.run_gap_finder(state, mode="section", section_id=section_id)
        actual = [{"category": g.category, "description": g.description} for g in result.new_gaps]
        expected = case["expected_gaps"]

        case_score = score_case(expected, actual)
        for k in totals:
            totals[k] += case_score[k]

        print(f"\n-- {case['case_id']} --")
        print("expected:")
        for e in expected:
            print(f"  [{e['category']}] {e['description']}")
        print("actual:")
        for a in actual:
            print(f"  [{a['category']}] {a['description']}")
        print(f"case hits={case_score['hits']}/{case_score['expected']} (actual={case_score['actual']})")

    print_totals(path.name, totals)


def run_critic_dataset(path: Path) -> None:
    print(f"\n=== critic: {path.name} ===")
    totals = {"hits": 0, "expected": 0, "actual": 0}
    for case in load_jsonl(path):
        state, fresh_item_ids = build_critic_state(case)
        result = critic_module.run_critic(state, fresh_item_ids=fresh_item_ids)
        actual = [{"category": "contradiction", "description": g.description} for g in result.new_gaps]
        expected = [{"category": "contradiction", "description": e["description"]} for e in case["expected_contradictions"]]

        case_score = score_case(expected, actual)
        for k in totals:
            totals[k] += case_score[k]

        print(f"\n-- {case['case_id']} --")
        print("expected:")
        for e in expected:
            print(f"  {e['description']}")
        print("actual:")
        for a in actual:
            print(f"  {a['description']}")
        print(f"case hits={case_score['hits']}/{case_score['expected']} (actual={case_score['actual']})")

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
