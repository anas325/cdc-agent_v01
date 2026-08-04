"""Tests for the benchmark harness: dataset, simulators, isolation, batch runner.

The dataset tests are the ones that pay off daily — an annotation typo (a
category that isn't a GapCategory, a gap_ref pointing nowhere, an
expected_evidence naming a file that was renamed) would otherwise only surface
partway through a long batch run.

The runner test drives the real compiled graph with ScriptedLLM, reusing the
fixtures from test_graph_flow.py (same pattern as test_decision_log.py), so it
covers the interrupt/resume cycle without any network access.
"""

from __future__ import annotations

import csv
import json

import pytest

from evals import harness
from evals.dataset import (
    BENCHMARK_DIR,
    Behavior,
    DatasetError,
    GroundTruth,
    StakeholderProfile,
    dataset_version,
    load_benchmark,
)
from evals.simulator import (
    OracleSimulator,
    SimulatedAnswer,
    StakeholderSimulator,
    keyword_score,
    make_simulator,
    normalize,
)
from src.config import load_sections, load_settings
from src.state import Gap, GapCategory, GapSeverity, SectionConfig, SectionStatus
from tests.test_graph_flow import (  # noqa: F401 - fixtures are used by name
    ScriptedLLM,
    graph,
    no_rag_hits,
    scripted_llm,
)

CATEGORIES = set(GapCategory.__args__)
SEVERITIES = set(GapSeverity.__args__)


@pytest.fixture(scope="module")
def cases():
    return load_benchmark()


# ---------------------------------------------------------------------------
# 1. The dataset itself
# ---------------------------------------------------------------------------


def test_every_manifest_case_loads_and_validates(cases):
    assert len(cases) >= 10, "the benchmark should hold at least the ten annotated CDCs"
    assert dataset_version() == "v1"

    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids), "duplicate case ids in the manifest"

    for case in cases:
        assert case.initial_cdc_text.strip(), f"{case.case_id}: empty initial_cdc.md"
        assert case.ground_truth.gaps, f"{case.case_id}: no annotated gaps"
        assert case.stakeholder.name, f"{case.case_id}: stakeholder profile has no name"


def test_annotations_use_only_valid_literals_and_section_ids(cases):
    valid_sections = {s.id for s in load_sections()}
    for case in cases:
        for gap in case.ground_truth.gaps:
            assert gap.category in CATEGORIES, f"{case.case_id}/{gap.id}"
            assert gap.severity in SEVERITIES, f"{case.case_id}/{gap.id}"
            assert gap.section_id in valid_sections, f"{case.case_id}/{gap.id}"
            assert gap.keywords, f"{case.case_id}/{gap.id}: keywords drive matching, don't leave them empty"


def test_rag_resolvable_gaps_cite_a_document_that_exists(cases):
    """A `resolvable_by: rag` gap is a promise that retrieval *can* close it."""
    seen_rag_case = False
    for case in cases:
        names = set(case.source_doc_names)
        for gap in case.ground_truth.gaps:
            if gap.resolvable_by != "rag":
                continue
            seen_rag_case = True
            assert gap.expected_evidence is not None, f"{case.case_id}/{gap.id}"
            assert gap.expected_evidence.document in names, f"{case.case_id}/{gap.id}"
            body = (case.source_docs_dir / gap.expected_evidence.document).read_text(encoding="utf-8")
            if gap.expected_evidence.quote:
                assert gap.expected_evidence.quote in body, (
                    f"{case.case_id}/{gap.id}: quoted evidence is not in "
                    f"{gap.expected_evidence.document} — the annotation has drifted"
                )
    assert seen_rag_case, "no RAG-resolvable gap in the whole benchmark: retrieval would never be measured"


def test_expected_answers_resolve_to_annotated_gaps(cases):
    for case in cases:
        gt = case.ground_truth
        for expected in gt.expected_answers:
            gap = gt.gap_by_id(expected.gap_ref)
            assert gap is not None, f"{case.case_id}: dangling gap_ref {expected.gap_ref}"
            # Answering a RAG-resolvable gap from the stakeholder would credit
            # the system for retrieval it never had to do.
            assert gap.resolvable_by != "rag", (
                f"{case.case_id}/{expected.gap_ref}: a gap the documents can answer "
                f"should not also carry a stakeholder answer"
            )
            assert expected.answer.strip()


def test_benchmark_holds_a_negative_control(cases):
    """At least one case must be well-specified, or precision is never tested."""
    smallest = min(cases, key=lambda c: len(c.ground_truth.gaps))
    assert len(smallest.ground_truth.gaps) <= 5
    assert not smallest.ground_truth.contradictions


def test_unknown_case_id_is_rejected():
    with pytest.raises(DatasetError, match="Unknown case id"):
        load_benchmark(case_ids=["cdc_999_does_not_exist"])


def test_case_id_must_match_its_directory(tmp_path):
    case_dir = tmp_path / "cdc_042_mismatch"
    (case_dir / "source_docs").mkdir(parents=True)
    (case_dir / "initial_cdc.md").write_text("# CDC", encoding="utf-8")
    (case_dir / "stakeholder.yaml").write_text("name: X\nrole: Y\n", encoding="utf-8")
    (case_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "case_id": "some_other_id",
                "title": "T",
                "dataset_version": "v1",
                "annotator": "test",
                "gaps": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.yaml").write_text(
        "dataset_version: v1\ncases:\n  - id: cdc_042_mismatch\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="expected the directory name"):
        load_benchmark(root=tmp_path)


# ---------------------------------------------------------------------------
# 2. Oracle simulator
# ---------------------------------------------------------------------------


def _gap(gap_id: str, description: str, section_id: str = "functional") -> Gap:
    return Gap(
        id=gap_id,
        section_ids=[section_id],
        category="business_rule",
        description=description,
        severity="blocking",
    )


def _ground_truth() -> GroundTruth:
    return GroundTruth.model_validate(
        {
            "case_id": "unit",
            "title": "unit",
            "dataset_version": "v1",
            "annotator": "test",
            "gaps": [
                {
                    "id": "GT-GAP-001",
                    "section_id": "functional",
                    "category": "business_rule",
                    "severity": "blocking",
                    "description": "Le seuil d'alerte n'est pas défini.",
                    "keywords": ["seuil", "alerte"],
                },
                {
                    "id": "GT-GAP-002",
                    "section_id": "technical",
                    "category": "integration",
                    "severity": "blocking",
                    "description": "L'intégration SAP n'est pas décrite.",
                    "keywords": ["sap", "synchronisation"],
                },
            ],
            "expected_answers": [
                {
                    "gap_ref": "GT-GAP-001",
                    "answer": "Le seuil est la consommation moyenne de 7 jours.",
                    "keywords": ["seuil"],
                },
                {
                    "gap_ref": "GT-GAP-002",
                    "answer": "Export SFTP nocturne vers SAP.",
                    "keywords": ["sap"],
                },
            ],
        }
    )


def test_oracle_answers_the_matching_annotated_gap():
    sim = OracleSimulator(_ground_truth())
    state = {"gaps": [_gap("g1", "Le seuil d'alerte de stock faible dépend du produit.")]}

    batch = sim.answer([{"gap_id": "g1", "text": "Comment est calculé le seuil d'alerte ?"}], state)

    (reply,) = batch.replies
    assert reply.matched_gap_ref == "GT-GAP-001"
    assert reply.skip is False
    assert "consommation moyenne" in reply.text
    assert batch.resume_payload == {"g1": {"text": reply.text, "skip": False}}


def test_oracle_says_it_does_not_know_when_nothing_is_annotated():
    """Guessing here would credit the system with information the benchmark never promised."""
    sim = OracleSimulator(_ground_truth())
    state = {"gaps": [_gap("g9", "La politique de sauvegarde n'est pas définie.")]}

    batch = sim.answer([{"gap_id": "g9", "text": "Quelle est la politique de sauvegarde ?"}], state)

    (reply,) = batch.replies
    assert reply.skip is True
    assert reply.reason == "no_ground_truth"
    assert reply.matched_gap_ref is None
    assert batch.unknown_count == 1


def test_oracle_is_accent_insensitive_and_deterministic():
    assert normalize("Délai de réponse") == "delai de reponse"
    assert keyword_score(["delai"], "Quel est le délai ?") == 1.0

    sim = OracleSimulator(_ground_truth())
    state = {"gaps": [_gap("g1", "L'intégration SAP et la synchronisation ne sont pas décrites.")]}
    question = [{"gap_id": "g1", "text": "Comment se fait la synchronisation avec SAP ?"}]

    first = sim.answer(question, state)
    second = sim.answer(question, state)
    assert first.resume_payload == second.resume_payload
    assert first.replies[0].matched_gap_ref == "GT-GAP-002"


# ---------------------------------------------------------------------------
# 3. LLM stakeholder simulator
# ---------------------------------------------------------------------------


def _profile(**overrides) -> StakeholderProfile:
    data = {
        "name": "Responsable logistique",
        "role": "Business Owner",
        "knowledge": ["Seuls les responsables magasin peuvent modifier le stock."],
        "unknown": ["Le SLA fournisseur"],
        "contradictions": ["Tous les employés peuvent modifier le stock."],
    }
    data.update(overrides)
    return StakeholderProfile.model_validate(data)


def test_stakeholder_prompt_carries_the_knowledge_and_the_refusal_rule(monkeypatch):
    seen: dict = {}

    def fake_call(prompt, model, **kwargs):
        seen["prompt"] = prompt
        seen["prompt_id"] = kwargs.get("prompt_id")
        return SimulatedAnswer(knows=True, answer="Les responsables magasin.", confidence=0.9)

    monkeypatch.setattr("evals.simulator.call_structured", fake_call)

    sim = StakeholderSimulator(_profile(), seed=0)
    sim.answer([{"gap_id": "g1", "text": "Qui peut modifier le stock ?"}], {"gaps": []})

    assert "Seuls les responsables magasin peuvent modifier le stock." in seen["prompt"]
    assert "knows` à false" in seen["prompt"], "the refusal rule must be in the prompt"
    assert "Le SLA fournisseur" in seen["prompt"]
    # Registered in src/prompts.py, so the call is versioned and cacheable.
    assert seen["prompt_id"] == "simulator.answer"


def test_stakeholder_that_does_not_know_maps_to_skip(monkeypatch):
    monkeypatch.setattr(
        "evals.simulator.call_structured",
        lambda prompt, model, **kw: SimulatedAnswer(knows=False, answer="", confidence=0.1),
    )
    sim = StakeholderSimulator(_profile(), seed=0)

    batch = sim.answer([{"gap_id": "g1", "text": "Quel est le SLA fournisseur ?"}], {"gaps": []})

    (reply,) = batch.replies
    assert reply.skip is True
    assert reply.reason == "unknown"
    # skip=True is exactly what the UI's "Je ne sais pas" sends, so the graph
    # builds an assumption from it.
    assert batch.resume_payload == {"g1": {"text": "", "skip": True}}


def test_simulator_failure_degrades_to_skip_instead_of_killing_the_case(monkeypatch):
    def boom(prompt, model, **kw):
        raise RuntimeError("backend down")

    monkeypatch.setattr("evals.simulator.call_structured", boom)
    sim = StakeholderSimulator(_profile(), seed=0)

    batch = sim.answer([{"gap_id": "g1", "text": "?"}], {"gaps": []})

    assert batch.replies[0].skip is True
    assert "simulator_error" in batch.replies[0].reason


def test_realistic_mode_forces_vagueness_hedging_and_contradictions():
    sim = make_simulator(
        "realistic", ground_truth=_ground_truth(), profile=_profile(), seed=3
    )
    assert sim.behavior.style == "vague"
    assert sim.behavior.unknown_rate > 0
    assert sim.behavior.allow_contradictions is True


def test_realistic_mode_is_reproducible_for_a_given_seed(monkeypatch):
    monkeypatch.setattr(
        "evals.simulator.call_structured",
        lambda prompt, model, **kw: SimulatedAnswer(knows=True, answer="oui", confidence=0.5),
    )
    questions = [{"gap_id": f"g{i}", "text": "?"} for i in range(8)]

    def run(seed):
        sim = make_simulator("realistic", ground_truth=_ground_truth(), profile=_profile(), seed=seed)
        return [(r.reason, r.text) for r in sim.answer(questions, {"gaps": []}).replies]

    assert run(11) == run(11)


def test_contradiction_injection_uses_the_profile_statements(monkeypatch):
    monkeypatch.setattr(
        "evals.simulator.call_structured",
        lambda prompt, model, **kw: SimulatedAnswer(knows=True, answer="réponse fidèle", confidence=1.0),
    )
    profile = _profile()
    behavior = Behavior(style="vague", unknown_rate=0.0, allow_contradictions=True)
    sim = StakeholderSimulator(profile, behavior=behavior, seed=1, mode="realistic")

    replies = sim.answer([{"gap_id": f"g{i}", "text": "?"} for i in range(20)], {"gaps": []}).replies

    injected = [r for r in replies if r.reason == "contradiction"]
    assert injected, "allow_contradictions should eventually inject one over 20 questions"
    assert injected[0].text in profile.contradictions
    # Only as many as the profile actually lists.
    assert len(injected) <= len(profile.contradictions)


def test_unknown_simulator_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown simulator mode"):
        make_simulator("wishful", ground_truth=_ground_truth(), profile=_profile())


# ---------------------------------------------------------------------------
# 4. Per-case isolation
# ---------------------------------------------------------------------------


def test_isolate_redirects_settings_and_restores_them(tmp_path):
    before = load_settings()
    patched = harness.case_settings(
        before,
        source_dir=tmp_path / "docs",
        persist_dir=tmp_path / "chroma",
        output_dir=tmp_path / "out",
        top_k=9,
    )

    with harness.isolate(patched):
        inside = load_settings()
        assert inside.rag.top_k == 9
        assert inside.quarto.output_dir.endswith("out")
        assert inside.rag.source_dir.endswith("docs")
        # Everything not overridden is carried over untouched.
        assert inside.llm.model == before.llm.model

    assert load_settings() is before


def test_isolate_restores_settings_even_on_exception(tmp_path):
    before = load_settings()
    patched = harness.case_settings(
        before, source_dir=tmp_path, persist_dir=tmp_path, output_dir=tmp_path
    )

    with pytest.raises(RuntimeError):
        with harness.isolate(patched):
            raise RuntimeError("case blew up")

    assert load_settings() is before


def test_output_written_under_isolation_does_not_touch_the_app_output_dir(tmp_path):
    """The graph's document writers read quarto.output_dir at call time."""
    from src.agents import final_validator

    base = load_settings()
    out = tmp_path / "artifacts"
    patched = harness.case_settings(
        base, source_dir=tmp_path, persist_dir=tmp_path, output_dir=out
    )

    state = {
        "gaps": [],
        "context_items": [],
        "section_statuses": {},
        "turn": 1,
        "stop_reason": None,
        "decision_log": [],
    }
    with harness.isolate(patched):
        path = final_validator.write_qa_report(state, unmapped=[], final_contradictions=[])

    assert path.parent == out
    assert path.exists()


def test_provider_override_accepts_prompt_id(monkeypatch):
    """Agents always pass prompt_id=; a replacement that drops it raises TypeError."""
    import importlib

    captured: dict = {}

    # override_llm_provider rebinds the name inside each agent module, which
    # monkeypatch can't undo on its own — register the originals first so the
    # rebinding doesn't leak into the rest of the suite.
    for path in harness.AGENT_MODULE_PATHS:
        module = importlib.import_module(path)
        monkeypatch.setattr(module, "call_structured", module.call_structured)

    monkeypatch.setattr(harness.llm_module, "_build_llm", lambda cfg, json_mode=False: object())

    def fake_call_structured(prompt, model, llm=None, max_retries=2, *, prompt_id=None):
        captured["prompt_id"] = prompt_id
        return "ok"

    monkeypatch.setattr(harness.llm_module, "call_structured", fake_call_structured)
    harness.override_llm_provider("ollama")

    from src.agents import gap_finder

    assert gap_finder.call_structured("p", object, prompt_id="gap_finder.section") == "ok"
    assert captured["prompt_id"] == "gap_finder.section"


def test_run_manifest_records_what_a_rerun_needs(tmp_path):
    settings = load_settings()
    manifest = harness.run_manifest(
        run_id="unit",
        dataset_version="v1",
        case_ids=["cdc_001_smartstock"],
        simulator_mode="oracle",
        seed=0,
        settings=settings,
        started_at="2026-01-01T00:00:00+00:00",
    )

    assert manifest["dataset_version"] == "v1"
    assert manifest["simulator_mode"] == "oracle"
    assert manifest["llm"]["model"] == settings.llm.model
    assert manifest["rag"]["chunk_size"] == 1200
    # The whole prompt registry, so a version bump shows up when diffing runs.
    from src import prompts

    assert manifest["prompt_versions"] == dict(prompts.PROMPT_VERSIONS)
    assert manifest["config_hashes"]["sections.yaml"]


# ---------------------------------------------------------------------------
# 5. The batch runner, end to end on the real graph
# ---------------------------------------------------------------------------


def test_runner_drives_a_case_through_the_interrupt_cycle(
    tmp_path, monkeypatch, graph, scripted_llm, no_rag_hits  # noqa: F811
):
    """One gap -> one question -> simulator answers -> graph resumes and ends."""
    from src.agents.critic import CriticOutput
    from src.agents.final_validator import FinalCheckOutput
    from src.agents.gap_filler import QuestionDraft
    from src.agents.gap_finder import GapCandidate, GapFinderOutput
    from src.agents.orchestrator import DedupVerdict
    from src.agents.synthesizer import SlotDraft
    from evals import run_benchmark

    sections = [
        SectionConfig(
            id="functional",
            title="Spécifications fonctionnelles",
            description="d",
            required=True,
            template_slot="functional_spec",
        )
    ]
    monkeypatch.setattr("src.graph.load_sections", lambda: sections)
    monkeypatch.setattr("src.graph.ingest_source_docs", lambda: 0)
    monkeypatch.setattr(run_benchmark, "load_sections", lambda: sections)
    # The shared fixture stops at the four loop agents; synthesis and final
    # validation also call the LLM, and this test runs all the way to END.
    monkeypatch.setattr("src.agents.synthesizer.call_structured", scripted_llm)
    monkeypatch.setattr("src.agents.final_validator.call_structured", scripted_llm)
    monkeypatch.setattr("src.agents.synthesizer.subprocess.run", lambda *a, **kw: None)

    found = GapFinderOutput(
        new_gaps=[
            GapCandidate(
                section_ids=["functional"],
                category="business_rule",
                description="Le seuil d'alerte n'est pas défini.",
                severity="blocking",
            )
        ],
        resolved_gap_ids=[],
        section_complete=False,
    )
    # After the answer the gap is no longer `open`, so the section completes on
    # its own; every later scan just has to find nothing new.
    quiet = GapFinderOutput(new_gaps=[], resolved_gap_ids=[], section_complete=None)
    complete = GapFinderOutput(new_gaps=[], resolved_gap_ids=[], section_complete=True)
    scripted_llm.add(GapFinderOutput, found, quiet, quiet, complete, complete, complete, complete)
    scripted_llm.add(QuestionDraft, QuestionDraft(question_text="Comment est calculé le seuil d'alerte ?"))
    scripted_llm.add(DedupVerdict, DedupVerdict(), DedupVerdict(), DedupVerdict())
    scripted_llm.add(
        CriticOutput, *[CriticOutput(contradictions=[]) for _ in range(4)]
    )
    scripted_llm.add(SlotDraft, SlotDraft(prose="Le seuil d'alerte est défini par référence produit."))
    scripted_llm.add(FinalCheckOutput, FinalCheckOutput(contradictions=[]))

    case = load_benchmark(case_ids=["cdc_001_smartstock"])[0]
    simulator = OracleSimulator(case.ground_truth)

    out = tmp_path / "results"
    base = load_settings()
    settings = harness.case_settings(
        base,
        source_dir=case.source_docs_dir,
        persist_dir=tmp_path / "chroma",
        output_dir=out / "artifacts",
    )

    with harness.isolate(settings):
        record = run_benchmark.run_case(
            case, simulator=simulator, loop_settings=base.loop.model_copy(update={"max_turns": 5})
        )

    assert record["status"] == "ok", record["error"]
    assert record["finished"] is True
    # It really went through the human-in-the-loop cycle.
    assert record["rounds"] == 1
    round_ = record["transcript"][0]
    assert round_["questions"][0]["text"] == "Comment est calculé le seuil d'alerte ?"
    assert round_["answers"][0]["matched_gap_ref"] == "GT-GAP-002"
    # ...and the answer became a user_answer context item.
    assert any(c["source"] == "user_answer" for c in record["context_items"])


def test_summary_row_reports_the_resolution_mix():
    record = {
        "case_id": "unit",
        "status": "ok",
        "finished": True,
        "turns": 3,
        "rounds": 2,
        "stop_reason": None,
        "wall_s": 1.5,
        "gaps": [
            {"severity": "blocking", "status": "user_answered", "category": "business_rule"},
            {"severity": "important", "status": "rag_answered", "category": "nfr"},
            {"severity": "nice_to_have", "status": "deferred", "category": "scope"},
        ],
        "context_items": [
            {"source": "initial_cdc"},
            {"source": "rag"},
            {"source": "user_answer"},
            {"source": "assumption"},
        ],
        "asked_questions": [{"id": "q1"}, {"id": "q2"}],
        "section_statuses": {"a": {"status": "complete"}, "b": {"status": "in_progress"}},
        "transcript": [{"answers": [{"skip": True}, {"skip": False}]}],
        "telemetry": {"llm_count": 12, "llm_s": 3.21, "retry_count": 1, "cache_hit_count": 4},
    }

    row = run_benchmark_summarize(record)

    assert row["gaps_total"] == 3
    assert row["gaps_blocking"] == 1
    assert (row["resolved_by_rag"], row["resolved_by_user"], row["assumed"]) == (1, 1, 1)
    assert row["unknown_answers"] == 1
    assert row["sections_complete"] == 1
    assert row["llm_calls"] == 12
    # Every CSV column must be present, or DictWriter would silently blank it.
    assert set(run_benchmark_columns()) <= set(row)


def run_benchmark_summarize(record):
    from evals.run_benchmark import summarize

    return summarize(record)


def run_benchmark_columns():
    from evals.run_benchmark import SUMMARY_COLUMNS

    return SUMMARY_COLUMNS


def test_main_writes_manifest_summary_and_report(tmp_path, monkeypatch):
    """End-to-end CLI shape, with run_case stubbed so no LLM is involved."""
    from evals import run_benchmark

    def fake_run_case(case, *, simulator, loop_settings):
        return {
            "case_id": case.case_id,
            "title": case.title,
            "thread_id": "t",
            "status": "ok",
            "error": None,
            "finished": True,
            "done": True,
            "stop_reason": None,
            "turns": 2,
            "rounds": 1,
            "wall_s": 0.1,
            "gaps": [{"severity": "blocking", "status": "user_answered", "category": "business_rule"}],
            "context_items": [{"source": "user_answer"}],
            "asked_questions": [{"id": "q"}],
            "section_statuses": {"functional": {"status": "complete"}},
            "transcript": [{"round": 0, "turn": 1, "questions": [], "answers": []}],
            "telemetry": {"llm_count": 1, "llm_s": 0.0},
        }

    monkeypatch.setattr(run_benchmark, "run_case", fake_run_case)

    exit_code = run_benchmark.main(
        ["--cases", "cdc_009_reservation_salles", "--run-id", "unit_run", "--out", str(tmp_path)]
    )
    assert exit_code == 0

    root = tmp_path / "unit_run"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_ids"] == ["cdc_009_reservation_salles"]
    assert manifest["simulator_mode"] == "oracle"

    with open(root / "summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "cdc_009_reservation_salles"

    assert "Benchmark run report" in (root / "report.md").read_text(encoding="utf-8")
    predictions = json.loads(
        (root / "cdc_009_reservation_salles" / "predictions.json").read_text(encoding="utf-8")
    )
    assert predictions["status"] == "ok"
    assert (root / "cdc_009_reservation_salles" / "transcript.jsonl").exists()


def test_benchmark_dir_is_where_the_loader_looks():
    assert BENCHMARK_DIR.exists()
    assert (BENCHMARK_DIR / "manifest.yaml").exists()
