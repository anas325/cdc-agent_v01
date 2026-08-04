"""Schema and loader for the annotated CDC benchmark (roadmap Phase 2).

Layout on disk:

    evals/datasets/benchmark/
        manifest.yaml                 # dataset_version + ordered case list
        cdc_001_smartstock/
            initial_cdc.md            # the vague CDC fed to the graph
            source_docs/              # this case's RAG corpus (may be empty)
            ground_truth.json         # expert annotations
            stakeholder.yaml          # synthetic stakeholder profile (Phase 3)

Ground truth is annotation, not prediction: it says what a careful human thinks
the system *should* find, plus what a fully-informed stakeholder *would* answer.
Nothing here is compared to a run — that is Phase 4's job. The models below just
make sure a typo (a category that isn't a GapCategory, a gap_ref pointing
nowhere, an expected_evidence document that isn't on disk) fails at load time
rather than halfway through a two-hour batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.state import GapCategory, GapSeverity

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent / "datasets" / "benchmark"

# How a gap is expected to be closed. Drives Phase 4's split of "retrieval
# problem" vs "reasoning problem" (roadmap §10) and tells the oracle simulator
# which gaps it is allowed to answer at all.
ResolvableBy = Literal["rag", "human", "neither"]


class EvidenceRef(BaseModel):
    """Where the answer to a RAG-resolvable gap actually lives."""

    document: str
    page: int | None = None
    quote: str | None = None


class GroundTruthGap(BaseModel):
    id: str  # GT-GAP-001, unique within the case
    section_id: str  # must exist in config/sections.yaml
    category: GapCategory
    severity: GapSeverity
    description: str
    # Distinctive terms from the gap, used to match a runtime gap to this
    # annotation (runtime gap ids are content-hashed, so matching is by content).
    keywords: list[str] = Field(default_factory=list)
    resolvable_by: ResolvableBy = "human"
    expected_evidence: EvidenceRef | None = None


class GroundTruthContradiction(BaseModel):
    id: str  # GT-CON-001
    statement_a: str
    statement_b: str
    section_ids: list[str] = Field(default_factory=list)
    severity: GapSeverity = "blocking"
    keywords: list[str] = Field(default_factory=list)


class ExpectedAnswer(BaseModel):
    """What a perfectly-informed stakeholder replies when asked about a gap."""

    gap_ref: str  # -> GroundTruthGap.id
    answer: str
    keywords: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    case_id: str
    title: str
    dataset_version: str
    annotator: str
    notes: str = ""
    gaps: list[GroundTruthGap] = Field(default_factory=list)
    contradictions: list[GroundTruthContradiction] = Field(default_factory=list)
    expected_answers: list[ExpectedAnswer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_refs(self) -> GroundTruth:
        gap_ids = [g.id for g in self.gaps]
        if len(set(gap_ids)) != len(gap_ids):
            raise ValueError(f"{self.case_id}: duplicate gap ids in ground_truth.json")

        con_ids = [c.id for c in self.contradictions]
        if len(set(con_ids)) != len(con_ids):
            raise ValueError(f"{self.case_id}: duplicate contradiction ids")

        known = set(gap_ids)
        for ans in self.expected_answers:
            if ans.gap_ref not in known:
                raise ValueError(
                    f"{self.case_id}: expected_answer.gap_ref {ans.gap_ref!r} matches no gap"
                )
        return self

    def gap_by_id(self, gap_id: str) -> GroundTruthGap | None:
        return next((g for g in self.gaps if g.id == gap_id), None)


class Behavior(BaseModel):
    """How the synthetic stakeholder answers, beyond *what* it knows."""

    style: Literal["precise", "vague"] = "precise"
    # Extra chance of "je ne sais pas" even for known facts — models a
    # distracted or hedging stakeholder. Applied with a seeded RNG.
    unknown_rate: float = 0.0
    allow_contradictions: bool = False


class StakeholderProfile(BaseModel):
    name: str
    role: str
    knowledge: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    behavior: Behavior = Behavior()
    # Statements that contradict `knowledge`, injected only in realistic mode
    # with allow_contradictions — this is what exercises the critic.
    contradictions: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    case_id: str
    path: Path
    title: str
    initial_cdc_text: str
    source_doc_names: list[str]
    ground_truth: GroundTruth
    stakeholder: StakeholderProfile

    @property
    def source_docs_dir(self) -> Path:
        return self.path / "source_docs"


class DatasetError(ValueError):
    pass


def _load_case(case_dir: Path, valid_section_ids: set[str]) -> BenchmarkCase:
    case_id = case_dir.name
    cdc_path = case_dir / "initial_cdc.md"
    gt_path = case_dir / "ground_truth.json"
    profile_path = case_dir / "stakeholder.yaml"

    for required in (cdc_path, gt_path, profile_path):
        if not required.exists():
            raise DatasetError(f"{case_id}: missing {required.name}")

    ground_truth = GroundTruth.model_validate(json.loads(gt_path.read_text(encoding="utf-8")))
    if ground_truth.case_id != case_id:
        raise DatasetError(
            f"{case_id}: ground_truth.case_id is {ground_truth.case_id!r}, expected the directory name"
        )

    profile = StakeholderProfile.model_validate(
        yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    )

    source_dir = case_dir / "source_docs"
    source_names = sorted(p.name for p in source_dir.glob("*") if p.is_file()) if source_dir.exists() else []

    # Cross-checks that only make sense with the section config and the files
    # on disk in hand, so they live here rather than on the models.
    for gap in ground_truth.gaps:
        if gap.section_id not in valid_section_ids:
            raise DatasetError(
                f"{case_id}/{gap.id}: section_id {gap.section_id!r} is not in config/sections.yaml "
                f"({sorted(valid_section_ids)})"
            )
        if gap.resolvable_by == "rag" and gap.expected_evidence is None:
            raise DatasetError(f"{case_id}/{gap.id}: resolvable_by='rag' requires expected_evidence")
        if gap.expected_evidence and gap.expected_evidence.document not in source_names:
            raise DatasetError(
                f"{case_id}/{gap.id}: expected_evidence document "
                f"{gap.expected_evidence.document!r} is not in source_docs/ ({source_names})"
            )

    for con in ground_truth.contradictions:
        unknown = [sid for sid in con.section_ids if sid not in valid_section_ids]
        if unknown:
            raise DatasetError(f"{case_id}/{con.id}: unknown section_ids {unknown}")

    # A gap the documents can answer must not also carry a stakeholder answer:
    # the oracle would then close it even when retrieval failed, hiding the
    # failure behind a human answer that never should have been needed.
    rag_ids = {g.id for g in ground_truth.gaps if g.resolvable_by == "rag"}
    both = sorted(a.gap_ref for a in ground_truth.expected_answers if a.gap_ref in rag_ids)
    if both:
        raise DatasetError(
            f"{case_id}: {both} are resolvable_by='rag' but also have an expected_answer"
        )

    return BenchmarkCase(
        case_id=case_id,
        path=case_dir,
        title=ground_truth.title,
        initial_cdc_text=cdc_path.read_text(encoding="utf-8"),
        source_doc_names=source_names,
        ground_truth=ground_truth,
        stakeholder=profile,
    )


def load_manifest(root: Path | None = None) -> dict:
    root = root or BENCHMARK_DIR
    path = root / "manifest.yaml"
    if not path.exists():
        raise DatasetError(f"No benchmark manifest at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_benchmark(
    root: Path | None = None, case_ids: list[str] | None = None
) -> list[BenchmarkCase]:
    """Load and validate benchmark cases, in manifest order.

    `case_ids` selects a subset (order still follows the manifest), which is how
    the runner supports iterating on one case without paying for all ten.
    """
    from src.config import load_sections

    root = root or BENCHMARK_DIR
    manifest = load_manifest(root)
    valid_section_ids = {s.id for s in load_sections()}

    listed = [entry["id"] if isinstance(entry, dict) else str(entry) for entry in manifest["cases"]]
    if case_ids:
        unknown = [cid for cid in case_ids if cid not in listed]
        if unknown:
            raise DatasetError(f"Unknown case id(s): {unknown}. Available: {listed}")
        listed = [cid for cid in listed if cid in case_ids]

    return [_load_case(root / cid, valid_section_ids) for cid in listed]


def dataset_version(root: Path | None = None) -> str:
    return str(load_manifest(root).get("dataset_version", "unknown"))
