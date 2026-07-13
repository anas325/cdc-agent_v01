from __future__ import annotations

import pytest

from src.app import score_section


def test_score_section_applies_weights_and_clamps_at_zero() -> None:
    gaps = [
        {"category": "contradiction", "severity": "blocking"},
        {"category": "scope", "severity": "important"},
        {"category": "edge_case", "severity": "nice_to_have"},
    ]

    assert score_section(gaps) == pytest.approx(71.7)


def test_score_section_returns_full_score_when_no_gaps() -> None:
    assert score_section([]) == 100.0
