"""Tests for the dev-loop LLM disk cache and the deterministic ids it needs.

The cache only pays off if prompts are byte-identical across runs, and prompts
embed context/gap ids — so id stability is tested here alongside the cache
itself.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.ids import stable_id
from src.llm import call_structured


class Answer(BaseModel):
    value: str


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class CountingLLM:
    """Stands in for a chat model; counts how often it is actually invoked."""

    def __init__(self, payload: str = '{"value": "hello"}') -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt: str) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.payload)


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CDC_LLM_CACHE_DIR", str(tmp_path / "llm"))
    return tmp_path / "llm"


def test_second_identical_call_is_served_from_disk(monkeypatch, cache_dir):
    monkeypatch.setenv("CDC_LLM_CACHE", "1")
    llm = CountingLLM()

    first = call_structured("prompt A", Answer, llm=llm)
    second = call_structured("prompt A", Answer, llm=llm)

    assert llm.calls == 1
    assert first.value == second.value == "hello"


def test_cache_is_off_by_default(monkeypatch, cache_dir):
    monkeypatch.delenv("CDC_LLM_CACHE", raising=False)
    llm = CountingLLM()

    call_structured("prompt A", Answer, llm=llm)
    call_structured("prompt A", Answer, llm=llm)

    assert llm.calls == 2


def test_different_prompts_do_not_share_an_entry(monkeypatch, cache_dir):
    monkeypatch.setenv("CDC_LLM_CACHE", "1")
    llm = CountingLLM()

    call_structured("prompt A", Answer, llm=llm)
    call_structured("prompt B", Answer, llm=llm)

    assert llm.calls == 2


def test_corrupt_entry_falls_through_to_a_live_call(monkeypatch, cache_dir):
    monkeypatch.setenv("CDC_LLM_CACHE", "1")
    llm = CountingLLM()
    call_structured("prompt A", Answer, llm=llm)

    for path in cache_dir.rglob("*.json"):
        path.write_text("not json at all", encoding="utf-8")

    result = call_structured("prompt A", Answer, llm=llm)

    assert llm.calls == 2
    assert result.value == "hello"


def test_cache_hit_is_reported_in_telemetry(monkeypatch, cache_dir):
    monkeypatch.setenv("CDC_LLM_CACHE", "1")
    from src import telemetry

    telemetry.reset()
    llm = CountingLLM()

    call_structured("prompt A", Answer, llm=llm)
    call_structured("prompt A", Answer, llm=llm)

    calls = telemetry.llm_calls()
    assert [c.cache_hit for c in calls] == [False, True]
    assert telemetry.summary()["cache_hit_count"] == 1


def test_stable_id_is_deterministic_and_input_sensitive():
    assert stable_id("ctx", "a", "b") == stable_id("ctx", "a", "b")
    assert stable_id("ctx", "a", "b") != stable_id("ctx", "a", "c")
    # Parts are separated, so ("ab","c") must not collide with ("a","bc").
    assert stable_id("ctx", "ab", "c") != stable_id("ctx", "a", "bc")
    assert stable_id("ctx", "a").startswith("ctx_")
