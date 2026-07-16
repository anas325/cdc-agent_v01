"""Slice a section/chapter out of the full raw CDC text, alias-aware."""


from __future__ import annotations

import re

from src.state import SectionConfig

_MAX_HEADING_LEN = 100

_HEADING_PREFIX_RE = re.compile(
    r"""^\s*
        (?:\#{1,6}\s*)?                 # markdown heading marker
        (?:[IVXLCDM]+[.\)]\s*)?         # roman numeral numbering, e.g. "II."
        (?:\d+(?:\.\d+)*[.\)]?\s*)?     # arabic numbering, e.g. "1." / "1.2" / "1)"
        (?P<name>.+?)
        \s*:?\s*$""",
    re.VERBOSE,
)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _heading_index(sections_config: list[SectionConfig]) -> dict[str, str]:
    """Map normalized heading text (title or alias) -> section id."""
    index: dict[str, str] = {}
    for section in sections_config:
        for name in (section.title, *section.aliases):
            normalized = _normalize(name)
            if normalized:
                index[normalized] = section.id
    return index


def extract_section(
    cdc_text: str,
    section_id: str,
    sections_config: list[SectionConfig],
) -> str | None:
    """Return the slice of `cdc_text` belonging to `section_id`, or None if not found.

    A line is recognized as the heading for a section if it matches (modulo
    markdown/numbering prefixes, case, punctuation) that section's `title` or
    one of its `aliases` in `sections_config` — e.g. "1. Contexte" or "## Contexte"
    both resolve to a section whose alias list contains "Contexte". The extracted
    text runs from that heading (exclusive) up to the next recognized heading of
    a *different* section, or the end of the document.

    Raises KeyError if `section_id` is not present in `sections_config`.
    """
    if not any(s.id == section_id for s in sections_config):
        raise KeyError(section_id)

    index = _heading_index(sections_config)
    lines = cdc_text.splitlines()

    start: int | None = None
    end: int | None = None
    for i, line in enumerate(lines):
        if len(line) > _MAX_HEADING_LEN:
            continue
        match = _HEADING_PREFIX_RE.match(line)
        if not match:
            continue
        matched_id = index.get(_normalize(match.group("name")))
        if matched_id is None:
            continue
        if start is None and matched_id == section_id:
            start = i + 1
        elif start is not None and matched_id != section_id:
            end = i
            break

    if start is None:
        return None
    return "\n".join(lines[start:end]).strip()
