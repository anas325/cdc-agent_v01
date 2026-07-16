"""Slice a section/chapter out of the full raw CDC text, alias-aware."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from src.state import SectionConfig

_MAX_HEADING_LEN = 100

_HEADING_PREFIX_RE = re.compile(
    r"""^\s*
        (?P<hashes>\#{1,6})?\s*                # markdown heading marker
        (?:(?P<roman>[IVXLCDM]+)[.\)]\s*)?      # roman numeral numbering, e.g. "II."
        (?:(?P<arabic>\d+(?:\.\d+)*)[.\)]?\s*)? # arabic numbering, e.g. "1." / "1.2" / "1)"
        (?P<name>.+?)
        \s*:?\s*$""",
    re.VERBOSE,
)


def _heading_level(match: re.Match[str]) -> int | None:
    """Structural depth of a heading line, or None if it has no explicit marker."""
    if match.group("hashes"):
        return len(match.group("hashes"))
    if match.group("roman"):
        return 1
    if match.group("arabic"):
        return match.group("arabic").count(".") + 1
    return None


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


class _Segment(NamedTuple):
    section_id: str | None  # None = preamble / text under an unrecognized heading
    lines: list[str]  # content lines (a recognized heading line is excluded)


def _segment_cdc(cdc_text: str, sections_config: list[SectionConfig]) -> list[_Segment]:
    """Partition `cdc_text` into contiguous segments, one per recognized section run.

    A line is recognized as the heading for a section if it matches (modulo
    markdown/numbering prefixes, case, punctuation) that section's `title` or
    one of its `aliases` in `sections_config` — e.g. "1. Contexte" or "## Contexte"
    both resolve to a section whose alias list contains "Contexte". A section
    segment runs from its heading (exclusive) up to whichever comes first:
    - the next heading recognized as belonging to a *different* section, or
    - the next heading with the same or shallower structural depth (markdown
      `#` count / numbering depth) as the starting heading, even if it isn't
      recognized — this keeps a partial `sections_config` or an unlisted
      heading from swallowing the rest of the document; the unrecognized
      heading and what follows open a `section_id=None` segment,
    - or the end of the document if neither occurs.

    Text before the first recognized heading forms a leading `section_id=None`
    segment. Segments may be empty; callers filter as needed.
    """
    index = _heading_index(sections_config)
    segments: list[_Segment] = [_Segment(None, [])]
    current_level: int | None = None

    for line in cdc_text.splitlines():
        match = _HEADING_PREFIX_RE.match(line) if len(line) <= _MAX_HEADING_LEN else None
        matched_id = index.get(_normalize(match.group("name"))) if match else None
        level = _heading_level(match) if match else None
        current = segments[-1]

        if matched_id is not None and matched_id != current.section_id:
            segments.append(_Segment(matched_id, []))
            current_level = level
            continue
        if (
            matched_id is None
            and current.section_id is not None
            and level is not None
            and current_level is not None
            and level <= current_level
        ):
            segments.append(_Segment(None, [line]))
            continue
        current.lines.append(line)

    return segments


def extract_section(
    cdc_text: str,
    section_id: str,
    sections_config: list[SectionConfig],
) -> str | None:
    """Return the slice of `cdc_text` belonging to `section_id`, or None if not found.

    Returns the first segment recognized for `section_id`; see `_segment_cdc`
    for the heading-recognition and boundary rules.

    Raises KeyError if `section_id` is not present in `sections_config`.
    """
    if not any(s.id == section_id for s in sections_config):
        raise KeyError(section_id)

    for segment in _segment_cdc(cdc_text, sections_config):
        if segment.section_id == section_id:
            return "\n".join(segment.lines).strip()
    return None


@dataclass
class CDCSplit:
    """Result of splitting a raw CDC into per-section texts.

    `sections` maps section id -> stripped text (non-empty entries only;
    repeated headings for the same section are merged). `unmatched` holds
    preamble and unrecognized-heading chunks in document order — nothing
    from the source text is dropped.
    """

    sections: dict[str, str] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    @property
    def matched_any(self) -> bool:
        return bool(self.sections)


def split_cdc_by_sections(cdc_text: str, sections_config: list[SectionConfig]) -> CDCSplit:
    split = CDCSplit()
    for segment in _segment_cdc(cdc_text, sections_config):
        text = "\n".join(segment.lines).strip()
        if not text:
            continue
        if segment.section_id is None:
            split.unmatched.append(text)
        elif segment.section_id in split.sections:
            split.sections[segment.section_id] += "\n\n" + text
        else:
            split.sections[segment.section_id] = text
    return split


def main() -> None:
    from src.config import load_sections

    data_path = Path(__file__).resolve().parents[2] / "data" / "smartStock.md"
    cdc_text = data_path.read_text(encoding="utf-8")
    sections_config = load_sections()

    for section_id in ("problem", "functional", "technical", "constraints"):
        extracted = extract_section(cdc_text, section_id, sections_config)
        print(f"=== {section_id} ===")
        print(extracted if extracted is not None else f"Section '{section_id}' not found.")
        print()


if __name__ == "__main__":
    main()

