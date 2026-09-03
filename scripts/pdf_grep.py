"""Search a PDF for a regex and report every match with its page number.

Usage:
    uv run python scripts/pdf_grep.py <file.pdf> <pattern> [-i] [-C N]

Options:
    -i, --ignore-case   Case-insensitive matching.
    -C, --context N      Print N characters of surrounding text around each match
                         (default: 40). Use 0 for match text only.
    --whole-page         Print the full page text for any page with a match.

Exit code is 0 if at least one match was found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader


def iter_matches(pdf_path: Path, pattern: re.Pattern[str], context: int):
    """Yield (page_number, match, page_text) for every match in the PDF.

    Page numbers are 1-based.
    """
    reader = PdfReader(str(pdf_path))
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for match in pattern.finditer(text):
            yield page_index, match, text


def format_snippet(match: re.Match[str], text: str, context: int) -> str:
    if context <= 0:
        return match.group(0).strip()
    start = max(0, match.start() - context)
    end = min(len(text), match.end() + context)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    snippet = text[start:end].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return f"{prefix}{snippet}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search a PDF for a regex and report matches with page numbers."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("pattern", help="Regular expression to search for.")
    parser.add_argument(
        "-i", "--ignore-case", action="store_true", help="Case-insensitive matching."
    )
    parser.add_argument(
        "-C",
        "--context",
        type=int,
        default=40,
        help="Characters of surrounding context to show (default: 40, 0 for match only).",
    )
    parser.add_argument(
        "--whole-page",
        action="store_true",
        help="Print the full text of every page that contains a match.",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"file not found: {args.pdf}")

    flags = re.IGNORECASE if args.ignore_case else 0
    try:
        pattern = re.compile(args.pattern, flags)
    except re.error as exc:
        parser.error(f"invalid regex: {exc}")

    total = 0
    pages_with_matches: dict[int, str] = {}
    counts_by_page: dict[int, int] = {}

    for page_number, match, text in iter_matches(args.pdf, pattern, args.context):
        total += 1
        counts_by_page[page_number] = counts_by_page.get(page_number, 0) + 1
        pages_with_matches.setdefault(page_number, text)
        snippet = format_snippet(match, text, args.context)
        print(f"{snippet}         {page_number}")
        

    if args.whole_page and pages_with_matches:
        print("\n" + "=" * 60)
        for page_number in sorted(pages_with_matches):
            print(f"\n--- full text of page {page_number} ---")
            print(pages_with_matches[page_number])

    print("\n" + "-" * 60)
    if total == 0:
        print("No matches found.")
        return 1

    summary = ", ".join(
        f"p{page}: {counts_by_page[page]}" for page in sorted(counts_by_page)
    )
    print(f"{total} match(es) across {len(counts_by_page)} page(s) -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
