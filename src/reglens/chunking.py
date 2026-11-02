"""Split regulation pages into citable chunks.

Written by Dave with Claude. tests/test_chunking.py describes what "done" looks like.

Goal: one chunk per article ("Pasal"), because that's how people cite
regulations. Circulars (SEOJK) have no Pasal; they use Roman-numeral sections
("I. KETENTUAN UMUM"), so those become the labels instead. POJKs end with an
official explanation (PENJELASAN) that repeats "Pasal 1, Pasal 2...", so those
chunks are labelled "Penjelasan Pasal 1" to keep citations unambiguous.
Attachments (LAMPIRAN) often quote other regulations, so everything after a
LAMPIRAN heading is labelled "LAMPIRAN I – <section>" (retrieval can then rank
the original text above the quote).
Long sections are split further (same label) so no chunk exceeds MAX_CHARS.
"""

import re
from bisect import bisect_right
from collections import defaultdict

from reglens.models import Chunk, Page

MAX_CHARS = 2000
OVERLAP_CHARS = 200

# "Pasal 12A" alone on a line
PASAL_HEADING = re.compile(r"^Pasal\s+\d+[A-Z]?\s*$", re.MULTILINE)
# "II. PENILAIAN KEMAMPUAN DAN KEPATUTAN": Roman numeral, dot, UPPERCASE title,
# but not table-of-contents lines like "I. FORMAT ........ 3"
SECTION_HEADING = re.compile(
    r"^(?![^\n]*\.{3})[IVXL]+\.\s+[A-Z][A-Z0-9 ,/&()'\-–.]*$", re.MULTILINE
)
# The start of a POJK's official explanation
PENJELASAN_HEADING = re.compile(r"^PENJELASAN\s*$", re.MULTILINE)
# "LAMPIRAN", "LAMPIRAN II", "LAMPIRAN DOKUMEN ..." in capitals at a line start
# (not ToC lines with "....", not "Lampiran" mentioned inside a sentence)
LAMPIRAN_HEADING = re.compile(r"^LAMPIRAN\b(?![^\n]*\.{3})[^\n]*$", re.MULTILINE)
LAMPIRAN_PREFIX = "LAMPIRAN"

# How to break long text, best first: paragraphs, then lines, then sentences
SEPARATORS = ["\n\n", "\n", ". "]


def _find_headings(text: str) -> list[tuple[int, str]]:
    """Return (position, label) for every heading in one document, in order."""
    headings = _main_headings(text)
    lampiran = [(m.start(), m.group().strip()[:60]) for m in LAMPIRAN_HEADING.finditer(text)]
    if not lampiran:
        return headings

    # Headings inside an attachment get its name in front: "LAMPIRAN I – II. FORMAT"
    starts = [pos for pos, _ in lampiran]
    labelled = []
    for pos, label in headings:
        i = bisect_right(starts, pos) - 1
        labelled.append((pos, label if i < 0 else f"{lampiran[i][1]} – {label}"))
    return sorted(labelled + lampiran)


def is_lampiran(label: str | None) -> bool:
    """True for chunks from an attachment (see LAMPIRAN_HEADING)."""
    return bool(label) and label.startswith(LAMPIRAN_PREFIX)


def _main_headings(text: str) -> list[tuple[int, str]]:
    """Pasal or Roman-section headings, ignoring attachments."""
    pasal = list(PASAL_HEADING.finditer(text))
    sections = list(SECTION_HEADING.finditer(text))

    # Circular (SEOJK): its sections start before any "Pasal" line appears
    # (Pasal lines in a circular are only quotes from the regulation it implements).
    if sections and (not pasal or sections[0].start() < pasal[0].start()):
        return [(m.start(), m.group().strip()) for m in sections]

    # Regulation (POJK): articles, relabelled once the PENJELASAN part begins
    penjelasan = PENJELASAN_HEADING.search(text)
    penjelasan_pos = penjelasan.start() if penjelasan else len(text)
    headings = []
    for m in pasal:
        label = m.group().strip()
        if m.start() > penjelasan_pos:
            label = f"Penjelasan {label}"
        headings.append((m.start(), label))
    if penjelasan:
        headings.append((penjelasan_pos, "Penjelasan Umum"))
    return sorted(headings)


def _hard_cut(text: str) -> list[str]:
    """Last resort: cut at the last space before MAX_CHARS, never mid-word."""
    pieces = []
    while len(text) > MAX_CHARS:
        cut = text.rfind(" ", 0, MAX_CHARS)
        if cut <= 0:
            cut = MAX_CHARS
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    pieces.append(text)
    return pieces


def _overlap_tail(text: str) -> str:
    """The last ~OVERLAP_CHARS of text, starting at a word boundary."""
    tail = text[-OVERLAP_CHARS:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def _split_long(text: str, level: int = 0) -> list[str]:
    """Split text over MAX_CHARS into overlapping pieces, at the best boundary available."""
    if len(text) <= MAX_CHARS:
        return [text]
    if level >= len(SEPARATORS):
        return _hard_cut(text)

    sep = SEPARATORS[level]
    parts = text.split(sep)

    # Any part that is itself too long gets split at the next, finer separator
    small_parts = []
    for part in parts:
        small_parts.extend(_split_long(part, level + 1))

    # Pack parts into pieces of at most MAX_CHARS, with overlap between pieces
    pieces = []
    current = ""
    for part in small_parts:
        if current and len(current) + len(sep) + len(part) > MAX_CHARS:
            pieces.append(current)
            tail = _overlap_tail(current)
            current = tail + sep + part if len(tail) + len(sep) + len(part) <= MAX_CHARS else part
        elif current:
            current += sep + part
        else:
            current = part
    pieces.append(current)
    return pieces


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """Turn pages (possibly from several documents) into citable chunks."""
    chunks = []

    # Step 1: group pages by document
    groups = defaultdict(list)
    for page in pages:
        groups[page.doc_id].append(page)

    for doc_id, doc_pages in groups.items():
        # Step 2: join pages, bookmark where each page starts
        text = ""
        offsets = []
        page_numbers = []
        for page in doc_pages:
            offsets.append(len(text))
            page_numbers.append(page.page_number)
            text += page.text + "\n\n"

        # Step 3: find the headings (Pasal, or Roman sections for circulars)
        headings = _find_headings(text)

        # Step 4: slice into (label, start, body)
        articles = []
        first = headings[0][0] if headings else len(text)
        preamble = text[:first].strip()
        if preamble:
            articles.append((None, 0, preamble))
        for i, (start, label) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if body:
                articles.append((label, start, body))

        # Step 6: turn each article into Chunk objects
        doc_title = doc_pages[0].doc_title
        seen_slugs = defaultdict(int)
        for label, start, body in articles:
            page_start = page_numbers[bisect_right(offsets, start) - 1]
            slug = re.sub(r"\W+", "-", (label or "preamble").lower()).strip("-")
            seen_slugs[slug] += 1
            if seen_slugs[slug] > 1:  # same heading twice (e.g. sections restart in an attachment)
                slug = f"{slug}-{seen_slugs[slug]}"
            for piece_index, piece in enumerate(_split_long(body)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::{slug}::{piece_index}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        pasal=label,
                        page_start=page_start,
                        text=piece,
                    )
                )

    return chunks
