"""Load regulation PDFs into clean, page-numbered text.

Built by Claude. Read it so you can explain it in an interview, but you
don't need to change it.

Uses PyMuPDF: on these OJK PDFs pypdf split many words ("d an",
"tiga p uluh"), which hurt both keyword and vector search. PyMuPDF gives each
word with its position; words at the same height are joined back into lines.
Some pages (e.g. POJK 16) hide text where PyMuPDF can't see it; when pypdf
finds clearly more text on a page, that page uses pypdf instead.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from reglens.models import Page

# A line that shows up on more than this share of pages is treated as a
# running header/footer (e.g. "SALINAN", "- 3 -", the regulation title).
REPEATED_LINE_SHARE = 0.5
MIN_PAGES_FOR_HEADER_DETECTION = 3

FALLBACK_SHARE = 0.8  # use pypdf for a page if PyMuPDF gets < 80% of pypdf's characters
SAME_LINE_TOLERANCE = 3  # points: words whose vertical centres differ less are on one line

PAGE_NUMBER_LINE = re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$")


def doc_id_from_path(path: Path) -> str:
    """'POJK 22 Tahun 2023.pdf' -> 'pojk-22-tahun-2023'."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or "document"


def title_from_path(path: Path) -> str:
    return re.sub(r"[_\s]+", " ", path.stem).strip()


def clean_text(text: str) -> str:
    """Normalise whitespace and rejoin words split across lines with a hyphen."""
    text = text.replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "perlin-\ndungan" -> "perlindungan"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repeated_lines(pages_lines: list[list[str]]) -> set[str]:
    if len(pages_lines) < MIN_PAGES_FOR_HEADER_DETECTION:
        return set()
    counts = Counter(line for lines in pages_lines for line in set(lines) if line)
    limit = REPEATED_LINE_SHARE * len(pages_lines)
    return {line for line, n in counts.items() if n > limit}


def page_text(page) -> str:
    """Rebuild a page's lines from word positions (top to bottom, left to right).

    page.get_text() alone puts each word of justified text on its own line,
    so headings like "III. PENYAMPAIAN LAPORAN" came out in pieces.
    """
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list] = []
    line_y = None
    for word in words:
        y = (word[1] + word[3]) / 2  # vertical centre; word = (x0, y0, x1, y1, text, ...)
        if line_y is None or abs(y - line_y) > SAME_LINE_TOLERANCE:
            lines.append([])
            line_y = y
        lines[-1].append(word)
    return "\n".join(" ".join(w[4] for w in sorted(line, key=lambda w: w[0])) for line in lines)


def load_pdf(path: Path) -> list[Page]:
    """Extract every page of one PDF, dropping headers, footers and page numbers."""
    with pymupdf.open(path) as pdf:
        raw_pages = [page_text(page) for page in pdf]
    for i, page in enumerate(PdfReader(str(path)).pages):
        fallback = page.extract_text() or ""
        if len(raw_pages[i]) < FALLBACK_SHARE * len(fallback):
            raw_pages[i] = fallback
    pages_lines = [[ln.strip() for ln in clean_text(t).split("\n")] for t in raw_pages]
    repeated = _repeated_lines(pages_lines)

    doc_id, title = doc_id_from_path(path), title_from_path(path)
    pages: list[Page] = []
    for number, lines in enumerate(pages_lines, start=1):
        kept = [ln for ln in lines if ln not in repeated and not PAGE_NUMBER_LINE.match(ln)]
        text = "\n".join(kept).strip()
        if text:
            pages.append(Page(doc_id=doc_id, doc_title=title, page_number=number, text=text))
    return pages


def load_folder(folder: Path) -> list[Page]:
    """Load every PDF in a folder, in file-name order."""
    pages: list[Page] = []
    for path in sorted(folder.glob("*.pdf")):
        pages.extend(load_pdf(path))
    return pages
