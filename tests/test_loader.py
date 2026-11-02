from pathlib import Path

import pytest

from reglens.config import get_settings
from reglens.loader import _repeated_lines, clean_text, doc_id_from_path, load_pdf


def test_doc_id_is_slug():
    assert doc_id_from_path(Path("POJK 22 Tahun 2023.pdf")) == "pojk-22-tahun-2023"


def test_clean_text_rejoins_hyphenated_words_and_collapses_space():
    raw = "perlin-\ndungan   konsumen\n\n\n\nPasal 1"
    assert clean_text(raw) == "perlindungan konsumen\n\nPasal 1"


def test_repeated_lines_detected_as_headers():
    pages = [["SALINAN", "Pasal 1", "isi"], ["SALINAN", "Pasal 2"], ["SALINAN", "Pasal 3"]]
    assert _repeated_lines(pages) == {"SALINAN"}


def test_no_header_detection_on_short_docs():
    assert _repeated_lines([["SALINAN"], ["SALINAN"]]) == set()


def test_real_pdf_loads_if_present():
    """Smoke test on the smallest PDF only, so the suite stays fast.
    Run `python scripts/ingest.py` to check every PDF."""
    pdfs = sorted(get_settings().raw_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if not pdfs:
        pytest.skip("No PDFs in data/raw yet")
    pages = load_pdf(pdfs[0])
    assert pages, f"{pdfs[0].name}: no text extracted (scanned PDF? needs OCR)"
    assert all(p.text.strip() for p in pages)
