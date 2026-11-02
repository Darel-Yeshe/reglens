"""The contract for chunk_pages(). These fail until Dave implements chunking."""

import pytest

from reglens.chunking import MAX_CHARS, chunk_pages, is_lampiran
from reglens.models import Page

pytestmark = pytest.mark.todo


def _pages() -> list[Page]:
    p1 = (
        "PERATURAN OTORITAS JASA KEUANGAN\n"
        "Menimbang: bahwa perlindungan konsumen perlu diatur.\n\n"
        "Pasal 1\n"
        "Dalam peraturan ini yang dimaksud dengan konsumen adalah setiap orang.\n\n"
        "Pasal 2\n"
        "(1) Pelaku usaha wajib menanggapi pengaduan.\n"
    )
    p2 = (
        "(2) Tanggapan diberikan paling lama 10 hari kerja.\n\n"
        "Pasal 3\n"
        "Peraturan ini mulai berlaku pada tanggal diundangkan.\n"
    )
    return [
        Page(doc_id="pojk-test", doc_title="POJK Test", page_number=1, text=p1),
        Page(doc_id="pojk-test", doc_title="POJK Test", page_number=2, text=p2),
    ]


def test_one_chunk_per_pasal_plus_preamble():
    labels = [c.pasal for c in chunk_pages(_pages())]
    assert labels == [None, "Pasal 1", "Pasal 2", "Pasal 3"]


def test_article_spanning_pages_stays_together_and_keeps_start_page():
    pasal2 = next(c for c in chunk_pages(_pages()) if c.pasal == "Pasal 2")
    assert "wajib menanggapi" in pasal2.text and "10 hari kerja" in pasal2.text
    assert pasal2.page_start == 1


def test_page_start_tracks_the_page_where_the_article_begins():
    pasal3 = next(c for c in chunk_pages(_pages()) if c.pasal == "Pasal 3")
    assert pasal3.page_start == 2


def test_chunk_ids_unique():
    ids = [c.chunk_id for c in chunk_pages(_pages())]
    assert len(ids) == len(set(ids))


def test_long_article_split_under_limit_and_keeps_label():
    long_text = "Pasal 9\n" + "\n\n".join(f"({i}) " + "kata " * 120 for i in range(1, 12))
    pages = [Page(doc_id="d", doc_title="D", page_number=1, text=long_text)]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert all(c.pasal == "Pasal 9" for c in chunks)


def test_documents_are_chunked_separately():
    a = Page(doc_id="a", doc_title="A", page_number=1, text="Pasal 1\nIsi A.")
    b = Page(doc_id="b", doc_title="B", page_number=1, text="Pasal 1\nIsi B.")
    chunks = chunk_pages([a, b])
    assert {c.doc_id for c in chunks} == {"a", "b"}
    assert all(("Isi A" in c.text) == (c.doc_id == "a") for c in chunks)


def test_lampiran_sections_are_prefixed():
    text = (
        "I. KETENTUAN UMUM\nIsi umum.\n\n"
        "II. KETENTUAN PENUTUP\nBerlaku sejak ditetapkan.\n\n"
        "LAMPIRAN I\nFormat laporan.\n\n"
        "I. FORMAT LAPORAN\nTabel.\n"
    )
    pages = [Page(doc_id="se", doc_title="SE", page_number=1, text=text)]
    labels = [c.pasal for c in chunk_pages(pages)]
    assert labels == ["I. KETENTUAN UMUM", "II. KETENTUAN PENUTUP", "LAMPIRAN I", "LAMPIRAN I – I. FORMAT LAPORAN"]
    assert is_lampiran(labels[-1]) and not is_lampiran(labels[0])
