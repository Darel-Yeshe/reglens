"""Tests for the retrieval tweaks, with a fake ChromaDB and fake embeddings (fast, offline)."""

import pytest

from reglens import retrieval
from reglens.config import get_settings
from reglens.models import Chunk


class FakeCollection:
    """Returns fixed results per query vector, like collection.query would."""

    def __init__(self, results_by_query: dict[str, list[tuple[str, str, float]]]):
        self.results_by_query = results_by_query  # query -> [(chunk_id, pasal, distance)]
        self.texts: dict[str, str] = {}  # chunk_id -> text, for keyword search

    def count(self) -> int:
        return 100

    def get(self, include):
        """Every stored chunk (used to build the BM25 index)."""
        rows = {cid: pasal for hits in self.results_by_query.values() for cid, pasal, _ in hits}
        return {
            "ids": list(rows),
            "documents": [self.texts.get(cid, "isi") for cid in rows],
            "metadatas": [{"doc_id": cid.split("::")[0], "doc_title": "T", "pasal": pasal, "page_start": 1}
                          for cid, pasal in rows.items()],
        }

    def query(self, query_embeddings, n_results):
        rows = self.results_by_query[query_embeddings[0]][:n_results]
        return {
            "ids": [[cid for cid, _, _ in rows]],
            "documents": [["isi"] * len(rows)],
            "metadatas": [[{"doc_id": cid.split("::")[0], "doc_title": "T", "pasal": pasal, "page_start": 1}
                           for cid, pasal, _ in rows]],
            "distances": [[d for _, _, d in rows]],
        }


@pytest.fixture
def setup(monkeypatch):
    """Fake embed (the 'vector' is just the query text) and reset settings."""
    monkeypatch.setattr(retrieval, "_embed", lambda texts, show_progress=False: list(texts))
    retrieval.translate_to_indonesian.cache_clear()
    retrieval._bm25_index.cache_clear()
    get_settings.cache_clear()
    yield monkeypatch
    retrieval._bm25_index.cache_clear()
    get_settings.cache_clear()


def _use(monkeypatch, collection, **env):
    monkeypatch.setattr(retrieval, "_collection", lambda: collection)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def test_looks_english():
    assert retrieval.looks_english("Within how many days must a complaint be answered?")
    assert not retrieval.looks_english("Berapa lama PUJK wajib menanggapi pengaduan?")


def test_lampiran_penalty_lets_the_original_win(setup):
    hits = [("seojk::lampiran::0", "LAMPIRAN", 0.10), ("pojk::pasal-75::0", "Pasal 75", 0.12)]
    collection = FakeCollection({"query: q": hits})

    _use(setup, collection, LAMPIRAN_PENALTY="0")
    assert [c.pasal for c in retrieval.retrieve("q", top_k=2)] == ["LAMPIRAN", "Pasal 75"]

    _use(setup, collection, LAMPIRAN_PENALTY="0.03")
    assert [c.pasal for c in retrieval.retrieve("q", top_k=2)] == ["Pasal 75", "LAMPIRAN"]


def test_translation_merges_both_searches(setup):
    question = "What is the deadline for the complaint?"
    collection = FakeCollection({
        f"query: {question}": [("a::x::0", "Pasal 1", 0.20), ("b::y::0", "Pasal 2", 0.25)],
        "query: Berapa batas waktu pengaduan?": [("c::z::0", "Pasal 3", 0.15), ("a::x::0", "Pasal 1", 0.30)],
    })
    setup.setattr(retrieval, "translate_to_indonesian", lambda q: "Berapa batas waktu pengaduan?")
    _use(setup, collection, TRANSLATE_QUERIES="true", LAMPIRAN_PENALTY="0")

    chunks = retrieval.retrieve(question, top_k=3)
    assert [c.chunk_id for c in chunks] == ["c::z::0", "a::x::0", "b::y::0"]  # merged, no duplicates


def test_bahasa_question_is_not_translated(setup):
    collection = FakeCollection({"query: Apa itu PUJK?": [("a::x::0", "Pasal 1", 0.2)]})

    def fail(q):
        raise AssertionError("should not translate")

    setup.setattr(retrieval, "translate_to_indonesian", fail)
    _use(setup, collection, TRANSLATE_QUERIES="true")
    assert len(retrieval.retrieve("Apa itu PUJK?", top_k=1)) == 1


def test_rrf_rewards_chunks_high_in_both_rankings():
    a, b, c = (Chunk(chunk_id=i, doc_id="d", doc_title="T", pasal=None, page_start=1, text="x") for i in "abc")
    fused = retrieval._rrf([[a, b, c], [b, c]])
    assert [x.chunk_id for x in fused] == ["b", "c", "a"]  # in both lists beats 1st in only one


def test_hybrid_finds_exact_term_the_vectors_missed(setup):
    hits = [("a::x::0", "Pasal 1", 0.10), ("b::y::0", "Pasal 2", 0.11), ("c::z::0", "Pasal 3", 0.40)]
    collection = FakeCollection({"query: rencana kerja LAPS": hits})
    collection.texts = {"a::x::0": "umum", "b::y::0": "umum", "c::z::0": "rencana kerja LAPS 30 September"}

    _use(setup, collection, HYBRID_SEARCH="false", LAMPIRAN_PENALTY="0")
    assert retrieval.retrieve("rencana kerja LAPS", top_k=1)[0].chunk_id == "a::x::0"

    _use(setup, collection, HYBRID_SEARCH="true", LAMPIRAN_PENALTY="0")
    assert retrieval.retrieve("rencana kerja LAPS", top_k=1)[0].chunk_id == "c::z::0"
