"""Tests for the scoring functions (no Gemini, no ChromaDB)."""

import json

import pytest

from reglens.evaluation import (
    citation_correct,
    failed_thresholds,
    load_golden,
    retrieval_rank,
    summarize,
)
from reglens.models import Answer, Chunk, Citation

EXPECTED = [{"doc_id": "pojk-22", "pasal": "Pasal 75"}]


def _chunk(doc_id: str, pasal: str | None, n: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}::{pasal}::{n}", doc_id=doc_id, doc_title=doc_id.upper(),
                 pasal=pasal, page_start=1, text="isi")


def test_retrieval_rank_finds_first_match():
    chunks = [_chunk("pojk-22", "Pasal 74"), _chunk("seojk-18", "VI. X"), _chunk("pojk-22", "Pasal 75")]
    assert retrieval_rank(chunks, EXPECTED) == 3


def test_retrieval_rank_none_when_missing():
    assert retrieval_rank([_chunk("pojk-22", "Pasal 74")], EXPECTED) is None


def test_same_pasal_in_another_document_does_not_count():
    assert retrieval_rank([_chunk("pojk-8", "Pasal 75")], EXPECTED) is None


def test_citation_correct_uses_doc_and_pasal():
    right, wrong = _chunk("pojk-22", "Pasal 75"), _chunk("pojk-8", "Pasal 75")
    cite = lambda c: Citation(doc_title=c.doc_title, pasal=c.pasal, chunk_id=c.chunk_id)  # noqa: E731
    good = Answer(question="q", answer="a", citations=[cite(right)])
    bad = Answer(question="q", answer="a", citations=[cite(wrong)])
    assert citation_correct(good, [right, wrong], EXPECTED) is True
    assert citation_correct(bad, [right, wrong], EXPECTED) is False


def test_summarize_metrics():
    results = [
        {"in_scope": True, "rank": 1, "answered": True, "citation_correct": True, "correct": True, "faithful": True},
        {"in_scope": True, "rank": 2, "answered": True, "citation_correct": False, "correct": False, "faithful": True},
        {"in_scope": True, "rank": None, "answered": False, "citation_correct": False},
        {"in_scope": False, "rank": None, "answered": False, "citation_correct": False},
    ]
    m = summarize(results)
    assert m["retrieval_hit_rate"] == round(2 / 3, 3)
    assert m["mrr"] == round((1 + 0.5) / 3, 3)
    assert m["answer_rate"] == round(2 / 3, 3)
    assert m["citation_accuracy"] == 0.5
    assert m["refusal_accuracy"] == 1.0
    assert m["correctness"] == 0.5 and m["faithfulness"] == 1.0


def test_summarize_retrieval_only_leaves_answer_metrics_empty():
    m = summarize([{"in_scope": True, "rank": 1}, {"in_scope": False, "rank": None}])
    assert m["retrieval_hit_rate"] == 1.0
    assert m["answer_rate"] is None and m["citation_accuracy"] is None and m["refusal_accuracy"] is None


def test_failed_thresholds_skips_unmeasured():
    metrics = {"retrieval_hit_rate": 0.6, "faithfulness": None}
    assert failed_thresholds(metrics, {"retrieval_hit_rate": 0.7, "faithfulness": 0.9}) == ["retrieval_hit_rate: 0.6 < 0.7"]


def test_load_golden_rejects_missing_fields(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"id": "q1", "question": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_golden(path)


def test_real_golden_set_is_valid():
    from reglens.config import PROJECT_ROOT
    rows = load_golden(PROJECT_ROOT / "evals" / "golden_set.jsonl")
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["expected"] for r in rows if r["in_scope"])
