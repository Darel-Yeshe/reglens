"""Tests for generate.py with a fake Gemini client, so they run offline and free."""

from types import SimpleNamespace

from reglens import generate
from reglens.models import Chunk

CHUNKS = [
    Chunk(chunk_id="pojk-22::pasal-37::0", doc_id="pojk-22", doc_title="POJK 22 Tahun 2023",
          pasal="Pasal 37", page_start=30, text="Pelaku usaha wajib menanggapi pengaduan paling lama 10 hari kerja."),
    Chunk(chunk_id="pojk-22::preamble::0", doc_id="pojk-22", doc_title="POJK 22 Tahun 2023",
          pasal=None, page_start=1, text="Menimbang bahwa pelindungan konsumen perlu diatur."),
]


def _fake_client(answer: str, sources: list[int], answered: bool):
    parsed = generate._LLMAnswer(answer=answer, sources=sources, answered=answered)
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed=parsed, text=parsed.model_dump_json())

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


def test_no_chunks_means_not_answered_without_calling_the_model(monkeypatch):
    monkeypatch.setattr(generate.genai, "Client", lambda **_: (_ for _ in ()).throw(AssertionError("called")))
    result = generate.answer_question("apa saja?", [])
    assert result.answered is False and result.citations == []


def test_sources_are_numbered_in_the_prompt(monkeypatch):
    client, calls = _fake_client("10 hari kerja [1].", [1], True)
    monkeypatch.setattr(generate.genai, "Client", lambda **_: client)
    generate.answer_question("berapa lama?", CHUNKS)
    prompt = calls[0]["contents"]
    assert "[1] POJK 22 Tahun 2023, Pasal 37" in prompt
    assert "[2] POJK 22 Tahun 2023, Pembukaan" in prompt
    assert calls[0]["config"].temperature == 0


def test_citations_map_back_to_chunks_and_skip_bad_numbers(monkeypatch):
    client, _ = _fake_client("10 hari kerja [1][1][7].", [1, 1, 7], True)
    monkeypatch.setattr(generate.genai, "Client", lambda **_: client)
    result = generate.answer_question("berapa lama?", CHUNKS)
    assert [c.chunk_id for c in result.citations] == ["pojk-22::pasal-37::0"]
    assert result.citations[0].pasal == "Pasal 37"


def test_refusal_has_no_citations(monkeypatch):
    client, _ = _fake_client("Tidak ditemukan.", [2], False)
    monkeypatch.setattr(generate.genai, "Client", lambda **_: client)
    result = generate.answer_question("harga bitcoin?", CHUNKS)
    assert result.answered is False and result.citations == []


def test_retries_when_gemini_is_busy(monkeypatch):
    client, calls = _fake_client("10 hari kerja [1].", [1], True)
    real = client.models.generate_content
    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise generate.errors.ServerError(503, {"error": {"message": "busy"}}, None)
        return real(**kwargs)

    client.models.generate_content = flaky
    monkeypatch.setattr(generate.genai, "Client", lambda **_: client)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    result = generate.answer_question("berapa lama?", CHUNKS)
    assert attempts["n"] == 2 and result.answered is True
