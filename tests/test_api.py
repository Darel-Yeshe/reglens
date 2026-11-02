from fastapi.testclient import TestClient

from reglens import api
from reglens.models import Answer

client = TestClient(api.app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_ask_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_returns_answer(monkeypatch):
    """Retrieval and Gemini are faked, so this test is offline, free and fast."""
    monkeypatch.setattr(api.retrieval, "retrieve", lambda q, top_k: [])
    monkeypatch.setattr(
        api.generate,
        "answer_question",
        lambda q, chunks: Answer(question=q, answer="Tidak ditemukan.", answered=False),
    )
    res = client.post("/ask", json={"question": "Berapa lama waktu menanggapi pengaduan?"})
    assert res.status_code == 200
    assert res.json()["answered"] is False
