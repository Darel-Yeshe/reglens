"""Score RegLens against the golden question set.

Written by Dave with Claude.

Metrics (in-scope = questions the regulations can answer):
- retrieval_hit_rate: share of in-scope questions where an expected article
  is among the top-k retrieved chunks. Measures SEARCH.
- mrr: mean reciprocal rank of the first expected article (1 = always first,
  0.5 = usually second, 0 = never found). Rewards finding it EARLY.
- answer_rate: share of in-scope questions the system answered (didn't refuse).
- citation_accuracy: share of answered in-scope questions that cite an
  expected article. Measures whether the ANSWER points to the right source.
- refusal_accuracy: share of out-of-scope questions correctly refused.
- correctness / faithfulness (only with the LLM judge): is the answer the same
  as the reference answer, and is every claim supported by the cited sources?
"""

import json
from pathlib import Path

from pydantic import BaseModel

from reglens.generate import generate_json
from reglens.models import Answer, Chunk

REQUIRED_KEYS = {"id", "lang", "in_scope", "question", "expected", "reference_answer"}


def load_golden(path: Path) -> list[dict]:
    """Read the golden set (one JSON object per line), checking every row has the needed fields."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_KEYS - row.keys()
        if missing:
            raise ValueError(f"{path.name} line {line_number}: missing {sorted(missing)}")
        rows.append(row)
    return rows


def _matches(doc_id: str, pasal: str | None, expected: list[dict]) -> bool:
    return any(doc_id == e["doc_id"] and pasal == e["pasal"] for e in expected)


def retrieval_rank(chunks: list[Chunk], expected: list[dict]) -> int | None:
    """1-based position of the first retrieved chunk that is an expected article, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if _matches(chunk.doc_id, chunk.pasal, expected):
            return rank
    return None


def citation_correct(answer: Answer, chunks: list[Chunk], expected: list[dict]) -> bool:
    """True if any citation in the answer points to an expected article."""
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    return any(_matches(doc_of.get(c.chunk_id, ""), c.pasal, expected) for c in answer.citations)


class Judgement(BaseModel):
    correct: bool
    faithful: bool
    reason: str


JUDGE_INSTRUCTION = """You grade answers from a question-answering system about Indonesian financial regulations.
Given the question, a reference answer written by an expert, the system's answer, and the
sources the system cited, decide:
- correct: true if the system's answer states the same key facts as the reference answer
  (numbers, deadlines, conditions). Extra correct detail is fine; a wrong or missing key fact is not.
- faithful: true if every claim in the system's answer is supported by the cited sources.
Give a one-sentence reason. Be strict about numbers and deadlines."""


def judge(question: str, reference: str, answer: Answer, chunks: list[Chunk]) -> Judgement:
    """Ask Gemini to grade one answer (LLM-as-a-judge)."""
    cited = {c.chunk_id for c in answer.citations}
    sources = "\n\n".join(f"- {c.doc_title}, {c.pasal}:\n{c.text}" for c in chunks if c.chunk_id in cited)
    prompt = (
        f"Question: {question}\n\nReference answer: {reference}\n\n"
        f"System answer: {answer.answer}\n\nCited sources:\n{sources or '(none)'}"
    )
    return generate_json(JUDGE_INSTRUCTION, prompt, Judgement)


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    """Turn per-question results into the headline metrics."""
    in_scope = [r for r in results if r["in_scope"]]
    out_scope = [r for r in results if not r["in_scope"]]
    answered = [r for r in in_scope if r.get("answered")]
    judged = [r for r in in_scope if r.get("correct") is not None]
    return {
        "questions": len(results),
        "retrieval_hit_rate": _rate([r["rank"] is not None for r in in_scope]),
        "mrr": round(sum(1 / r["rank"] for r in in_scope if r["rank"]) / len(in_scope), 3) if in_scope else None,
        "answer_rate": _rate([r["answered"] for r in in_scope if "answered" in r]),
        "citation_accuracy": _rate([r["citation_correct"] for r in answered]),
        "refusal_accuracy": _rate([not r["answered"] for r in out_scope if "answered" in r]),
        "correctness": _rate([r["correct"] for r in judged]),
        "faithfulness": _rate([r["faithful"] for r in judged]),
    }


def failed_thresholds(metrics: dict, thresholds: dict) -> list[str]:
    """Names of metrics below their minimum (metrics that weren't measured are skipped)."""
    return [
        f"{name}: {metrics[name]} < {minimum}"
        for name, minimum in thresholds.items()
        if metrics.get(name) is not None and metrics[name] < minimum
    ]
