"""Write an answer from retrieved chunks, with citations, or say "I don't know".

Written by Dave with Claude.

How it works:
1. The retrieved chunks are numbered [1]..[5] in the prompt.
2. Gemini must answer ONLY from those sources and cite them by number.
3. Gemini returns JSON (answer, sources, answered) that matches a schema,
   so we parse it reliably instead of scraping text.
4. The source numbers are mapped back to the chunks to build Citations.
"""

import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from reglens.config import get_settings
from reglens.models import Answer, Chunk, Citation

SYSTEM_INSTRUCTION = """You answer questions about Indonesian financial regulations (OJK).
Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge.
2. Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
3. If the sources do not contain the answer, set "answered" to false and say
   briefly that the answer is not in the regulations provided.
4. Answer in the same language as the question.
5. Be concise: 1 to 4 sentences."""

NOT_FOUND = "Maaf, saya tidak menemukan jawabannya dalam regulasi yang tersedia."

MAX_ATTEMPTS = 4
RETRY_CODES = {429, 500, 503}  # rate limited, server error, overloaded


class _LLMAnswer(BaseModel):
    """The JSON shape Gemini must return."""

    answer: str
    sources: list[int]
    answered: bool


def _format_sources(chunks: list[Chunk]) -> str:
    """Number each chunk so the model can cite it as [1], [2], ..."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        label = chunk.pasal or "Pembukaan"
        blocks.append(f"[{number}] {chunk.doc_title}, {label} (page {chunk.page_start}):\n{chunk.text}")
    return "\n\n".join(blocks)


def _citations(source_numbers: list[int], chunks: list[Chunk]) -> list[Citation]:
    """Turn [1], [3] back into Citation objects, ignoring duplicates and invalid numbers."""
    citations = []
    for number in dict.fromkeys(source_numbers):
        if 1 <= number <= len(chunks):
            chunk = chunks[number - 1]
            citations.append(
                Citation(doc_title=chunk.doc_title, pasal=chunk.pasal, chunk_id=chunk.chunk_id)
            )
    return citations


def generate_json(system_instruction: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Call Gemini and get back an object of `schema`, retrying if Gemini is busy.

    Shared by answer generation and the evaluation judge.
    """
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=settings.gemini_chat_model, contents=prompt, config=config
            )
            break
        except errors.APIError as err:
            if err.code not in RETRY_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = 5 * 2**attempt
            print(f"Gemini busy ({err.code}), retrying in {wait}s...")
            time.sleep(wait)
    return response.parsed or schema.model_validate_json(response.text)


def answer_question(question: str, chunks: list[Chunk]) -> Answer:
    """Generate a cited answer from the given chunks."""
    if not chunks:
        return Answer(question=question, answer=NOT_FOUND, answered=False)

    prompt = f"Sources:\n\n{_format_sources(chunks)}\n\nQuestion: {question}"
    parsed = generate_json(SYSTEM_INSTRUCTION, prompt, _LLMAnswer)

    return Answer(
        question=question,
        answer=parsed.answer,
        citations=_citations(parsed.sources, chunks) if parsed.answered else [],
        answered=parsed.answered,
    )
