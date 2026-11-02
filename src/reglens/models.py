"""Data shapes shared across the pipeline: loader -> chunker -> index -> answer."""

from pydantic import BaseModel, Field


class Page(BaseModel):
    """One cleaned page of text from a source PDF."""

    doc_id: str = Field(description="Stable id from the file name, e.g. 'pojk-22-2023'")
    doc_title: str = Field(description="Human-readable title shown in citations")
    page_number: int = Field(ge=1)
    text: str


class Chunk(BaseModel):
    """A piece of a regulation small enough to retrieve and cite."""

    chunk_id: str = Field(description="Unique, stable id, e.g. 'pojk-22-2023::pasal-37::0'")
    doc_id: str
    doc_title: str
    pasal: str | None = Field(default=None, description="Article label, e.g. 'Pasal 37'")
    page_start: int = Field(ge=1)
    text: str


class Citation(BaseModel):
    doc_title: str
    pasal: str | None = None
    chunk_id: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []
    answered: bool = Field(
        default=True,
        description="False when the system says it doesn't know (out of scope / not found)",
    )
