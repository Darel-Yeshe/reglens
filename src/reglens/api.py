"""HTTP API. Run locally with:  uvicorn reglens.api:app --reload --app-dir src"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reglens import generate, retrieval
from reglens.config import get_settings
from reglens.models import Answer

app = FastAPI(title="RegLens", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": get_settings().llm_ready}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    settings = get_settings()
    try:
        chunks = retrieval.retrieve(req.question, top_k=settings.top_k)
        return generate.answer_question(req.question, chunks)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
