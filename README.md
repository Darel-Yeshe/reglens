# RegLens

Ask questions about Indonesian financial regulations (OJK) in Bahasa or English, and get answers that cite the exact article, or an honest "I don't know".

> Status: working end to end (search, cited answers, web demo, evaluation). Live demo link coming soon.

## How it works

1. **Index (once):** 15 regulation PDFs (1,073 pages) → clean text (PyMuPDF, pypdf fallback) → one chunk per article (*Pasal*) or circular section, attachments (*Lampiran*) labelled separately → 1,672 chunks → local embeddings (`multilingual-e5-small`) → ChromaDB.
2. **Search:** English questions are also translated to Bahasa Indonesia (Gemini) and both versions are searched; attachment chunks get a small distance penalty so the regulation's own text outranks attachments that quote it; top 5 chunks are kept.
3. **Answer:** Gemini answers **only** from those 5 chunks, cites them, and returns structured JSON. If the chunks don't contain the answer, it says so instead of guessing.
4. **Evaluate:** a golden question set (in-scope and out-of-scope, Bahasa and English) is scored on every change: retrieval hit rate, MRR, citation accuracy, correct refusals, plus an LLM judge for correctness and faithfulness.

## Project layout

```
src/reglens/
  config.py      settings from .env
  models.py      Page, Chunk, Answer
  loader.py      PDF -> clean pages
  chunking.py    pages -> article-level chunks
  retrieval.py   local embeddings + ChromaDB search, translation, Lampiran penalty, optional BM25 hybrid
  generate.py    cited answers (Gemini)
  api.py         FastAPI: /health, /ask
app/streamlit_app.py  web demo
scripts/ingest.py  load -> chunk -> (optionally) index
scripts/ask.py     ask one question from the terminal
evals/             golden question set + scoring harness
tests/             unit tests
data/raw/          your PDFs (not committed)
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then add your Gemini API key (free at aistudio.google.com)
python -m pytest
python scripts/ingest.py --index   # PDFs in data/raw/ -> ChromaDB (~6 min on a laptop CPU)
```

- Web demo: `python -m streamlit run app/streamlit_app.py`
- Terminal: `python scripts/ask.py "Berapa lama PUJK wajib menanggapi pengaduan?"`
- API: `uvicorn reglens.api:app --reload --app-dir src`, then open http://127.0.0.1:8000/docs
- Evaluation: `python evals/run_evals.py` (add `--retrieval-only` for a fast search-only run)

## Data

Public regulations from [ojk.go.id](https://www.ojk.go.id). PDFs are kept out of the repo; put them in `data/raw/`.

## Results

Golden set: 26 questions (22 answerable from the regulations, 4 out of scope), Bahasa and English.

| Metric | Baseline | Current |
| --- | --- | --- |
| Retrieval hit rate (right article in top 5) | 0.727 | **0.818–0.864** |
| MRR (how high the right article ranks) | 0.585 | **0.668–0.680** |
| Answer rate (answerable questions answered) | 0.864 | **0.955** |
| Citation accuracy | 0.842 | **0.857** |
| Correct "I don't know" (out of scope) | 1.0 | **1.0** |
| Faithfulness (LLM judge) | 1.0 | **1.0** |
| Correctness (LLM judge) | 0.895 | 0.857 |

Baseline = vector search only, pypdf text. What changed, each measured separately:

| Experiment | Effect | Kept? |
| --- | --- | --- |
| Translate English questions to Bahasa before searching | hit rate +0.09 | ✅ |
| Label attachments (*Lampiran*) and push them down the ranking | hit rate +0.05, MRR +0.05 | ✅ |
| Both together | hit rate 0.727 → 0.864 | ✅ |
| PyMuPDF text extraction (fixes split words like "d an") | MRR +0.01, cleaner citations | ✅ |
| Hybrid search (BM25 + vectors, Reciprocal Rank Fusion) | MRR +0.06 but hit rate −0.09 | ❌ (switch: `HYBRID_SEARCH`) |

Hit rate varies slightly between runs because Gemini's translations aren't always word-for-word identical.
Known gaps: questions whose answer is restated in a second regulation (the system answers correctly but cites the other one), and multi-part answers where one condition is left out.
