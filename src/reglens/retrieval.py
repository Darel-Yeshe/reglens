"""Index chunks into ChromaDB and retrieve the best ones for a question.

Written by Dave with Claude.

Concepts:
- Embedding: a list of numbers representing a text's meaning. Similar
  meanings -> nearby vectors, even across Bahasa and English.
- Embedding model: intfloat/multilingual-e5-small, running locally on your
  PC via sentence-transformers. Free, no API key, no rate limits. Downloaded
  once (~500 MB) the first time it's used.
- ChromaDB: a vector database that runs inside your Python process and
  saves to a folder (settings.chroma_dir). No server, no account.
- Cosine distance: how Chroma compares two vectors (based on the angle
  between them). Chroma returns distances, so SMALLER = more similar.

Two optional tweaks (switched on in .env, measured with evals/run_evals.py):
- TRANSLATE_QUERIES: an English question is also translated to Bahasa
  Indonesia by Gemini, both versions are searched, and the results merged.
- LAMPIRAN_PENALTY: added to the distance of attachment (LAMPIRAN) chunks,
  so a regulation's own text outranks an attachment that quotes it.
- HYBRID_SEARCH: also run a keyword search (BM25) and merge both rankings with
  Reciprocal Rank Fusion. Vectors catch meaning; BM25 catches exact terms
  ("LAPS", "30 September") that vectors can blur.
"""

import re
from functools import lru_cache

from pydantic import BaseModel

from reglens.chunking import is_lampiran
from reglens.config import get_settings
from reglens.models import Chunk

BATCH_SIZE = 32
ADD_BATCH = 500
FETCH_FACTOR = 3  # fetch 3x top_k candidates, then re-rank and keep top_k
RRF_K = 60  # standard Reciprocal Rank Fusion constant: higher = flatter
LAMPIRAN_BM25_FACTOR = 0.5  # BM25 score multiplier for attachments (when LAMPIRAN_PENALTY > 0)

# Words too common to help a keyword search
STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dalam", "dengan", "oleh", "pada",
    "tidak", "adalah", "apa", "apakah", "berapa", "bagaimana", "kapan", "siapa", "harus",
    "wajib", "itu", "ini", "sebagai", "juga", "dapat", "the", "of", "a", "an", "to", "in",
    "is", "are", "what", "how", "when", "which", "must", "does", "do", "for", "and", "or",
}

# Small, common words that tell the two languages apart
ENGLISH_WORDS = {
    "the", "what", "is", "are", "how", "when", "which", "who", "does", "do", "must",
    "should", "of", "for", "to", "in", "a", "an", "can", "within", "many", "much",
    "and", "or", "by", "be", "after", "before", "their", "its", "if", "with",
}
INDONESIAN_WORDS = {
    "yang", "apa", "apakah", "berapa", "bagaimana", "kapan", "siapa", "wajib", "harus",
    "dan", "atau", "di", "ke", "dari", "untuk", "dalam", "dengan", "oleh", "pada",
    "tidak", "adalah", "jika", "setelah", "sebelum", "paling", "lama", "hari", "bagi",
}


@lru_cache
def _model():
    """The embedding model, loaded once and reused.

    Imported here, not at the top of the file: sentence-transformers pulls in
    PyTorch, which takes a long time to import. Importing it only when a model
    is actually needed keeps tests and the API startup fast.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def _collection():
    """The ChromaDB collection where chunks live (created on first use)."""
    import chromadb  # imported lazily for the same reason as above

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _doc_text(chunk: Chunk) -> str:
    """How a chunk is written for embedding (e5 models expect a "passage: " prefix)."""
    return f"passage: {chunk.doc_title} {chunk.pasal or ''}\n{chunk.text}"


def _query_text(question: str) -> str:
    """How a question is written for embedding (e5 models expect a "query: " prefix)."""
    return f"query: {question}"


def _embed(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Turn texts into vectors on your own computer, in batches of 32."""
    vectors = _model().encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def build_index(chunks: list[Chunk]) -> int:
    """Embed all chunks and store them in ChromaDB. Returns the count stored."""
    collection = _collection()
    vectors = _embed([_doc_text(c) for c in chunks], show_progress=True)
    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i : i + ADD_BATCH]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[i : i + ADD_BATCH],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "pasal": c.pasal or "",
                    "page_start": c.page_start,
                }
                for c in batch
            ],
        )
    return collection.count()


def looks_english(text: str) -> bool:
    """Guess the language by counting common English vs Indonesian words."""
    words = re.findall(r"[a-z]+", text.lower())
    english = sum(w in ENGLISH_WORDS for w in words)
    indonesian = sum(w in INDONESIAN_WORDS for w in words)
    return english > indonesian


class _Translation(BaseModel):
    indonesian: str


TRANSLATE_INSTRUCTION = """Translate the user's question about Indonesian financial regulations
into formal Bahasa Indonesia, using the terms OJK regulations use (e.g. "Pelaku Usaha Jasa
Keuangan", "pengaduan", "Konsumen"). Keep abbreviations like OJK, PUJK, LAPS SJK as they are.
Return only the translation."""


@lru_cache(maxsize=256)
def translate_to_indonesian(question: str) -> str | None:
    """Ask Gemini for a Bahasa version of the question (None if Gemini fails)."""
    from reglens.generate import generate_json  # lazy: only needed when translating

    try:
        return generate_json(TRANSLATE_INSTRUCTION, question, _Translation).indonesian.strip() or None
    except Exception as err:  # search still works with the original question
        print(f"Translation failed ({err}); searching the original question only.")
        return None


def _search(collection, query: str, n: int) -> list[tuple[float, Chunk]]:
    """One vector search: (distance, chunk) pairs, closest first."""
    [vector] = _embed([_query_text(query)])
    res = collection.query(query_embeddings=[vector], n_results=n)
    return [
        (distance, _to_chunk(chunk_id, text, meta))
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    """Rebuild a Chunk from what ChromaDB stores."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=meta["doc_id"],
        doc_title=meta["doc_title"],
        pasal=meta["pasal"] or None,
        page_start=meta["page_start"],
        text=text,
    )


def _vector_ranked(collection, queries: list[str], n: int, penalty: float) -> list[Chunk]:
    """Vector search for each query; each chunk keeps its best distance (+ penalty if attachment)."""
    best: dict[str, tuple[float, Chunk]] = {}
    for query in queries:
        for distance, chunk in _search(collection, query, n):
            if is_lampiran(chunk.pasal):
                distance += penalty
            if chunk.chunk_id not in best or distance < best[chunk.chunk_id][0]:
                best[chunk.chunk_id] = (distance, chunk)
    return [chunk for _, chunk in sorted(best.values(), key=lambda pair: pair[0])]


def _tokens(text: str) -> list[str]:
    """Lowercase words without stopwords: what BM25 compares."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


@lru_cache
def _bm25_index():
    """Build the BM25 keyword index from every chunk in ChromaDB (once per run)."""
    from rank_bm25 import BM25Okapi  # lazy: only needed for hybrid search

    stored = _collection().get(include=["documents", "metadatas"])
    chunks = [_to_chunk(i, t, m) for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"])]
    # Title and label are included so "LAPS" or "Pasal 85" can match too
    bm25 = BM25Okapi([_tokens(f"{c.doc_title} {c.pasal or ''}\n{c.text}") for c in chunks])
    return bm25, chunks


def _keyword_ranked(queries: list[str], n: int, demote_lampiran: bool) -> list[Chunk]:
    """BM25 search for each query; each chunk keeps its best score."""
    bm25, chunks = _bm25_index()
    best = [0.0] * len(chunks)
    for query in queries:
        for i, score in enumerate(bm25.get_scores(_tokens(query))):
            if demote_lampiran and is_lampiran(chunks[i].pasal):
                score *= LAMPIRAN_BM25_FACTOR
            best[i] = max(best[i], score)
    order = sorted(range(len(chunks)), key=lambda i: best[i], reverse=True)
    return [chunks[i] for i in order[:n] if best[i] > 0]


def _rrf(rankings: list[list[Chunk]]) -> list[Chunk]:
    """Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over the rankings.

    Uses ranks, not scores, because cosine distances and BM25 scores are on
    different scales and can't be added directly.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (RRF_K + rank)
            by_id[chunk.chunk_id] = chunk
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


def retrieve(question: str, top_k: int = 5) -> list[Chunk]:
    """Return the top_k chunks most relevant to the question."""
    settings = get_settings()
    collection = _collection()
    if collection.count() == 0:
        return []

    # Step 1: which queries to search (the question, plus a Bahasa version if enabled)
    queries = [question]
    if settings.translate_queries and looks_english(question):
        translated = translate_to_indonesian(question)
        if translated:
            queries.append(translated)

    # Step 2: vector search (fetch extra candidates when we re-rank afterwards)
    tweaked = len(queries) > 1 or settings.lampiran_penalty > 0 or settings.hybrid_search
    n = min(top_k * FETCH_FACTOR if tweaked else top_k, collection.count())
    ranked = _vector_ranked(collection, queries, n, settings.lampiran_penalty)

    # Step 3: optionally merge with a keyword (BM25) ranking
    if settings.hybrid_search:
        keyword = _keyword_ranked(queries, n, demote_lampiran=settings.lampiran_penalty > 0)
        ranked = _rrf([ranked, keyword])

    return ranked[:top_k]
