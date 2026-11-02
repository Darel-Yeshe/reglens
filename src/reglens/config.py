"""Settings loaded from environment variables or a local .env file.

Every value has a safe default so tests run without an API key.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini (answers)
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.5-flash-lite"

    # Local embedding model (runs on your PC, no API key, no limits)
    embedding_model: str = "intfloat/multilingual-e5-small"

    # ChromaDB (local, file-based)
    chroma_collection: str = "reglens-chunks"

    # Retrieval
    top_k: int = 5
    translate_queries: bool = True  # also search a Bahasa translation of English questions
    lampiran_penalty: float = 0.03  # distance added to attachment chunks (0 = off)
    hybrid_search: bool = False  # also run BM25 keyword search and merge (RRF)

    # Paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    @property
    def llm_ready(self) -> bool:
        """True once a Gemini API key is configured."""
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
