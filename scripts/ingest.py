"""Load PDFs -> chunk -> save chunks.jsonl (and optionally index them in ChromaDB).

Usage (from the project root, venv active):
    python scripts/ingest.py            # load + chunk, print stats
    python scripts/ingest.py --index    # also embed + store in ChromaDB
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.chunking import chunk_pages  # noqa: E402
from reglens.config import get_settings  # noqa: E402
from reglens.loader import load_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="embed chunks and store them in ChromaDB")
    args = parser.parse_args()

    settings = get_settings()
    pages = load_folder(settings.raw_dir)
    if not pages:
        sys.exit(f"No PDFs with text found in {settings.raw_dir}")
    docs = sorted({p.doc_id for p in pages})
    print(f"Loaded {len(pages)} pages from {len(docs)} documents")

    chunks = chunk_pages(pages)
    sizes = sorted(len(c.text) for c in chunks)
    print(
        f"Made {len(chunks)} chunks | chars min {sizes[0]}, "
        f"median {sizes[len(sizes) // 2]}, max {sizes[-1]} | "
        f"{sum(c.pasal is None for c in chunks)} without a pasal label"
    )

    out = settings.processed_dir / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
    print(f"Saved {out}")

    if args.index:
        from reglens.retrieval import build_index

        print(f"Indexed {build_index(chunks)} chunks into ChromaDB collection '{settings.chroma_collection}'")


if __name__ == "__main__":
    main()
