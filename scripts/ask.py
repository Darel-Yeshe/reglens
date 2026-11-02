"""Ask RegLens a question from the terminal.

Usage (venv active, after `python scripts/ingest.py --index`):
    python scripts/ask.py "Berapa lama pelaku usaha wajib menanggapi pengaduan?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reglens.config import get_settings  # noqa: E402
from reglens.generate import answer_question  # noqa: E402
from reglens.retrieval import retrieve  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python scripts/ask.py "your question"')
    question = " ".join(sys.argv[1:])

    chunks = retrieve(question, top_k=get_settings().top_k)
    print("\nRetrieved:")
    for number, c in enumerate(chunks, start=1):
        print(f"  [{number}] {c.doc_title[:50]} | {c.pasal or '-'} | page {c.page_start}")

    answer = answer_question(question, chunks)
    print(f"\nAnswer ({'answered' if answer.answered else 'NOT answered'}):\n  {answer.answer}")
    if answer.citations:
        print("\nCitations:")
        for line in dict.fromkeys(f"{c.doc_title}, {c.pasal or 'Pembukaan'}" for c in answer.citations):
            print(f"  - {line}")


if __name__ == "__main__":
    main()
