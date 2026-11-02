"""Run the golden question set through RegLens and print the scores.

Usage (venv active, after `python scripts/ingest.py --index`):
    python evals/run_evals.py --retrieval-only   # search only: fast, no answers (Gemini only to translate if TRANSLATE_QUERIES=true)
    python evals/run_evals.py --no-judge         # + answers, citations, refusals
    python evals/run_evals.py                    # + LLM judge (correctness, faithfulness)
    python evals/run_evals.py --check            # exit with an error if below thresholds.json

Each run is saved to evals/results/ so you can compare before/after a change.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reglens.config import get_settings  # noqa: E402
from reglens.evaluation import (  # noqa: E402
    citation_correct,
    failed_thresholds,
    judge,
    load_golden,
    retrieval_rank,
    summarize,
)
from reglens.generate import answer_question  # noqa: E402
from reglens.retrieval import retrieve  # noqa: E402

EVALS = ROOT / "evals"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="only score search (no answers, no judge)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between Gemini calls (free tier)")
    parser.add_argument("--check", action="store_true", help="fail if metrics are below thresholds.json")
    args = parser.parse_args()

    golden = load_golden(EVALS / "golden_set.jsonl")[: args.limit]
    settings = get_settings()
    top_k = settings.top_k
    config = {"top_k": top_k, "translate_queries": settings.translate_queries,
              "lampiran_penalty": settings.lampiran_penalty, "hybrid_search": settings.hybrid_search}
    print(f"Settings: {config}\n")
    results = []

    for row in golden:
        chunks = retrieve(row["question"], top_k=top_k)
        result = {"id": row["id"], "in_scope": row["in_scope"], "question": row["question"],
                  "rank": retrieval_rank(chunks, row["expected"]) if row["in_scope"] else None}

        if not args.retrieval_only:
            answer = answer_question(row["question"], chunks)
            result |= {"answered": answer.answered, "answer": answer.answer,
                       "citation_correct": citation_correct(answer, chunks, row["expected"])}
            time.sleep(args.delay)

            if row["in_scope"] and answer.answered and not args.no_judge:
                verdict = judge(row["question"], row["reference_answer"], answer, chunks)
                result |= {"correct": verdict.correct, "faithful": verdict.faithful, "reason": verdict.reason}
                time.sleep(args.delay)

        results.append(result)
        _print_row(result)

    metrics = summarize(results)
    print("\n=== Summary ===")
    for name, value in metrics.items():
        print(f"  {name:20} {'-' if value is None else value}")

    out_dir = EVALS / "results"
    out_dir.mkdir(exist_ok=True)
    mode = "retrieval" if args.retrieval_only else "nojudge" if args.no_judge else "full"
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M}-{mode}.json"
    out.write_text(json.dumps({"settings": config, "metrics": metrics, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")

    if args.check:
        failures = failed_thresholds(metrics, json.loads((EVALS / "thresholds.json").read_text()))
        if failures:
            sys.exit("Below threshold:\n  " + "\n  ".join(failures))
        print("All thresholds met.")


def _print_row(r: dict) -> None:
    if r["in_scope"]:
        found = f"rank {r['rank']}" if r["rank"] else "NOT FOUND"
    else:
        found = "out-of-scope"
    extra = ""
    if "answered" in r:
        extra += " | answered" if r["answered"] else " | refused"
        if r["in_scope"] and r["answered"]:
            extra += " | cite OK" if r["citation_correct"] else " | cite WRONG"
    if "correct" in r:
        extra += f" | correct={r['correct']} faithful={r['faithful']}"
    print(f"{r['id']} | {found}{extra} | {r['question'][:60]}")


if __name__ == "__main__":
    main()
