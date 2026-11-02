# Evaluation

`golden_set.jsonl` is the regression test suite for RegLens: one question per line, with where the answer lives and what it should say.

| Field | Meaning |
| --- | --- |
| `id` | Stable id: `q001`, `q002`, ... |
| `lang` | `id` (Bahasa) or `en` |
| `in_scope` | `false` for questions the regulations can't answer; the system must refuse these |
| `question` | Phrased the way a real user would ask |
| `expected` | List of `{doc_id, pasal}` where the answer lives (more than one if several regulations say the same thing). Empty for out-of-scope |
| `reference_answer` | The correct answer, checked by a human against the PDF |

`doc_id` and `pasal` must match the chunk labels exactly (see `data/processed/chunks.jsonl`).
Write questions **without** looking at what the system answers, so the set stays honest.

## Running

```powershell
python evals/run_evals.py --retrieval-only   # search only: free, ~1 min
python evals/run_evals.py --no-judge         # + answers, citations, refusals (Gemini)
python evals/run_evals.py                    # + LLM judge: correctness, faithfulness
python evals/run_evals.py --check            # fail if below thresholds.json
```

Each run is saved in `results/` so you can compare before and after a change.

## Metrics

| Metric | Question it answers |
| --- | --- |
| `retrieval_hit_rate` | Did search put the right article in the top 5? |
| `mrr` | How high up was it? (1.0 = always first) |
| `answer_rate` | How often did it answer an answerable question? |
| `citation_accuracy` | When it answered, did it cite the right article? |
| `refusal_accuracy` | Did it refuse the out-of-scope questions? |
| `correctness` | Does the answer match the reference? (LLM judge) |
| `faithfulness` | Is every claim backed by the cited sources? (LLM judge) |
