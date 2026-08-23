# GRASP-RAG Experiments

## Prerequisites

Ensure `.env` has all four API keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

Install dependencies:
```bash
pip install anthropic google-genai openai
```

Domains: **agriculture, cs, legal, health**  
(health needs indexing first — see Setup below)

---

## Setup — Index corpora

```bash
python experiments/index_corpus.py --domain agriculture
python experiments/index_corpus.py --domain cs
python experiments/index_corpus.py --domain legal
python experiments/index_corpus.py --domain health    # not yet indexed — run this first
```

---

## Experiment 1 — Win Rate (UltraDomain, reference-anchored)

> Core Win Rate experiment. Evaluates GRASP-RAG vs NaiveRAG / LightRAG / PathRAG
> using ALL UltraDomain questions per domain, with the reference answer as the
> evaluation anchor. Outcome is Win / Lose / Undefined only — no Tie.
>
> Win definition: GRASP-RAG wins both forward + reverse orderings for ≥2 judges.
> Lose definition: Baseline wins both forward + reverse orderings for ≥2 judges.
> Undefined: judges disagree on ordering, ≥2 judges mark Cannot Evaluate, or
>            one system answers "don't know" while the other gives a wrong answer.

### Step 1 — Extract questions from raw dataset

```bash
python experiments/generate_questions.py --domain agriculture --mode ultradomain
python experiments/generate_questions.py --domain cs          --mode ultradomain
python experiments/generate_questions.py --domain legal       --mode ultradomain
python experiments/generate_questions.py --domain health      --mode ultradomain
```

### Step 2 — Run all methods

```bash
python experiments/run_queries.py --domain agriculture --ultradomain --methods naive_rag lightrag grasp_rag pathrag
python experiments/run_queries.py --domain cs          --ultradomain --methods naive_rag lightrag grasp_rag pathrag
python experiments/run_queries.py --domain legal       --ultradomain --methods naive_rag lightrag grasp_rag pathrag
python experiments/run_queries.py --domain health      --ultradomain --methods naive_rag lightrag grasp_rag pathrag
```

> Legal has 438 questions — this will take a while. Script is resumable.

### Step 3 — Evaluate (reference-anchored)

```bash
python experiments/evaluate.py --domain agriculture --ultradomain
python experiments/evaluate.py --domain cs          --ultradomain
python experiments/evaluate.py --domain legal       --ultradomain
python experiments/evaluate.py --domain health      --ultradomain
```

---

## Experiment 2 — Motif Classification Accuracy + Per-type Win Rate

> Two goals:
>   A. Measure GRASP-RAG's classifier accuracy vs GPT-4o oracle labels
>   B. Show per-query-type win rate (does adaptive routing help?)
>
> Requires Experiment 1 eval results to be available.

### Step 2a — Run GRASP-RAG classifier on all questions (subtask)

This is fast and independent of full query runs. Run it early.

```bash
python experiments/classify_questions.py --domain agriculture
python experiments/classify_questions.py --domain cs
python experiments/classify_questions.py --domain legal
python experiments/classify_questions.py --domain health
```

Output: `data/labels/{domain}_classified.json`

### Step 2b — Generate oracle ground-truth labels (GPT-4o)

Uses GPT-4o with explicit type definitions as the labeling oracle.

```bash
python experiments/label_questions.py --domain agriculture
python experiments/label_questions.py --domain cs
python experiments/label_questions.py --domain legal
python experiments/label_questions.py --domain health
```

Output: `data/labels/{domain}_gt_labels.json`

### Step 2c — Evaluate classifier accuracy + per-type win rate

Requires Exp 1 eval results (`data/results/{domain}_ultradomain_eval.json`).

```bash
python experiments/eval_classifier.py --domain agriculture
python experiments/eval_classifier.py --domain cs
python experiments/eval_classifier.py --domain legal
python experiments/eval_classifier.py --domain health
```

Output: Printed tables showing:
- Per-type classifier accuracy (GRASP-RAG auto-label vs GPT-4o oracle)
- Per-type win rate vs each baseline (grouped by oracle label)

---

## Quick Smoke Test

Run this first to verify the full pipeline before committing to long runs.

```bash
python experiments/run_queries.py --domain agriculture --ultradomain --methods grasp_rag naive_rag --limit 3
python experiments/evaluate.py --domain agriculture --ultradomain --baseline naive_rag
python experiments/classify_questions.py --domain agriculture
python experiments/eval_classifier.py --domain agriculture
```

---

## Results Location

| Experiment | Answers | Results | Labels |
|---|---|---|---|
| Exp 1 Win Rate | `data/answers/{domain}_ultradomain.jsonl` | `data/results/{domain}_ultradomain_eval.json` | — |
| Exp 2 Classifier | — | — | `data/labels/{domain}_classified.json` |
| Exp 2 Oracle Labels | — | — | `data/labels/{domain}_gt_labels.json` |

---

## Notes

- `run_queries.py` is **resumable** — safe to re-run if interrupted
- `classify_questions.py` and `label_questions.py` are **resumable**
- Each `evaluate.py` run re-judges everything — results are **not** resumable
- Win/Lose rates exclude Undefined outcomes from the denominator
- Clear LLM cache if prompts changed: `rm data/working_dir/*/kv_store_llm_response_cache.json`
