# GRASP-RAG Experiment 3 (MASHQA) — answer-quality external validation

Runbook for Experiment 3. Experiments 1 and 2 are documented in
`EXPERIMENTS.md` and are unchanged by this work.

---

## Why this experiment exists

**Experiment 3 is external validation**: repeat Experiment 1's methodology
(GRASP-RAG vs NaiveRAG/LightRAG/PathRAG, judged against a real reference
answer) on a dataset E1/E2 never touched, one with **no published or derived
question-type answer key**.

The only "question type" concept here is **GRASP-RAG's own classifier
prediction**, used purely as a reporting/stratification label, never
validated against any ground truth. Nothing is derived from the gold answer
text — that isn't available at real inference time.

This experiment measures **answer quality**: does GRASP-RAG's generated
answer, judged against MASHQA's real human-written answer as the reference
anchor, beat NaiveRAG / LightRAG / PathRAG the way it does on UltraDomain in
Experiment 1?

**Known limitation:** MASHQA articles are short (most are a single LightRAG
chunk). Almost any retrieval mechanism — GRASP-RAG's motifs, NaiveRAG's dense
search, LightRAG's hybrid mode, PathRAG's paths — tends to pull in most of
the same source article regardless of which node/chunk triggered it. Expect
win-rate gaps here to be smaller or noisier than E1's UltraDomain numbers;
that's a property of this corpus's chunk-to-article ratio, not necessarily of
the systems. Kept as a control variable (same chunking as E1/E2) rather than
tuned away. See `docs/EXPERIMENT_3_LIMITATIONS.md` for the full list.

---

## System roster and design decisions

- **Systems**: `grasp_rag` (adaptive) vs `naive_rag`, `lightrag`, `pathrag` —
  the same 4 systems as Experiment 1. No fixed-motif ablations.
- **Model**: `gpt-5.4-mini` for all generation (`grasp_rag/llm_config.py`'s
  `GENERATION_MODEL`, unmodified).
- **Judges**: Claude Haiku 4.5, Gemini 3.6 Flash, DeepSeek V4 Flash — the
  same 3-judge, OpenAI-excluded panel `evaluate.py`/E1 uses (OpenAI was
  dropped from judging after it was found to inflate GRASP-RAG's win rate
  via self-preference, per `judge_bias.py`).
- **Reference anchor**: MASHQA's own human-written `answer_text` per question
  (from `e3_prep.py`) — not a synthetic reference. Judged with
  `JUDGE_PROMPT_GT`, exactly E1's protocol: forward+reverse ordering, 2-of-3
  majority, Win/Lose/Undefined, no Tie.
- **"Question type"**: GRASP-RAG's own classifier prediction only (from
  `e3_classify.py`), used for a secondary per-type breakdown.
- **Sample size**: 1,000 stratified questions (capped 3/article, stratified
  by predicted type) from the 3,493-question MASHQA test-split pool. Each
  judged question costs ~4-6 generation calls + 18 judge calls (3 baselines
  x 2 orderings x 3 judges), so the full pool is not judged. Final sample by
  predicted type: single_entity 381, bridge 223, multi_entity 215, broad 181.
- **MASHQA train split — not used.** The test split (521 articles / 3,493
  questions) already far exceeds the judged sample.

---

## Data

MASHQA is public (https://github.com/mingzhu0527/MASHQA). **No MASHQA-derived
data is committed to this repo** — `data/e3_data/` is gitignored. Download the
MASHQA test-split JSON yourself and regenerate everything with the pipeline
below.

---

## Environment prerequisites

- **LightRAG** — clone into `references/LightRAG/LightRAG` and
  `pip install -e .`; the pinned commit is in
  `references/LightRAG/PINNED_COMMIT.txt`.
- **PathRAG** — clone `https://github.com/bupt-gamma/pathrag` into
  `references/PathRAG/PathRAG` at the commit in
  `references/PathRAG/PINNED_COMMIT.txt`. Not pip-installable (no
  `setup.py`/`pyproject.toml`) — imported via `sys.path` insertion, same
  mechanism as LightRAG.
- **`vllm` stub** (committed): PathRAG's `llm.py` unconditionally imports
  `vllm` for a local-model path this project never uses, and real `vllm` has
  no working build on Windows/CPU. `references/PathRAG/_vllm_stub/vllm/
  __init__.py` satisfies the import. Any script importing PathRAG must put
  `_vllm_stub` on `sys.path` *before* PathRAG's own path (see
  `e3_answer_quality.py`).
- **`openai`/`anthropic` pinned exactly** in `requirements.txt`
  (`openai==1.109.1` / `anthropic==0.125.0`): installing PathRAG's
  dependencies once silently upgraded both to untested major versions.
- `.env` with the API keys listed in `.env.example`.

---

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 0 — corpus prep | `experiments/e3_prep.py` | `data/e3_data/corpus.jsonl`, `questions.jsonl` |
| 1 — graph index | `experiments/index_corpus.py --mashqa` | `data/working_dir/mashqa/` |
| 2 — classify all questions | `experiments/e3_classify.py` | `data/e3_data/classified.jsonl` |
| 3 — stratified sample + 4-system generation + judging | `experiments/e3_answer_quality.py` | `data/e3_data/answer_quality.jsonl` |

### Stage 0 — corpus prep

```bash
python experiments/e3_prep.py path/to/mashqa_test.json --out data/e3_data
```

Produces 521 articles and 3,493 questions, each question carrying its
human-written `answer_text`.

### Stage 1 — graph index

```bash
python experiments/index_corpus.py --mashqa --data data/e3_data
```

Same LightRAG construction as E1/E2 (identical chunker, extraction prompt,
gleaning, merge, models), in its own working dir.

### Stage 2 — classify all questions

Cheap (VDB candidate fetch + one classifier call per question, no
generation) — run on the full pool before sampling, same rationale as E1/E2's
`classify_questions.py --domain X` step.

```bash
python experiments/e3_classify.py
```

Output: `{qid, question, classified_type, classified_anchors}`. Resumable.
Prints the predicted-type distribution on completion.

### Stage 3 — answer generation + judging

```bash
python experiments/e3_answer_quality.py --total-cap 600
```

For each sampled question: generates an answer from all 4 systems, then
judges GRASP-RAG pairwise against each baseline anchored on the real MASHQA
answer. Resumable — skips already-logged qids. Re-running draws the **next
non-overlapping batch** (already-judged qids are excluded and per-article
caps carry over), which is how the 1,000-question sample was built up.

Useful flags:
- `--limit N` — quick pilot (still respects `--cap-per-article`/`--cap-per-type`)
- `--question "exact text"` — run one specific question, bypassing sampling
- `--report-only` — re-print the result tables from the existing output file
  without spending any API calls
- `--concurrency` — default 2; each concurrent question costs ~22 LLM calls

`experiments/e3_inspect.py` is a debugging helper (not part of the pipeline)
for looking at what GRASP-RAG retrieved for a given question, node or chunk.

### Reporting

Printed at the end of a run, or via `--report-only`:

- **Table A — aggregate win rate.** GRASP-RAG's win rate (%) vs each of
  NaiveRAG/LightRAG/PathRAG, pooled across every judged (question x
  dimension) pair — directly comparable in shape to Experiment 1's headline
  table. Per-dimension detail tables (reusing `evaluate.py`'s
  `_print_dim_table`) print alongside it.
- **Table B — per-predicted-type win rate.** Same calculation, split by
  GRASP-RAG's own classifier-predicted type
  (single_entity/bridge/multi_entity/broad).

---

## Notes

- Do not modify E1/E2 graph construction, prompts, models, or
  `run_queries.py`/`evaluate.py` — this experiment's transfer claim and
  judge-panel validity depend on reusing them unchanged. All E3 code is
  additive (`e3_prep.py`, `e3_classify.py`, `e3_answer_quality.py`,
  `e3_inspect.py`, and the `--mashqa` entry point in `index_corpus.py`).
- If a run appears to hang with no new log output, it's safe to kill and
  rerun — both stages are resumable.
