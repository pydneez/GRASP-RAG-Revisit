"""
Experiment 3 (redone) — Stage 2: answer generation + reference-anchored judging.

External validation of E1's win-rate methodology on MASHQA — a dataset E1/E2
never touched, and one with no published question-type answer key (see
EXPERIMENT_3.md). GRASP-RAG (adaptive) vs NaiveRAG / LightRAG / PathRAG,
generated over the *same* already-built MASHQA graph
(data/working_dir/mashqa/), judged against MASHQA's own real, human-written
answer (question["answer_text"]) as the reference anchor — exactly E1's
protocol (JUDGE_PROMPT_GT, 3 independent judges, forward+reverse ordering,
2-of-3 majority, Win/Lose/Undefined, no Tie), just with a real answer instead
of an UltraDomain-style synthetic reference.

No retrieval-structure measurement here — an earlier retrieval-structure
pilot on MASHQA was discarded and is not part of this pipeline. The
only "question type" concept used here is GRASP-RAG's own classifier
prediction (from e3_classify.py), used purely as a reporting/stratification
label — never validated against any ground truth in this script.

Systems: grasp_rag (adaptive), naive_rag (LightRAG naive mode), lightrag
(LightRAG hybrid mode), pathrag. No fixed-motif ablations — this redo isn't
re-testing routing-vs-no-routing;
it's testing GRASP-RAG-as-shipped against the three external baselines E1
already compares against, replicated here on a new dataset.

Pipeline per sampled question:
  1. Generate an answer from all 4 systems (grasp_rag's classifier call is
     logged again here via _last_classification — independent of, and not
     reused from, e3_classify.py's earlier pass; that earlier pass only
     existed to let sampling stratify by predicted type before spending any
     generation/judging budget).
  2. Judge GRASP-RAG pairwise against each of the 3 baselines, anchored on
     the real gold answer_text (JUDGE_PROMPT_GT) — 3 comparisons x 2
     orderings x 3 judges = 18 judge calls/question, on top of ~4-6
     generation calls/question. This is why sampling (not all 3,493) is
     used for this stage — see pick_sample()/EXPERIMENT_3.md's cost
     rationale.

Reuses evaluate.py's multi_judge()/_print_dim_table() exactly as E1 does.
Does not modify evaluate.py, run_queries.py, or any E1/E2 file.

Usage:
    python experiments/e3_answer_quality.py --limit 20             # quick pilot
    python experiments/e3_answer_quality.py --total-cap 600        # full scale
    python experiments/e3_answer_quality.py --report-only          # re-print
        tables from the existing --out file without spending any API calls
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

# evaluate.py's _print_dim_table prints a Unicode warning glyph (U+26A0) for
# low-n dimensions; Windows' default console codepage (cp1252) can't encode
# it and raises UnicodeEncodeError. Fixed here rather than in evaluate.py,
# which this script doesn't modify.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
_PATHRAG_VLLM_STUB = os.path.join(_ROOT, "references", "PathRAG", "_vllm_stub")
_PATHRAG_PATH = os.path.join(_ROOT, "references", "PathRAG", "PathRAG")
sys.path.insert(0, _PATHRAG_VLLM_STUB)  # must precede _PATHRAG_PATH — see its __init__.py
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _PATHRAG_PATH)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from lightrag import LightRAG
from lightrag.llm.openai import openai_embed

from grasp_rag import GRASPRAG
from grasp_rag.llm_config import generation_complete, pathrag_generation_complete
from evaluate import DIMENSIONS, JUDGE_PROMPT_GT, _LOW_N_THRESHOLD, _print_dim_table, multi_judge

BASELINES = ["naive_rag", "lightrag", "pathrag"]
CLASSIFIER_TYPES = ["single_entity", "bridge", "multi_entity", "broad"]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pick_sample(questions, classified, cap_per_article, cap_per_type, total_cap, seed,
                exclude_qids=None, prior_article_counts=None):
    """Stratified sample by GRASP-RAG's *predicted* type (from e3_classify.py),
    capped per source article so no single long article dominates a bucket.
    Stratifies on the classifier's own prediction, not anything derived from
    the gold answer — see the module docstring and EXPERIMENT_3.md for why.

    total_cap/cap_per_type describe the size of THIS batch, not a cumulative
    grand total — pass exclude_qids (already-judged qids from a prior run's
    --out) to draw a fresh, non-overlapping next batch. Do not try to get a
    bigger cumulative sample by just re-running with a larger total_cap
    instead: cap_per_article's bookkeeping is shared across types in a single
    pass, so changing the target size can shift which specific questions get
    accepted partway through — verified empirically to drop up to ~17% of a
    previous batch rather than being a strict superset. prior_article_counts
    (doc_id -> count already used across prior batches) keeps cap_per_article
    a real cumulative cap across repeated runs, not just a per-batch one."""
    rng = random.Random(seed)
    exclude_qids = exclude_qids or set()
    by_type: dict[str, list] = {}
    for q in questions:
        if q["qid"] in exclude_qids:
            continue
        c = classified.get(q["qid"])
        if c is None:
            continue
        by_type.setdefault(c["classified_type"], []).append(q)

    sample, seen_qid, per_article_count = [], set(), dict(prior_article_counts or {})

    def _try_add(q):
        if q["qid"] in seen_qid or len(sample) >= total_cap:
            return False
        doc_id = q.get("doc_id")
        if per_article_count.get(doc_id, 0) >= cap_per_article:
            return False
        sample.append(q)
        seen_qid.add(q["qid"])
        per_article_count[doc_id] = per_article_count.get(doc_id, 0) + 1
        return True

    # Scarcest type first. cap_per_article is a budget shared across all types
    # (and across prior batches), so whichever type is processed first gets the
    # pick of the articles that still have headroom. Processing in a fixed
    # order starved the rarest type: measured on the second MASHQA batch, the
    # three abundant types consumed the article headroom and left `broad` able
    # to draw only 36 of the 111 candidates that were otherwise reachable.
    type_order = sorted(CLASSIFIER_TYPES, key=lambda t: len(by_type.get(t, [])))

    remaining_types = len(type_order)
    remaining_budget = total_cap
    for t in type_order:
        effective_cap = min(cap_per_type, max(1, remaining_budget // remaining_types))
        qs = list(by_type.get(t, []))
        rng.shuffle(qs)
        added = 0
        for q in qs:
            if added >= effective_cap:
                break
            if _try_add(q):
                added += 1
        remaining_types -= 1
        remaining_budget = total_cap - len(sample)

    # Top-up pass: an even per-type split under-delivers whenever some type's
    # pool (or its remaining article headroom) can't fill its share, and the
    # first loop never revisits a type to reclaim that leftover. Fill any
    # unused budget from whatever candidates remain, so total_cap is actually
    # honoured. Balance is the goal of the first pass; this pass trades
    # balance for hitting the requested n once balance is no longer possible.
    if len(sample) < total_cap:
        leftovers = [q for t in type_order for q in by_type.get(t, [])]
        rng.shuffle(leftovers)
        for q in leftovers:
            if len(sample) >= total_cap:
                break
            _try_add(q)   # already guards duplicates, article cap, total_cap
    return sample


async def run_lightrag_mode(rag: LightRAG, question: str, mode: str) -> str:
    from lightrag import QueryParam
    try:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


async def run_pathrag(pathrag_inst, question: str) -> str:
    try:
        from PathRAG import QueryParam as PRQueryParam
        result = await pathrag_inst.aquery(question, param=PRQueryParam(mode="hybrid"))
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


async def run_grasp(grasp: GRASPRAG, question: str) -> str:
    try:
        result = await grasp.aquery(question)
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


def dim_counts_from_records(records, baseline):
    """{dim: {"Win": n, "Loss": n, "CannotEvaluate": n, "PositionBias": n, "NoMajority": n}}
    for one baseline, across the given records. "Win" = grasp_rag beat this
    baseline (judged against the real gold answer as anchor)."""
    counts = {d: {"Win": 0, "Loss": 0, "CannotEvaluate": 0, "PositionBias": 0, "NoMajority": 0}
              for d in DIMENSIONS}
    for r in records:
        outcome = r["outcomes"].get(baseline)
        if not outcome:
            continue
        for d in DIMENSIONS:
            v = outcome.get(d)
            if v == "Answer 1":
                counts[d]["Win"] += 1
            elif v == "Answer 2":
                counts[d]["Loss"] += 1
            elif v in counts[d]:
                counts[d][v] += 1
    return counts


def aggregate_win_rate(records, baseline):
    """Single headline number: wins / (wins+losses), pooled across all
    dimensions and questions — matches paper Table 1's win-rate definition."""
    wins = losses = 0
    for r in records:
        outcome = r["outcomes"].get(baseline)
        if not outcome:
            continue
        for d in DIMENSIONS:
            v = outcome.get(d)
            if v == "Answer 1":
                wins += 1
            elif v == "Answer 2":
                losses += 1
    decided = wins + losses
    return (100 * wins / decided if decided else None), decided


def print_report(records):
    print(f"\n{'='*70}\nTABLE A — Aggregate win rate (GRASP-RAG vs each baseline, {len(records)} questions)\n{'='*70}")
    print(f"  {'Baseline':<12} {'Win%':>7} {'n (decided)':>12}")
    for baseline in BASELINES:
        wr, n = aggregate_win_rate(records, baseline)
        wr_str = f"{wr:.1f}%" if wr is not None else "n/a"
        print(f"  {baseline:<12} {wr_str:>7} {n:>12}")

    print(f"\nPer-dimension detail:")
    for baseline in BASELINES:
        print(f"\n--- GRASP-RAG vs {baseline} ---")
        _print_dim_table(dim_counts_from_records(records, baseline))

    print(f"\n{'='*70}\nTABLE B — Per-predicted-type win rate (classifier's own guess, no ground truth used)\n{'='*70}")
    for t in CLASSIFIER_TYPES:
        subset = [r for r in records if r["classified_type"] == t]
        if not subset:
            continue
        print(f"\n=== predicted type: {t} (n={len(subset)}) ===")
        print(f"  {'Baseline':<12} {'Win%':>7} {'n (decided)':>12}")
        for baseline in BASELINES:
            wr, n = aggregate_win_rate(subset, baseline)
            wr_str = f"{wr:.1f}%" if wr is not None else "n/a"
            print(f"  {baseline:<12} {wr_str:>7} {n:>12}")
    print(f"\n(dimensions/rows with fewer than {_LOW_N_THRESHOLD} decided questions "
          f"are flagged unreliable in the per-dimension tables above, per evaluate.py's convention)")


async def main(a):
    questions = load_jsonl(a.questions)
    classified = {r["qid"]: r for r in load_jsonl(a.classified)}
    if not questions:
        print(f"No questions found at {a.questions}"); sys.exit(1)
    if not classified:
        print(f"No classifications found at {a.classified}")
        print("Run experiments/e3_classify.py first.")
        sys.exit(1)

    done = {r["qid"] for r in load_jsonl(a.out)}
    if a.report_only:
        records = load_jsonl(a.out)
        if not records:
            print(f"Nothing to report — {a.out} is empty or missing.")
            sys.exit(1)
        print_report(records)
        return

    if a.question:
        match = next((q for q in questions if q["question"] == a.question), None)
        if match is None:
            print(f"No exact match for: {a.question!r}")
            return
        sample = [match]
    elif a.all:
        # Bypass stratified sampling entirely — pick_sample()'s per-type
        # budget split is designed to cap a large pool DOWN to a small
        # budget (its fixed processing order never reclaims a smaller
        # type's leftover share for an earlier, larger one), so it
        # under-fills rather than including everything when total_cap
        # approaches the pool size. Stratification is meaningless anyway
        # once you're taking the whole pool.
        sample = [q for q in questions if q["qid"] in classified]
    else:
        # Excluding already-judged qids (and seeding per-article counts from
        # them) makes re-running this same command repeatedly fetch the next
        # non-overlapping batch, rather than risk silently reshuffling a
        # previous batch's membership — see pick_sample()'s docstring.
        qid_to_doc = {q["qid"]: q.get("doc_id") for q in questions}
        prior_article_counts: dict[str, int] = {}
        for qid in done:
            doc_id = qid_to_doc.get(qid)
            if doc_id:
                prior_article_counts[doc_id] = prior_article_counts.get(doc_id, 0) + 1
        sample = pick_sample(questions, classified, a.cap_per_article, a.cap_per_type,
                             a.limit or a.total_cap, a.seed,
                             exclude_qids=done, prior_article_counts=prior_article_counts)
        if len(sample) < (a.limit or a.total_cap):
            print(f"(only {len(sample)} new questions available — remaining pool "
                  f"is smaller than the requested batch size)")
    print(f"{len(sample)} questions selected")

    if not os.path.exists(a.working_dir):
        print(f"Working dir not found: {a.working_dir}")
        print("Run: python experiments/index_corpus.py --mashqa")
        sys.exit(1)

    rag = LightRAG(working_dir=a.working_dir, llm_model_func=generation_complete,
                   embedding_func=openai_embed)
    await rag.initialize_storages()

    grasp = GRASPRAG(working_dir=a.working_dir, cascade=True)
    await grasp.initialize()

    from PathRAG import PathRAG as _PathRAG
    from PathRAG.llm import openai_embedding as pr_embed
    from PathRAG.utils import EmbeddingFunc
    from grasp_rag.llm_config import GENERATION_MODEL
    pathrag_inst = _PathRAG(
        working_dir=a.working_dir,
        llm_model_func=pathrag_generation_complete,
        llm_model_name=GENERATION_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=pr_embed),
    )

    sem = asyncio.Semaphore(a.concurrency)
    grasp_lock = asyncio.Lock()  # _last_classification is a shared attr — serialize grasp calls, same as run_queries.py
    write_lock = asyncio.Lock()
    records = [r for r in load_jsonl(a.out)]

    async def _run_one(q, i, total, out_f):
        if q["qid"] in done:
            return
        async with sem:
            print(f"[{i+1}/{total}] {q['question']}")
            async with grasp_lock:
                grasp_ans = await run_grasp(grasp, q["question"])
                classification = grasp._last_classification or {}
            naive_ans, lightrag_ans, pathrag_ans = await asyncio.gather(
                run_lightrag_mode(rag, q["question"], "naive"),
                run_lightrag_mode(rag, q["question"], "hybrid"),
                run_pathrag(pathrag_inst, q["question"]),
            )
            gold = q["answer_text"]
            outcomes, per_judge = {}, {}
            for baseline, base_ans in [("naive_rag", naive_ans), ("lightrag", lightrag_ans),
                                       ("pathrag", pathrag_ans)]:
                outcome, pj = await multi_judge(
                    q["question"], grasp_ans, base_ans,
                    prompt_template=JUDGE_PROMPT_GT, reference=gold)
                outcomes[baseline] = outcome
                per_judge[baseline] = pj

        record = {
            "qid": q["qid"], "question": q["question"],
            "classified_type": classification.get("query_type"),
            "classified_anchors": classification.get("anchors"),
            "reference": gold,
            "answers": {"grasp_rag": grasp_ans, "naive_rag": naive_ans,
                        "lightrag": lightrag_ans, "pathrag": pathrag_ans},
            "outcomes": outcomes, "per_judge": per_judge,
        }
        async with write_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
        records.append(record)
        print(f"  outcomes: { {b: outcomes[b] for b in BASELINES} }")

    with open(a.out, "a", encoding="utf-8") as out_f:
        await asyncio.gather(*[
            _run_one(q, i, len(sample), out_f) for i, q in enumerate(sample)
        ])

    await rag.finalize_storages()
    await grasp.finalize()

    print_report(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=os.path.join(_ROOT, "data", "e3_data", "questions.jsonl"))
    p.add_argument("--classified", default=os.path.join(_ROOT, "data", "e3_data", "classified.jsonl"))
    p.add_argument("--working-dir", default=os.path.join(_ROOT, "data", "working_dir", "mashqa"))
    p.add_argument("--out", default=os.path.join(_ROOT, "data", "e3_data", "answer_quality.jsonl"))
    p.add_argument("--question", default=None,
                    help="Run exactly one question, by exact text match. Bypasses sampling.")
    p.add_argument("--limit", type=int, default=None,
                    help="Quick-pilot mode: total question cap, cap-per-article/type still apply. Omit for full scale.")
    p.add_argument("--cap-per-article", type=int, default=3)
    p.add_argument("--cap-per-type", type=int, default=150)
    p.add_argument("--total-cap", type=int, default=600)
    p.add_argument("--all", action="store_true",
                    help="Run every classified question (all 3,493), bypassing stratified "
                         "sampling and the caps above entirely. Real cost: ~22 LLM calls/"
                         "question, ~77,000 total — see EXPERIMENT_3.md's sample-size "
                         "rationale before using this.")
    p.add_argument("--concurrency", type=int, default=2,
                    help="Concurrent questions in flight — each makes ~4-6 generation calls + "
                         "18 judge calls (3 baselines x 2 orderings x 3 judges). Keep modest.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-only", action="store_true",
                    help="Skip generation/judging entirely — just reprint TABLE A/B from --out.")
    asyncio.run(main(p.parse_args()))
