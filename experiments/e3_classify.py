"""
Experiment 3 (redone) — Stage 1: classify every MASHQA question.

Mirrors classify_questions.py's E1/E2 pattern (candidate fetch + one classifier
call, no retrieval/generation) applied to the MASHQA question set instead of
UltraDomain. This is deliberately the *only* place a "question type" is
established for the new E3 design: the classifier's own predicted
query_type, nothing derived from the answer text. See EXPERIMENT_3.md for why
— a type inferred from the gold answer isn't available at real inference
time, so it was ruled out for this redo.

Cheap and independent of the full 4-system run: run it once on everything
before sampling, so `e3_answer_quality.py` can stratify by predicted type
without needing to classify the whole pool inline.

Reads:  data/e3_data/questions.jsonl
Writes: data/e3_data/classified.jsonl  → {qid, question, classified_type, classified_anchors}

Resumable: already-classified qids are skipped on re-run.

Usage:
    python experiments/e3_classify.py
    python experiments/e3_classify.py --limit 50       # quick check
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _ROOT)

from grasp_rag import GRASPRAG


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


async def main(a):
    questions = load_jsonl(a.questions)
    if not questions:
        print(f"No questions found at {a.questions}")
        print("Run experiments/e3_prep.py first.")
        sys.exit(1)
    if a.limit:
        questions = questions[: a.limit]

    if not os.path.exists(a.working_dir):
        print(f"Working dir not found: {a.working_dir}")
        print("Run: python experiments/index_corpus.py --mashqa")
        sys.exit(1)

    done: set[str] = set()
    if os.path.exists(a.out):
        for r in load_jsonl(a.out):
            done.add(r["qid"])
        print(f"Resuming: {len(done)}/{len(questions)} already classified")

    todo = [q for q in questions if q["qid"] not in done]
    if not todo:
        print(f"All {len(questions)} questions already classified.")
        return

    grasp = GRASPRAG(working_dir=a.working_dir)
    await grasp.initialize()
    grasp._ensure_retriever()

    sem = asyncio.Semaphore(a.concurrency)
    write_lock = asyncio.Lock()
    counter = [0]

    async def _classify_one(item, out_f):
        q = item["question"]
        async with sem:
            candidates = await grasp._fetch_candidates(q)
            query_type, anchors = await grasp._classifier.classify(q, candidates=candidates)
            anchors = await grasp._resolve_anchors(anchors)
        record = {
            "qid": item["qid"],
            "question": q,
            "classified_type": query_type,
            "classified_anchors": anchors,
        }
        async with write_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            counter[0] += 1
            print(f"  [{counter[0] + len(done)}/{len(questions)}] {q[:60]}... -> {query_type}")

    with open(a.out, "a", encoding="utf-8") as out_f:
        await asyncio.gather(*[_classify_one(item, out_f) for item in todo])

    await grasp.finalize()

    all_done = load_jsonl(a.out)
    from collections import Counter
    dist = Counter(r["classified_type"] for r in all_done)
    print(f"\nDone — {len(all_done)} questions classified -> {a.out}")
    print("Predicted-type distribution:")
    for t, n in dist.most_common():
        print(f"  {t:<15} {n:>5}  ({100*n/len(all_done):.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=os.path.join(_ROOT, "data", "e3_data", "questions.jsonl"))
    p.add_argument("--working-dir", default=os.path.join(_ROOT, "data", "working_dir", "mashqa"))
    p.add_argument("--out", default=os.path.join(_ROOT, "data", "e3_data", "classified.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10,
                    help="Classifier-only calls are cheap (no generation) — safe to run higher than the answer-quality script.")
    asyncio.run(main(p.parse_args()))
