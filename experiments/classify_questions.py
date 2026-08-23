"""
Experiment 2 subtask — Run GRASP-RAG classifier on all UltraDomain questions.

This is the FIRST step for Exp 2. Run it before label_questions.py and eval_classifier.py.

Reads:  data/questions/{domain}_ultradomain.json
Writes: data/labels/{domain}_classified.json  → {question: {classified_type, classified_anchors}}

Resumable: already-classified questions are skipped on re-run.

Usage:
    python experiments/classify_questions.py --domain agriculture
    python experiments/classify_questions.py --all-domains
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _ROOT)

from grasp_rag import GRASPRAG

DOMAINS = ["agriculture", "cs", "legal", "health"]


async def classify_domain(domain: str):
    wd = os.path.join(_ROOT, "data", "working_dir", domain)
    q_path = os.path.join(_ROOT, "data", "questions", f"{domain}_ultradomain.json")
    out_dir = os.path.join(_ROOT, "data", "labels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{domain}_classified.json")

    if not os.path.exists(wd):
        print(f"  Working dir not found: {wd}")
        print("  Run index_corpus.py first.")
        return
    if not os.path.exists(q_path):
        print(f"  Questions not found: {q_path}")
        print("  Run: python experiments/generate_questions.py --domain {domain} --mode ultradomain")
        return

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    # Load already-classified for resumability
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)
        print(f"  Resuming: {len(done)}/{len(questions)} already classified")

    todo = [item for item in questions if item["question"] not in done]
    if not todo:
        print(f"  All {len(questions)} questions already classified.")
        return

    grasp = GRASPRAG(working_dir=wd)
    await grasp.initialize()
    grasp._ensure_retriever()

    sem = asyncio.Semaphore(10)
    counter = [0]
    save_lock = asyncio.Lock()

    async def _classify_one(item: dict):
        q = item["question"]
        async with sem:
            candidates = await grasp._fetch_candidates(q)
            query_type, anchors = await grasp._classifier.classify(q, candidates=candidates)
            anchors = await grasp._resolve_anchors(anchors)
        async with save_lock:
            done[q] = {"classified_type": query_type, "classified_anchors": anchors}
            counter[0] += 1
            print(f"  [{counter[0]+len(questions)-len(todo)}/{len(questions)}] {q[:60]}... → {query_type}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(done, f, indent=2, ensure_ascii=False)

    await asyncio.gather(*[_classify_one(item) for item in todo])

    await grasp.finalize()
    print(f"\n  Done — {len(done)} questions classified → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--all-domains", action="store_true",
                        help="Run all four domains sequentially")
    args = parser.parse_args()

    domains = DOMAINS if args.all_domains else [args.domain]
    for domain in domains:
        print(f"\n{'='*55}\n  Domain: {domain.upper()}\n{'='*55}")
        asyncio.run(classify_domain(domain))


if __name__ == "__main__":
    main()
