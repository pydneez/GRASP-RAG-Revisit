"""
Step 3: Run all methods on the generated questions and collect answers.

Methods:
  - naive_rag   : LightRAG mode="naive"  (chunking + vector search)
  - lightrag    : LightRAG mode="hybrid" (dual-level k-hop)
  - grasp_rag   : GRASP-RAG (query-adaptive motif)

All methods share the same indexed graph (same working_dir).
Results are saved incrementally so the script can be resumed if interrupted.

Usage:
    python experiments/run_queries.py --domain agriculture
    python experiments/run_queries.py --domain agriculture --methods naive_rag lightrag grasp_rag
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
_PATHRAG_PATH = os.path.join(_ROOT, "references", "PathRAG", "PathRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _PATHRAG_PATH)
sys.path.insert(0, _ROOT)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed

from grasp_rag import GRASPRAG
from grasp_rag.llm_config import GENERATION_MODEL, generation_complete, pathrag_generation_complete

DOMAINS = ["agriculture", "cs", "legal", "health"]
FIXED_MOTIF_METHODS = ["grasp_rag_fixed_ego", "grasp_rag_fixed_bridge",
                        "grasp_rag_fixed_steiner", "grasp_rag_fixed_broad"]
FIXED_MOTIF_TYPES = {
    "grasp_rag_fixed_ego": "single_entity",
    "grasp_rag_fixed_bridge": "bridge",
    "grasp_rag_fixed_steiner": "multi_entity",
    "grasp_rag_fixed_broad": "broad",
}
ALL_METHODS = ["naive_rag", "lightrag", "grasp_rag", "grasp_rag_nocascade",
               "grasp_rag_flat_context", "grasp_rag_oracle", "pathrag",
               *FIXED_MOTIF_METHODS]


def working_dir(domain: str) -> str:
    return os.path.join(_ROOT, "data", "working_dir", domain)


def questions_path(domain: str, typed: bool = False, bridge_gt: bool = False,
                   ultradomain: bool = False, typed_balanced: bool = False) -> str:
    if bridge_gt:
        suffix = "_bridge_gt"
    elif typed_balanced:
        suffix = "_typed_balanced"
    elif typed:
        suffix = "_typed"
    elif ultradomain:
        suffix = "_ultradomain"
    else:
        suffix = ""
    return os.path.join(_ROOT, "data", "questions", f"{domain}{suffix}.json")


def answers_path(domain: str, typed: bool = False, bridge_gt: bool = False,
                 ultradomain: bool = False, typed_balanced: bool = False) -> str:
    path = os.path.join(_ROOT, "data", "answers")
    os.makedirs(path, exist_ok=True)
    if bridge_gt:
        suffix = "_bridge_gt"
    elif typed_balanced:
        suffix = "_typed_balanced"
    elif typed:
        suffix = "_typed"
    elif ultradomain:
        suffix = "_ultradomain"
    else:
        suffix = ""
    return os.path.join(path, f"{domain}{suffix}.jsonl")


async def run_lightrag_method(rag: LightRAG, question: str, mode: str) -> str:
    try:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


async def run_grasp(grasp: GRASPRAG, question: str) -> str:
    try:
        result = await grasp.aquery(question)
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


async def run_pathrag(pathrag, question: str) -> str:
    try:
        from PathRAG import QueryParam as PRQueryParam
        result = await pathrag.aquery(question, param=PRQueryParam(mode="hybrid"))
        return result if isinstance(result, str) else f"ERROR: query returned {type(result).__name__}"
    except Exception as e:
        return f"ERROR: {e}"


async def run(domain: str, methods: list[str], typed: bool = False,
              bridge_gt: bool = False, ultradomain: bool = False,
              typed_balanced: bool = False,
              limit: int | None = None, sample_per_type: int | None = None,
              concurrency: int = 1):
    wd = working_dir(domain)
    if not os.path.exists(wd):
        print(f"Working dir not found: {wd}")
        print("Run index_corpus.py first.")
        sys.exit(1)

    q_path = questions_path(domain, typed=typed, bridge_gt=bridge_gt,
                            ultradomain=ultradomain, typed_balanced=typed_balanced)
    if not os.path.exists(q_path):
        print(f"Questions not found: {q_path}")
        hint = "generate_bridge_gt.py" if bridge_gt else "generate_questions.py"
        print(f"Run {hint} first.")
        sys.exit(1)

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    if sample_per_type:
        from collections import defaultdict
        by_type: dict[str, list] = defaultdict(list)
        for q in questions:
            by_type[q.get("query_type", "unknown")].append(q)
        questions = [q for qs in by_type.values() for q in qs[:sample_per_type]]
    print(f"Loaded {len(questions)} questions for domain={domain} "
          f"(typed={typed}, typed_balanced={typed_balanced})")

    # Load already-completed answers for resumability
    out_path = answers_path(domain, typed=typed, bridge_gt=bridge_gt,
                            ultradomain=ultradomain, typed_balanced=typed_balanced)
    done: set[tuple[str, str]] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["question"], r["method"]))
        print(f"Resuming: {len(done)} answers already saved")

    # Initialize LightRAG (shared instance for naive_rag and lightrag modes)
    rag = LightRAG(
        working_dir=wd,
        llm_model_func=generation_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    # Initialize GRASP-RAG variants (share same working_dir → same graph)
    grasp = None
    if "grasp_rag" in methods:
        grasp = GRASPRAG(working_dir=wd, cascade=True)
        await grasp.initialize()

    grasp_nocascade = None
    if "grasp_rag_nocascade" in methods:
        grasp_nocascade = GRASPRAG(working_dir=wd, cascade=False)
        await grasp_nocascade.initialize()

    grasp_flat = None
    if "grasp_rag_flat_context" in methods:
        grasp_flat = GRASPRAG(working_dir=wd, cascade=True, flat_context=True)
        await grasp_flat.initialize()

    grasp_oracle = None
    if "grasp_rag_oracle" in methods:
        oracle_path = os.path.join(_ROOT, "data", "labels", f"{domain}_gt_labels.json")
        if not os.path.exists(oracle_path):
            print(f"Oracle labels not found: {oracle_path}")
            print("Run experiments/label_questions.py first.")
            sys.exit(1)
        with open(oracle_path, encoding="utf-8") as f:
            oracle_labels = json.load(f)
        grasp_oracle = GRASPRAG(working_dir=wd, cascade=True, oracle_labels=oracle_labels)
        await grasp_oracle.initialize()
        print(f"Loaded {len(oracle_labels)} oracle labels for {domain}")

    # Initialize Fixed-motif GRASP-RAG variants — same graph, routing pinned to
    # one motif regardless of classifier output (headline Adaptive-vs-Fixed ablation).
    grasp_fixed: dict[str, GRASPRAG] = {}
    for method_name in FIXED_MOTIF_METHODS:
        if method_name in methods:
            g = GRASPRAG(working_dir=wd, force_motif=FIXED_MOTIF_TYPES[method_name])
            await g.initialize()
            grasp_fixed[method_name] = g

    # Initialize PathRAG (shares same working_dir → same graph)
    pathrag_inst = None
    if "pathrag" in methods:
        from PathRAG import PathRAG as _PathRAG
        from PathRAG.llm import openai_embedding as pr_embed
        from PathRAG.utils import EmbeddingFunc
        pathrag_inst = _PathRAG(
            working_dir=wd,
            llm_model_func=pathrag_generation_complete,
            llm_model_name=GENERATION_MODEL,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=pr_embed,
            ),
        )

    sem        = asyncio.Semaphore(concurrency)
    grasp_lock = asyncio.Lock()   # _last_classification is a shared attr — serialize grasp calls
    write_lock = asyncio.Lock()   # prevent interleaved writes to the output file

    async def _process_question(i: int, item: dict, out_f):
        q = item["question"]
        async with sem:
            all_cached = all((q, m) in done for m in methods)
            if all_cached:
                print(f"[{i+1}/{len(questions)}] (all cached) {q[:60]}...")
                return

            print(f"\n[{i+1}/{len(questions)}] {q[:80]}...")

            for method in methods:
                if (q, method) in done:
                    print(f"  {method}: (cached)")
                    continue

                print(f"  {method}: ...", end="", flush=True)

                classification = None
                if method == "naive_rag":
                    answer = await run_lightrag_method(rag, q, mode="naive")
                elif method == "lightrag":
                    answer = await run_lightrag_method(rag, q, mode="hybrid")
                elif method == "grasp_rag":
                    async with grasp_lock:
                        answer = await run_grasp(grasp, q)
                        classification = grasp._last_classification
                elif method == "grasp_rag_nocascade":
                    async with grasp_lock:
                        answer = await run_grasp(grasp_nocascade, q)
                        classification = grasp_nocascade._last_classification
                elif method == "grasp_rag_flat_context":
                    async with grasp_lock:
                        answer = await run_grasp(grasp_flat, q)
                        classification = grasp_flat._last_classification
                elif method == "grasp_rag_oracle":
                    async with grasp_lock:
                        answer = await run_grasp(grasp_oracle, q)
                        classification = grasp_oracle._last_classification
                elif method == "pathrag":
                    answer = await run_pathrag(pathrag_inst, q)
                elif method in FIXED_MOTIF_METHODS:
                    async with grasp_lock:
                        g = grasp_fixed[method]
                        answer = await run_grasp(g, q)
                        classification = g._last_classification
                else:
                    continue

                print(f" done ({len(answer)} chars)")
                record = {
                    "question": q,
                    "method": method,
                    "answer": answer,
                    "domain": domain,
                    "query_type": item.get("query_type"),
                    "persona": item.get("persona"),
                    "task": item.get("task"),
                    "classified_type": classification.get("query_type") if classification else None,
                    "classified_anchors": classification.get("anchors") if classification else None,
                }
                async with write_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

    with open(out_path, "a", encoding="utf-8") as out_f:
        await asyncio.gather(*[
            _process_question(i, item, out_f)
            for i, item in enumerate(questions)
        ])

    await rag.finalize_storages()
    if grasp:
        await grasp.finalize()
    if grasp_nocascade:
        await grasp_nocascade.finalize()
    if grasp_flat:
        await grasp_flat.finalize()
    if grasp_oracle:
        await grasp_oracle.finalize()
    for g in grasp_fixed.values():
        await g.finalize()

    print(f"\nAll answers saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--typed", action="store_true",
                        help="Use typed question set (*_typed.json / *_typed.jsonl)")
    parser.add_argument("--ultradomain", action="store_true",
                        help="Use UltraDomain question set (*_ultradomain.json / *_ultradomain.jsonl)")
    parser.add_argument("--typed-balanced", action="store_true",
                        help="Use Part B balanced typed question set "
                             "(*_typed_balanced.json / *_typed_balanced.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (for quick testing)")
    parser.add_argument("--sample-per-type", type=int, default=None,
                        help="Sample N questions from each query_type (e.g. 5 = 20 total)")
    parser.add_argument("--bridge-gt", action="store_true",
                        help="Use bridge ground-truth question set (*_bridge_gt.json)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of questions to process in parallel (default: 1). "
                             "Try 3–5. GRASP-RAG calls are serialized internally to avoid "
                             "race conditions on classification.")
    args = parser.parse_args()
    asyncio.run(run(args.domain, args.methods, typed=args.typed,
                    bridge_gt=args.bridge_gt, ultradomain=args.ultradomain,
                    typed_balanced=args.typed_balanced,
                    limit=args.limit, sample_per_type=args.sample_per_type,
                    concurrency=args.concurrency))


if __name__ == "__main__":
    main()
