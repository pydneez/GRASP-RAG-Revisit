"""
Capture one question's full prompt + answer for all four methods.

Produces data/report/example_{domain}.json with:
  question, query_type, and for each method:
    - retrieval_summary  (what was retrieved)
    - prompt             (exact text sent to the LLM)
    - answer             (LLM response)

Usage:
    python experiments/capture_example.py --domain agriculture --question "What is the relationship between Phosphorus and Beekeeping Industry?"
    python experiments/capture_example.py --domain agriculture --index 0 --typed
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
_PATHRAG_PATH  = os.path.join(_ROOT, "references", "PathRAG", "PathRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _PATHRAG_PATH)
sys.path.insert(0, _ROOT)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from grasp_rag import GRASPRAG
from grasp_rag.grasp_rag import _RESPONSE_PROMPT, _UNCERTAINTY_THRESHOLD


async def capture_naive_rag(rag: LightRAG, question: str) -> dict:
    prompt = await rag.aquery(question, param=QueryParam(mode="naive", only_need_prompt=True))
    answer = await rag.aquery(question, param=QueryParam(mode="naive"))
    return {
        "retrieval_summary": "Vector search over raw text chunks (no graph). Top-k chunks by cosine similarity are concatenated as context.",
        "prompt": prompt,
        "answer": answer,
    }


async def capture_lightrag(rag: LightRAG, question: str) -> dict:
    prompt = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_prompt=True))
    answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    return {
        "retrieval_summary": "Hybrid mode: combines local (entity ego-network) and global (community summaries) graph retrieval, plus vector chunks.",
        "prompt": prompt,
        "answer": answer,
    }


async def capture_pathrag(pathrag, question: str) -> dict:
    from PathRAG import QueryParam as PRQueryParam
    from PathRAG.prompt import PROMPTS

    # only_need_prompt is checked AFTER the LLM cache in PathRAG, so when an answer
    # is already cached it returns the cached answer instead of the raw prompt.
    # Fix: get context separately via only_need_context (cache-safe path), then
    # format the prompt template manually so we see the actual prompt sent to the LLM.
    context = await pathrag.aquery(question, param=PRQueryParam(mode="hybrid", only_need_context=True))
    response_type = PRQueryParam().response_type  # default: "Multiple Paragraphs"
    prompt = PROMPTS["rag_response"].format(context_data=context, response_type=response_type)

    answer = await pathrag.aquery(question, param=PRQueryParam(mode="hybrid"))
    return {
        "retrieval_summary": "Path-based retrieval: finds top-k entity pairs, extracts graph paths between them, ranks paths by relevance, uses path chains as context.",
        "prompt": prompt,
        "answer": answer,
    }


async def capture_grasp_rag(grasp: GRASPRAG, question: str) -> dict:
    grasp._ensure_retriever()

    candidates = await grasp._fetch_candidates(question)
    query_type, anchors = await grasp._classifier.classify(question, candidates=candidates)
    paths = grasp._retriever.retrieve_direct(query_type, anchors)
    confidence = grasp._retriever.uncertainty(paths)

    fallback_used = False
    if query_type == "broad" or confidence < _UNCERTAINTY_THRESHOLD or not paths:
        fallback_used = True
        context = "(fell back to LightRAG hybrid — no confident motif match)"
        prompt = "(LightRAG hybrid fallback)"
        answer = await grasp._rag.aquery(question, param=QueryParam(mode="hybrid"))
    else:
        context = await grasp._build_structured_context(paths, query_type, question)
        prompt = _RESPONSE_PROMPT.format(context=context, query=question)
        answer = await grasp._rag.llm_model_func(prompt)

    retrieval_summary = (
        f"Motif classified as: {query_type}\n"
        f"Anchors identified: {anchors}\n"
        f"Paths retrieved: {len(paths)} path(s)\n"
        f"Confidence: {confidence:.3f}\n"
        f"Fallback used: {fallback_used}"
    )

    return {
        "retrieval_summary": retrieval_summary,
        "context": context if not fallback_used else None,
        "prompt": prompt,
        "answer": answer,
    }


async def run(domain: str, question: str, query_type: str):
    wd = os.path.join(_ROOT, "data", "working_dir", domain)

    print(f"Question : {question}")
    print(f"Type     : {query_type}")
    print(f"Domain   : {domain}\n")

    # Shared LightRAG instance
    rag = LightRAG(working_dir=wd, llm_model_func=gpt_4o_mini_complete, embedding_func=openai_embed)
    await rag.initialize_storages()

    grasp = GRASPRAG(working_dir=wd)
    await grasp.initialize()

    from PathRAG import PathRAG as _PathRAG
    from PathRAG.llm import gpt_4o_mini_complete as pr_llm, openai_embedding as pr_embed
    from PathRAG.utils import EmbeddingFunc
    pathrag = _PathRAG(
        working_dir=wd,
        llm_model_func=pr_llm,
        llm_model_name="gpt-4o-mini",
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=pr_embed),
    )

    result = {
        "question": question,
        "query_type": query_type,
        "domain": domain,
        "methods": {}
    }

    for name, coro in [
        ("naive_rag",  capture_naive_rag(rag, question)),
        ("lightrag",   capture_lightrag(rag, question)),
        ("pathrag",    capture_pathrag(pathrag, question)),
        ("grasp_rag",  capture_grasp_rag(grasp, question)),
    ]:
        print(f"Running {name}...")
        try:
            result["methods"][name] = await coro
        except Exception as e:
            result["methods"][name] = {"error": str(e)}
        print(f"  done")

    await rag.finalize_storages()
    await grasp.finalize()

    out_dir = os.path.join(_ROOT, "data", "report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"example_{domain}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="agriculture")
    parser.add_argument("--question", type=str, default=None,
                        help="Exact question text to use")
    parser.add_argument("--index", type=int, default=0,
                        help="Index into the typed question set (used if --question not given)")
    parser.add_argument("--typed", action="store_true",
                        help="Pick from typed question set (default: typed)")
    parser.add_argument("--query-type", type=str, default=None,
                        help="Override query_type label in output")
    args = parser.parse_args()

    if args.question:
        question = args.question
        query_type = args.query_type or "unknown"
    else:
        suffix = "_typed" if args.typed else ""
        q_path = os.path.join(_ROOT, "data", "questions", f"{args.domain}{suffix}.json")
        with open(q_path) as f:
            qs = json.load(f)
        item = qs[args.index]
        question  = item["question"]
        query_type = args.query_type or item.get("query_type", "unknown")

    asyncio.run(run(args.domain, question, query_type))


if __name__ == "__main__":
    main()
