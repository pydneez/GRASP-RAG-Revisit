"""
Capture the complete flow for "How does Cows connect to Agriculture?"
using STORED answers (from agriculture_typed.jsonl) so judge results
match the original evaluation (15/15 win).

Output: data/report/full_flow_cows.json
"""

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

QUESTION  = "How does Cows connect to Agriculture in the context of farming?"
QUERY_TYPE = "bridge"
DOMAIN    = "agriculture"
BASELINES = ["naive_rag", "lightrag", "pathrag"]
DIMENSIONS = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]

METHOD_DESCRIPTIONS = {
    "naive_rag": (
        "Vector search over raw text chunks. "
        "Top-k chunks ranked by cosine similarity are concatenated as context — no graph is used."
    ),
    "lightrag": (
        "Hybrid graph retrieval: extracts high-level and low-level keywords from the query, "
        "then combines local retrieval (entity ego-network) with global retrieval "
        "(community summaries), plus vector chunk augmentation."
    ),
    "pathrag": (
        "Path-based retrieval: retrieves top-k entity pairs by vector similarity, "
        "extracts and ranks graph paths between those pairs, and uses the scored "
        "path chains as structured tabular context."
    ),
    "grasp_rag": (
        "Query-adaptive motif retrieval: classifies the query into one of four motif types "
        "(single_entity, bridge, multi_entity, broad). For bridge queries, finds the "
        "shortest knowledge-graph path between the two anchor entities, ranks paths by "
        "edge-keyword relevance (Idea 3), then re-queries the vector store with edge "
        "keywords to retrieve supporting passages (Idea 1)."
    ),
}

# ── Load stored answers ───────────────────────────────────────────────────────

def load_stored_answers() -> dict[str, str]:
    path = os.path.join(_ROOT, "data", "answers", f"{DOMAIN}_typed.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["question"] == QUESTION:
                out[r["method"]] = r["answer"]
    return out

# ── Load stored eval verdicts ─────────────────────────────────────────────────

def load_stored_verdicts() -> dict[str, dict]:
    """Returns {baseline: {dim: Win/Loss/Tie}}"""
    path = os.path.join(_ROOT, "data", "results", f"{DOMAIN}_typed_eval.json")
    out = {}
    with open(path) as f:
        results = json.load(f)
    for r in results:
        if r["question"] == QUESTION:
            out[r["baseline"]] = r["outcome"]
    return out

# ── Retrieval prompt capture (prompts only, no LLM call) ──────────────────────

async def get_naive_prompt(rag):
    return await rag.aquery(QUESTION, param=QueryParam(mode="naive", only_need_prompt=True))

async def get_lightrag_prompt(rag):
    return await rag.aquery(QUESTION, param=QueryParam(mode="hybrid", only_need_prompt=True))

async def get_pathrag_prompt(pathrag):
    from PathRAG import QueryParam as PRQ
    from PathRAG.prompt import PROMPTS
    context = await pathrag.aquery(QUESTION, param=PRQ(mode="hybrid", only_need_context=True))
    return PROMPTS["rag_response"].format(
        context_data=context,
        response_type=PRQ().response_type,
    )

async def get_grasp_prompt(grasp):
    grasp._ensure_retriever()
    candidates = await grasp._fetch_candidates(QUESTION)
    query_type, anchors = await grasp._classifier.classify(QUESTION, candidates=candidates)
    paths = grasp._retriever.retrieve_direct(query_type, anchors)
    confidence = grasp._retriever.uncertainty(paths)

    if query_type == "broad" or confidence < _UNCERTAINTY_THRESHOLD or not paths:
        return {
            "classified_motif": query_type,
            "anchors_identified": anchors,
            "paths_retrieved": 0,
            "confidence": round(confidence, 4),
            "fallback_used": True,
            "retrieval_prompt": "(LightRAG hybrid fallback)",
            "graph_context": None,
        }

    context = await grasp._build_structured_context(paths, query_type, QUESTION)
    prompt  = _RESPONSE_PROMPT.format(context=context, query=QUESTION)
    return {
        "classified_motif": query_type,
        "anchors_identified": anchors,
        "paths_retrieved": len(paths),
        "confidence": round(confidence, 4),
        "fallback_used": False,
        "retrieval_prompt": prompt,
        "graph_context": context,
    }

# ── Judge prompt builder (uses stored answers) ────────────────────────────────

JUDGE_PROMPT_TMPL = """\
You are an impartial expert evaluator for AI-generated answers. \
Compare the two answers below to the given question.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

For each dimension, decide which answer is better: "A", "B", or "Tie".

Definitions:
- comprehensiveness: How thoroughly does the answer address all aspects of the question?
- diversity: How varied and rich are the perspectives and insights offered?
- logicality: How coherent and internally consistent is the reasoning?
- relevance: How directly and precisely does the answer address the question?
- coherence: How well-structured and readable is the answer?

Respond ONLY in JSON, no other text:
{{
  "comprehensiveness": "A" | "B" | "Tie",
  "diversity": "A" | "B" | "Tie",
  "logicality": "A" | "B" | "Tie",
  "relevance": "A" | "B" | "Tie",
  "coherence": "A" | "B" | "Tie"
}}"""


def build_judge_prompts(grasp_answer: str, baseline_answer: str) -> dict:
    return {
        "judge_prompt_forward": JUDGE_PROMPT_TMPL.format(
            question=QUESTION,
            answer_a=grasp_answer,
            answer_b=baseline_answer,
        ),
        "judge_prompt_reverse": JUDGE_PROMPT_TMPL.format(
            question=QUESTION,
            answer_a=baseline_answer,
            answer_b=grasp_answer,
        ),
        "note": (
            "Forward: Answer A = GRASP-RAG, Answer B = baseline. "
            "Reverse: Answer A = baseline, Answer B = GRASP-RAG. "
            "Final verdict = consistent outcome across both passes."
        ),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

async def run():
    wd = os.path.join(_ROOT, "data", "working_dir", DOMAIN)

    print("Loading stored answers and verdicts...")
    stored_answers  = load_stored_answers()
    stored_verdicts = load_stored_verdicts()
    for m, ans in stored_answers.items():
        print(f"  {m}: {len(ans)} chars")
    print()

    print("Initialising RAG systems for prompt capture...")
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
    print("Ready.\n")

    # ── Retrieval prompts ─────────────────────────────────────────────────────
    print("Capturing retrieval prompts...")

    naive_prompt   = await get_naive_prompt(rag)
    print("  naive_rag done")
    light_prompt   = await get_lightrag_prompt(rag)
    print("  lightrag done")
    path_prompt    = await get_pathrag_prompt(pathrag)
    print("  pathrag done")
    grasp_meta     = await get_grasp_prompt(grasp)
    print("  grasp_rag done\n")

    # ── Assemble methods block ────────────────────────────────────────────────
    methods = {
        "naive_rag": {
            "description":      METHOD_DESCRIPTIONS["naive_rag"],
            "retrieval_prompt": naive_prompt,
            "answer":           stored_answers.get("naive_rag", ""),
        },
        "lightrag": {
            "description":      METHOD_DESCRIPTIONS["lightrag"],
            "retrieval_prompt": light_prompt,
            "answer":           stored_answers.get("lightrag", ""),
        },
        "pathrag": {
            "description":      METHOD_DESCRIPTIONS["pathrag"],
            "retrieval_prompt": path_prompt,
            "answer":           stored_answers.get("pathrag", ""),
        },
        "grasp_rag": {
            "description":       METHOD_DESCRIPTIONS["grasp_rag"],
            "classified_motif":  grasp_meta["classified_motif"],
            "anchors_identified":grasp_meta["anchors_identified"],
            "paths_retrieved":   grasp_meta["paths_retrieved"],
            "confidence":        grasp_meta["confidence"],
            "fallback_used":     grasp_meta["fallback_used"],
            "graph_context":     grasp_meta["graph_context"],
            "retrieval_prompt":  grasp_meta["retrieval_prompt"],
            "answer":            stored_answers.get("grasp_rag", ""),
        },
    }

    # ── Judge comparisons (stored verdicts + reconstructed prompts) ───────────
    grasp_answer = stored_answers.get("grasp_rag", "")
    judge_comparisons = []
    for baseline in BASELINES:
        base_answer = stored_answers.get(baseline, "")
        verdict     = stored_verdicts.get(baseline, {})
        prompts     = build_judge_prompts(grasp_answer, base_answer)
        judge_comparisons.append({
            "grasp_rag_vs":    baseline,
            "final_outcome":   verdict,
            "verdict_source":  "stored_eval (agriculture_typed_eval.json)",
            **prompts,
        })
        wins   = sum(1 for v in verdict.values() if v == "Win")
        losses = sum(1 for v in verdict.values() if v == "Loss")
        print(f"  vs {baseline}: W{wins} L{losses}  "
              + " | ".join(f"{d}: {verdict.get(d,'?')}" for d in DIMENSIONS))

    # ── Write output ──────────────────────────────────────────────────────────
    report = {
        "question":   QUESTION,
        "query_type": QUERY_TYPE,
        "domain":     DOMAIN,
        "note": (
            "Answers loaded from data/answers/agriculture_typed.jsonl "
            "(original evaluation run). Judge verdicts from "
            "data/results/agriculture_typed_eval.json."
        ),
        "methods":           methods,
        "judge_comparisons": judge_comparisons,
    }

    await rag.finalize_storages()
    await grasp.finalize()

    out_dir  = os.path.join(_ROOT, "data", "report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "full_flow_cows.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
