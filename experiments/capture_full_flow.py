"""
Capture the complete evaluation flow for one question per query type.

For each of the 4 question types (single_entity, bridge, multi_entity, broad):
  1. Retrieval prompt  → sent to LLM to get the answer
  2. Answer           → what the LLM returns (per method)
  3. Judge prompt     → sent to judge LLM (GRASP-RAG vs each baseline)
  4. Judge verdict    → per-dimension Win / Loss / Tie outcome

Output: data/report/full_flow_agriculture.json

Usage:
    python experiments/capture_full_flow.py --domain agriculture
"""

import argparse
import asyncio
import json
import os
import re
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

# ── Judge setup (mirrors evaluate.py) ────────────────────────────────────────

DIMENSIONS = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]

JUDGE_PROMPT = """\
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

METHOD_DESCRIPTIONS = {
    "naive_rag": (
        "Vector search over raw text chunks. "
        "Top-k chunks ranked by cosine similarity are concatenated as context — no graph is used."
    ),
    "lightrag": (
        "Hybrid graph retrieval: extracts high-level and low-level keywords from the query, "
        "then combines local retrieval (entity ego-network around matched entities) with "
        "global retrieval (community-level summaries), plus vector chunk augmentation."
    ),
    "pathrag": (
        "Path-based retrieval: retrieves top-k entity pairs by vector similarity, "
        "extracts and ranks graph paths between those pairs, and uses the scored path "
        "chains as structured tabular context."
    ),
    "grasp_rag": (
        "Query-adaptive motif retrieval: classifies the query into one of four motif types "
        "(single_entity → ego-network, bridge → shortest-path chain, "
        "multi_entity → Steiner-tree subgraph, broad → Personalized PageRank / hybrid fallback). "
        "The matching graph traversal strategy builds the context; a unified prose prompt "
        "synthesises the answer."
    ),
}

# ── Per-method prompt capture ─────────────────────────────────────────────────

async def capture_naive(rag, question):
    prompt = await rag.aquery(question, param=QueryParam(mode="naive", only_need_prompt=True))
    answer = await rag.aquery(question, param=QueryParam(mode="naive"))
    return {"retrieval_prompt": prompt, "answer": answer}


async def capture_lightrag(rag, question):
    prompt = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_prompt=True))
    answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    return {"retrieval_prompt": prompt, "answer": answer}


async def capture_pathrag(pathrag, question):
    from PathRAG import QueryParam as PRQ
    from PathRAG.prompt import PROMPTS
    context = await pathrag.aquery(question, param=PRQ(mode="hybrid", only_need_context=True))
    prompt = PROMPTS["rag_response"].format(
        context_data=context,
        response_type=PRQ().response_type,
    )
    answer = await pathrag.aquery(question, param=PRQ(mode="hybrid"))
    return {"retrieval_prompt": prompt, "answer": answer}


async def capture_grasp(grasp, question):
    grasp._ensure_retriever()
    candidates = await grasp._fetch_candidates(question)
    query_type, anchors = await grasp._classifier.classify(question, candidates=candidates)
    paths = grasp._retriever.retrieve_direct(query_type, anchors)
    confidence = grasp._retriever.uncertainty(paths)

    if query_type == "broad" or confidence < _UNCERTAINTY_THRESHOLD or not paths:
        prompt  = "(LightRAG hybrid fallback — no confident motif match)"
        context = None
        answer  = await grasp._rag.aquery(question, param=QueryParam(mode="hybrid"))
        fallback = True
    else:
        context = await grasp._build_structured_context(paths, query_type, question)
        prompt  = _RESPONSE_PROMPT.format(context=context, query=question)
        answer  = await grasp._rag.llm_model_func(prompt)
        fallback = False

    return {
        "classified_motif": query_type,
        "anchors_identified": anchors,
        "paths_retrieved": len(paths),
        "confidence": round(confidence, 4),
        "fallback_used": fallback,
        "graph_context": context,
        "retrieval_prompt": prompt,
        "answer": answer,
    }


# ── Judge ─────────────────────────────────────────────────────────────────────

async def run_judge(question, answer_grasp, answer_baseline):
    async def call(a, b):
        p = JUDGE_PROMPT.format(question=question, answer_a=a, answer_b=b)
        try:
            raw = await gpt_4o_mini_complete(p)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            verdict = json.loads(m.group(0)) if m else {}
        except Exception:
            verdict = {}
        return p, verdict

    fwd_prompt, fwd_verdict = await call(answer_grasp, answer_baseline)
    rev_prompt, rev_verdict = await call(answer_baseline, answer_grasp)

    # Merge with position-bias correction (mirrors evaluate.py)
    outcome = {}
    for dim in DIMENSIONS:
        f = fwd_verdict.get(dim, "Tie")
        r = rev_verdict.get(dim, "Tie")
        r_flip = {"A": "B", "B": "A", "Tie": "Tie"}.get(r, "Tie")
        if f == "A" and r_flip == "A":
            outcome[dim] = "Win"
        elif f == "B" and r_flip == "B":
            outcome[dim] = "Loss"
        elif f == "A" and r_flip == "Tie":
            outcome[dim] = "Win"
        elif f == "Tie" and r_flip == "A":
            outcome[dim] = "Win"
        elif f == "B" and r_flip == "Tie":
            outcome[dim] = "Loss"
        elif f == "Tie" and r_flip == "B":
            outcome[dim] = "Loss"
        else:
            outcome[dim] = "Tie"

    return {
        "judge_prompt_forward":  fwd_prompt,   # GRASP-RAG = Answer A
        "judge_verdict_forward": fwd_verdict,
        "judge_prompt_reverse":  rev_prompt,   # baseline  = Answer A
        "judge_verdict_reverse": rev_verdict,
        "final_outcome": outcome,              # Win / Loss / Tie for GRASP-RAG
    }


# ── Main ──────────────────────────────────────────────────────────────────────

# One question per type — picked to be representative
QUESTION_PICKS = {
    "single_entity": "What is Sustainable Agriculture?",
    "bridge":        "What is the relationship between Phosphorus and Beekeeping Industry?",
    "multi_entity":  "What are the interactions between Varroa Mites, Honey Bees, and Colony Collapse Disorder in the context of the beekeeping industry?",
    "broad":         "Summarise the key challenges facing modern agriculture practitioners.",
}

BASELINES = ["naive_rag", "lightrag", "pathrag"]


async def run(domain: str):
    wd = os.path.join(_ROOT, "data", "working_dir", domain)

    print("Initialising RAG systems...")
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

    report = {"domain": domain, "examples": []}

    for qtype, question in QUESTION_PICKS.items():
        print(f"── {qtype}: {question[:70]}...")
        entry = {
            "query_type": qtype,
            "question": question,
            "methods": {},
            "judge_comparisons": [],
        }

        # ── Step 1: capture each method ───────────────────────────────────────
        captures = {}
        for name, coro in [
            ("naive_rag", capture_naive(rag, question)),
            ("lightrag",  capture_lightrag(rag, question)),
            ("pathrag",   capture_pathrag(pathrag, question)),
            ("grasp_rag", capture_grasp(grasp, question)),
        ]:
            print(f"  {name}...", end="", flush=True)
            try:
                data = await coro
            except Exception as e:
                data = {"error": str(e)}
            captures[name] = data
            entry["methods"][name] = {
                "description": METHOD_DESCRIPTIONS[name],
                **data,
            }
            print(f" done ({len(data.get('answer',''))} chars)")

        # ── Step 2: judge GRASP-RAG vs each baseline ──────────────────────────
        grasp_answer = captures.get("grasp_rag", {}).get("answer", "")
        for baseline in BASELINES:
            base_answer = captures.get(baseline, {}).get("answer", "")
            if not grasp_answer or not base_answer:
                continue
            print(f"  judge: grasp_rag vs {baseline}...", end="", flush=True)
            j = await run_judge(question, grasp_answer, base_answer)
            entry["judge_comparisons"].append({
                "grasp_rag_vs": baseline,
                **j,
            })
            outcome_str = " | ".join(f"{d}: {v}" for d, v in j["final_outcome"].items())
            print(f" {outcome_str}")

        report["examples"].append(entry)
        print()

    await rag.finalize_storages()
    await grasp.finalize()

    out_dir = os.path.join(_ROOT, "data", "report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"full_flow_{domain}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="agriculture")
    parser.add_argument("--question", type=str, default=None,
                        help="Run for a single question instead of the default 4")
    parser.add_argument("--query-type", type=str, default="unknown",
                        help="Query type label when using --question")
    args = parser.parse_args()

    if args.question:
        # Override QUESTION_PICKS with just the one question
        global QUESTION_PICKS
        QUESTION_PICKS = {args.query_type: args.question}

    asyncio.run(run(args.domain))


if __name__ == "__main__":
    main()
