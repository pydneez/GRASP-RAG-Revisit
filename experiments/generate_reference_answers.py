"""
Step 2b: Generate reference (ground-truth) answers for the Part B balanced
typed question set (data/questions/{domain}_typed_balanced.json).

Unlike Part A (UltraDomain), which ships expert reference answers, Part B's
questions are newly generated (see generate_questions.py --typed-suffix
_typed_balanced) and need reference answers of their own before they can be
used with evaluate.py's reference-anchored judging (JUDGE_PROMPT_GT).

Grounding is deliberately NOT done via any of the four GRASP-RAG motifs
(ego/bridge/Steiner/PPR) — using one of those to build the reference would
structurally favor whichever Fixed variant happens to match that motif when
we later compare Adaptive vs Fixed. Instead this pulls a generous, motif-
agnostic pool of evidence (top-k chunk vector search + top-k entity vector
search) and asks a stronger oracle-tier model (ORACLE_LABEL_MODEL, see
grasp_rag/llm_config.py) to write a comprehensive answer from that pool only.

Resumable: questions that already have a non-empty reference_answers field
are skipped on re-run, so interruptions don't lose progress or re-bill.

Usage:
    python experiments/generate_reference_answers.py --domain agriculture
    python experiments/generate_reference_answers.py --all-domains
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

from lightrag import LightRAG
from lightrag.llm.openai import openai_embed

from grasp_rag.llm_config import ORACLE_LABEL_MODEL, generation_complete

DOMAINS = ["agriculture", "cs", "legal", "health"]

REFERENCE_PROMPT = """\
---Role---

You are a domain expert writing a reference (ground-truth) answer for an \
evaluation benchmark. Your answer will be used to judge OTHER systems' answers \
by comparison — it must be accurate, comprehensive, and strictly grounded in \
the evidence below.

---Evidence---

{context}

---Question---

{query}

---Instructions---

Write a comprehensive, well-organized answer using ONLY the evidence above. \
Do not use outside knowledge. If the evidence is insufficient to fully answer \
the question, say so explicitly rather than guessing. Do not mention "the \
evidence" or "the context" in your answer — write as a standalone factual \
answer, in the same style an expert reference answer would be written.
"""


async def _gather_context(rag: LightRAG, question: str,
                          chunk_k: int = 20, entity_k: int = 15) -> str:
    """Motif-agnostic grounding: generous chunk + entity vector search, no graph traversal."""
    chunk_task = rag.chunks_vdb.query(question, top_k=chunk_k)
    entity_task = rag.entities_vdb.query(question, top_k=entity_k)
    chunks, entities = await asyncio.gather(chunk_task, entity_task, return_exceptions=True)

    parts = []
    if not isinstance(entities, Exception) and entities:
        lines = []
        for e in entities:
            name = e.get("entity_name", "")
            desc = e.get("content", "")
            if name:
                lines.append(f"- {desc or name}")
        if lines:
            parts.append("Relevant entities:\n" + "\n\n".join(lines))

    if not isinstance(chunks, Exception) and chunks:
        passages = [c.get("content", "") for c in chunks if c.get("content", "").strip()]
        if passages:
            parts.append("Relevant passages:\n" + "\n\n".join(
                f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
            ))

    return "\n\n".join(parts)


async def generate_domain(domain: str, concurrency: int = 5):
    q_path = os.path.join(_ROOT, "data", "questions", f"{domain}_typed_balanced.json")
    if not os.path.exists(q_path):
        print(f"Questions not found: {q_path}. Run generate_questions.py --typed-suffix "
              f"_typed_balanced first.")
        return

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    todo = [q for q in questions if not q.get("reference_answers")]
    print(f"[{domain}] {len(questions)} questions total, {len(todo)} need reference answers")
    if not todo:
        return

    wd = os.path.join(_ROOT, "data", "working_dir", domain)
    # llm_model_func is required by LightRAG's constructor but never actually
    # invoked here — this script only uses rag's vector-store query methods,
    # answer generation goes through a separate direct client call below.
    rag = LightRAG(working_dir=wd, llm_model_func=generation_complete, embedding_func=openai_embed)
    await rag.initialize_storages()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def _save():
        # Whole-file rewrite under a lock — question sets are small (~80 each),
        # so this is cheap enough to do after every completion and keeps a
        # crash mid-domain from losing already-paid-for answers.
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

    async def _one(item: dict, i: int):
        async with sem:
            try:
                context = await _gather_context(rag, item["question"])
                if not context.strip():
                    item["reference_answers"] = ["[No grounding evidence found]"]
                    print(f"  [{i+1}/{len(todo)}] WARNING: no context — {item['question'][:60]}")
                else:
                    prompt = REFERENCE_PROMPT.format(context=context, query=item["question"])
                    resp = await client.chat.completions.create(
                        model=ORACLE_LABEL_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=1200,
                    )
                    answer = resp.choices[0].message.content or ""
                    item["reference_answers"] = [answer]
                    print(f"  [{i+1}/{len(todo)}] done ({len(answer)} chars) — {item['question'][:60]}")
            except Exception as e:
                print(f"  [{i+1}/{len(todo)}] ERROR: {e} — {item['question'][:60]}")
                return
            async with write_lock:
                await _save()

    await asyncio.gather(*[_one(item, i) for i, item in enumerate(todo)])
    print(f"[{domain}] saved → {q_path}")

    await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--all-domains", action="store_true")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    domains = DOMAINS if args.all_domains else [args.domain]
    for domain in domains:
        asyncio.run(generate_domain(domain, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
