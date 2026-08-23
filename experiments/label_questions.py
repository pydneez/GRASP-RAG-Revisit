"""
Experiment 2 — LLM oracle labeling (ground-truth query types).

Uses a stronger-tier OpenAI model (see grasp_rag/llm_config.py::ORACLE_LABEL_MODEL)
as the oracle to assign query_type labels to all UltraDomain questions. The oracle
receives the same type definitions as the GRASP-RAG classifier but is a stronger
model than the classifier, making it suitable as ground truth for accuracy evaluation.

Type definitions:
  single_entity — asks about one specific named thing (definition, description, properties)
  bridge        — asks about the relationship/connection between exactly 2 named entities
  multi_entity  — asks to compare, integrate, or reason across 3+ named entities
  broad         — general question with no specific named entity focus

Reads:  data/questions/{domain}_ultradomain.json
Writes: data/labels/{domain}_gt_labels.json  → {question: gt_query_type}

Resumable: already-labeled questions are skipped on re-run.

Usage:
    python experiments/label_questions.py --domain agriculture
    python experiments/label_questions.py --all-domains
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
sys.path.insert(0, _ROOT)

from grasp_rag.llm_config import ORACLE_LABEL_MODEL

DOMAINS = ["agriculture", "cs", "legal", "health"]

LABEL_PROMPT = """\
Classify the following question into exactly one of these four query types:

- "single_entity": The question asks about ONE specific named entity (a thing, concept, process,
  organisation, technology, etc.) — asking for its definition, description, properties, role,
  or characteristics. Even if a domain phrase appears (e.g. "in agriculture"), if only one
  thing is the subject of inquiry, it is single_entity.

- "bridge": The question asks specifically about the RELATIONSHIP, CONNECTION, or CAUSATION
  between EXACTLY 2 named entities. Both entities must be subjects of inquiry, not just context
  words. Domain phrases like "in law" or "in CS" are NOT a second entity.

- "multi_entity": The question asks to COMPARE, INTEGRATE, or REASON across THREE OR MORE
  named entities simultaneously (e.g. "Compare X, Y, and Z", "How do X, Y, Z interact?").

- "broad": The question is general or thematic — it does NOT focus on specific named entities.
  It asks about trends, approaches, challenges, or summaries across a domain.

Question: {question}

Respond with ONLY a JSON object, no other text:
{{"query_type": "single_entity" | "bridge" | "multi_entity" | "broad"}}"""


async def label_with_gpt4(question: str, client) -> str:
    prompt = LABEL_PROMPT.format(question=question)
    try:
        response = await client.chat.completions.create(
            model=ORACLE_LABEL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=50,
        )
        text = response.choices[0].message.content or ""
        m = re.search(r'"query_type"\s*:\s*"(single_entity|bridge|multi_entity|broad)"', text)
        if m:
            return m.group(1)
    except Exception as e:
        print(f"\n  [gpt4o] error: {e}")
    return "broad"  # safe fallback


async def label_domain(domain: str):
    q_path = os.path.join(_ROOT, "data", "questions", f"{domain}_ultradomain.json")
    out_dir = os.path.join(_ROOT, "data", "labels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{domain}_gt_labels.json")

    if not os.path.exists(q_path):
        print(f"  Questions not found: {q_path}")
        print(f"  Run: python experiments/generate_questions.py --domain {domain} --mode ultradomain")
        return

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    # Load already-labeled for resumability
    done: dict[str, str] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)
        print(f"  Resuming: {len(done)}/{len(questions)} already labeled")

    todo = [item for item in questions if item["question"] not in done]
    if not todo:
        print(f"  All {len(questions)} questions already labeled.")
        _print_distribution(done)
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("  ERROR: OPENAI_API_KEY not set.")
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(15)
    counter = [0]
    save_lock = asyncio.Lock()

    async def _label_one(item: dict):
        q = item["question"]
        async with sem:
            label = await label_with_gpt4(q, client)
        async with save_lock:
            done[q] = label
            counter[0] += 1
            n_done = counter[0] + len(questions) - len(todo)
            print(f"  [{n_done}/{len(questions)}] {q[:60]}... → {label}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(done, f, indent=2, ensure_ascii=False)

    await asyncio.gather(*[_label_one(item) for item in todo])

    print(f"\n  Done — {len(done)} questions labeled → {out_path}")
    _print_distribution(done)


def _print_distribution(labels: dict[str, str]):
    from collections import Counter
    counts = Counter(labels.values())
    total = sum(counts.values())
    print("\n  Label distribution:")
    for qt in ["single_entity", "bridge", "multi_entity", "broad"]:
        n = counts.get(qt, 0)
        print(f"    {qt:<15}: {n:>4}  ({100*n/total:.1f}%)" if total else f"    {qt}: {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--all-domains", action="store_true",
                        help="Run all four domains sequentially")
    args = parser.parse_args()

    domains = DOMAINS if args.all_domains else [args.domain]
    for domain in domains:
        print(f"\n{'='*55}\n  Domain: {domain.upper()}\n{'='*55}")
        asyncio.run(label_domain(domain))


if __name__ == "__main__":
    main()
