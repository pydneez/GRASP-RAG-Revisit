"""
After evaluation, show per-question outcomes and identify the best question
for each query type where GRASP-RAG wins most clearly.

Usage:
    python experiments/find_winners.py --domain agriculture --typed
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

DIMENSIONS = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
BASELINES  = ["naive_rag", "lightrag", "pathrag"]


def score(outcomes: dict[str, str]) -> int:
    """Net wins across all dimensions: +1 Win, 0 Tie, -1 Loss."""
    return sum(1 if v == "Win" else -1 if v == "Loss" else 0
               for v in outcomes.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="agriculture")
    parser.add_argument("--typed", action="store_true")
    args = parser.parse_args()

    suffix = "_typed" if args.typed else ""
    result_path = os.path.join(_ROOT, "data", "results",
                               f"{args.domain}{suffix}_eval.json")
    if not os.path.exists(result_path):
        print(f"No eval results at {result_path}")
        sys.exit(1)

    results = json.load(open(result_path))

    # Load query types from answers file
    answers_path = os.path.join(_ROOT, "data", "answers",
                                f"{args.domain}{suffix}.jsonl")
    q_types: dict[str, str] = {}
    if os.path.exists(answers_path):
        with open(answers_path) as f:
            for line in f:
                r = json.loads(line)
                q_types[r["question"]] = r.get("query_type", "?")

    # Group results: question → baseline → outcome
    by_q: dict[str, dict] = {}
    for r in results:
        q  = r["question"]
        bl = r["baseline"]
        by_q.setdefault(q, {})[bl] = r["outcome"]

    # Score each question: total net wins across all baselines + dimensions
    rows = []
    for q, bl_outcomes in by_q.items():
        total = sum(score(bl_outcomes.get(bl, {})) for bl in BASELINES)
        rows.append((total, q, bl_outcomes))
    rows.sort(key=lambda x: -x[0])

    print(f"\n{'Score':>6}  {'Type':<14}  {'Question'}")
    print("─" * 90)
    for total, q, bl_outcomes in rows:
        qtype = q_types.get(q, "?")
        print(f"  {total:>4}  {qtype:<14}  {q[:65]}")
        for bl in BASELINES:
            out = bl_outcomes.get(bl, {})
            wins   = sum(1 for v in out.values() if v == "Win")
            losses = sum(1 for v in out.values() if v == "Loss")
            ties   = sum(1 for v in out.values() if v == "Tie")
            print(f"         vs {bl:<12}  W{wins} L{losses} T{ties}  "
                  + "  ".join(f"{d[:4]}:{out.get(d,'?')[0]}" for d in DIMENSIONS))
        print()

    # Best per query type
    print("\n── Best question per type ──────────────────────────────────────────────────")
    seen_types: set[str] = set()
    for total, q, bl_outcomes in rows:
        qtype = q_types.get(q, "?")
        if qtype not in seen_types:
            seen_types.add(qtype)
            print(f"\n[{qtype}]  score={total}")
            print(f"  Q: {q}")
            for bl in BASELINES:
                out = bl_outcomes.get(bl, {})
                summary = " | ".join(f"{d[:5]}:{v[0]}" for d, v in out.items())
                print(f"  vs {bl:<12}: {summary}")

    print("\n──────────────────────────────────────────────────────────────────────────")
    print("Pick a question above and run:")
    print('  python experiments/capture_full_flow.py --domain agriculture \\')
    print('    --question "QUESTION TEXT HERE"')


if __name__ == "__main__":
    main()
