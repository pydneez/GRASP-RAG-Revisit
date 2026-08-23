"""
Per-motif-type win-rate breakdown.

Reads the eval results JSON and the answers JSONL, joins them on question text,
then shows Win/Loss/Tie per dimension broken down by query_type.

This is the core analytical table for the paper: it shows WHICH motif types
GRASP-RAG excels at vs struggles with, validating the adaptive routing design.

Usage:
    python experiments/analyze_motif.py --domain agriculture --typed
    python experiments/analyze_motif.py --domain agriculture --typed --baseline lightrag
    python experiments/analyze_motif.py --domain agriculture --typed --baseline grasp_rag_nocascade
"""

import argparse
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]
QUERY_TYPES = ["single_entity", "bridge", "multi_entity", "broad"]


def load_query_types(domain: str, suffix: str) -> dict[str, str]:
    """Return {question_text: query_type} from the answers JSONL."""
    path = os.path.join(_ROOT, "data", "answers", f"{domain}{suffix}.jsonl")
    mapping: dict[str, str] = {}
    if not os.path.exists(path):
        print(f"Answers file not found: {path}")
        return mapping
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qt = r.get("query_type")
            if qt:
                mapping[r["question"]] = qt
    return mapping


def analyze(domain: str, baseline: str, suffix: str):
    results_path = os.path.join(_ROOT, "data", "results", f"{domain}{suffix}_eval.json")
    if not os.path.exists(results_path):
        print(f"Results not found: {results_path}")
        return

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    q2type = load_query_types(domain, suffix)

    # Filter to requested baseline
    records = [r for r in data if r["baseline"] == baseline]
    if not records:
        available = sorted({r["baseline"] for r in data})
        print(f"No results for baseline '{baseline}'. Available: {available}")
        return

    # Aggregate: {query_type: {dim: {Win/Loss/CannotEvaluate/PositionBias/NoMajority: count}}}
    counts: dict[str, dict[str, dict[str, int]]] = {
        qt: {d: {"Win": 0, "Loss": 0, "CannotEvaluate": 0, "PositionBias": 0, "NoMajority": 0}
             for d in DIMENSIONS}
        for qt in QUERY_TYPES + ["ALL"]
    }

    for r in records:
        qt = q2type.get(r["question"], "unknown")
        outcome = r.get("outcome", {})
        for dim, verdict in outcome.items():
            if dim not in DIMENSIONS:
                continue
            if qt in counts:
                counts[qt][dim][verdict] += 1
            counts["ALL"][dim][verdict] += 1

    # Print table
    print(f"\n{'='*70}")
    print(f"  GRASP-RAG vs {baseline}  |  domain={domain}")
    print(f"{'='*70}")

    col_w = 10
    for qt in QUERY_TYPES + ["ALL"]:
        total_q = sum(counts[qt][DIMENSIONS[0]].values())
        if total_q == 0:
            continue
        separator = "─" * 70 if qt != "ALL" else "═" * 70
        print(f"\n  [{qt.upper()}]  ({total_q} questions)")
        print(f"  {'Dimension':<22} {'Win%':>{col_w}} {'Loss%':>{col_w}} "
              f"{'CantEval%':>{col_w}} {'PosBias%':>{col_w}} {'NoMaj%':>{col_w}}")
        print(f"  {separator[:66]}")
        for dim in DIMENSIONS:
            c = counts[qt][dim]
            n_decided = c["Win"] + c["Loss"]
            n_undecided = c.get("CannotEvaluate", 0) + c.get("PositionBias", 0) + c.get("NoMajority", 0)
            n_total   = n_decided + n_undecided
            if n_total == 0:
                continue
            w  = 100 * c["Win"]  / n_decided if n_decided else 0.0
            l  = 100 * c["Loss"] / n_decided if n_decided else 0.0
            ce = 100 * c.get("CannotEvaluate", 0) / n_total
            pb = 100 * c.get("PositionBias", 0) / n_total
            nm = 100 * c.get("NoMajority", 0) / n_total
            print(f"  {dim:<22} {w:>{col_w}.1f}% {l:>{col_w}.1f}% "
                  f"{ce:>{col_w}.1f}% {pb:>{col_w}.1f}% {nm:>{col_w}.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["agriculture", "cs", "legal", "health"],
                        default="agriculture")
    parser.add_argument("--baseline", default="naive_rag",
                        help="Which baseline to break down (default: naive_rag)")
    parser.add_argument("--typed", action="store_true")
    parser.add_argument("--ultradomain", action="store_true")
    parser.add_argument("--bridge-gt", action="store_true")
    args = parser.parse_args()

    if args.bridge_gt:
        suffix = "_bridge_gt"
    elif args.typed:
        suffix = "_typed"
    elif args.ultradomain:
        suffix = "_ultradomain"
    else:
        suffix = ""

    analyze(args.domain, args.baseline, suffix)


if __name__ == "__main__":
    main()
