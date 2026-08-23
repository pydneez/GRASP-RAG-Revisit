"""
export_eval_explanations.py — Export judge reasoning to CSV for human inspection.

evaluate.py's output JSON records the final outcome per dimension (Win/Loss/
CannotEvaluate/PositionBias/NoMajority) but the judges' actual explanations for
WHY are buried in nested per_judge dicts, six deep per dimension (3 judges x
2 orderings). This flattens that into one row per (question, baseline, dimension)
with every judge's verdict and explanation visible side by side — so a "Win" or
a "CannotEvaluate" always comes with the reasoning that produced it.

Usage:
    python experiments/export_eval_explanations.py --domain agriculture --ultradomain
    python experiments/export_eval_explanations.py --domain agriculture --typed-balanced
    python experiments/export_eval_explanations.py --all-domains --typed-balanced
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "data" / "results"
DOMAINS = ["agriculture", "cs", "legal", "health"]
JUDGES = ["claude", "gemini", "deepseek"]
DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]


def load_eval_records(domain: str, suffix: str) -> list[dict]:
    path = RESULTS_DIR / f"{domain}{suffix}_eval.json"
    if not path.exists():
        print(f"[skip] {path} not found", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [r for r in records if r.get("type") != "summary"]


def build_rows(records: list[dict]) -> tuple[list[str], list[list]]:
    fieldnames = [
        "domain", "baseline", "query_type", "classified_type",
        "question", "dimension", "outcome",
    ]
    for j in JUDGES:
        fieldnames += [f"{j}_fwd_winner", f"{j}_fwd_explanation",
                       f"{j}_rev_winner", f"{j}_rev_explanation"]

    rows = []
    for r in records:
        per_judge = r.get("per_judge", {})
        outcome = r.get("outcome", {})
        for dim in DIMENSIONS:
            row = [
                r.get("domain", ""),
                r.get("baseline", ""),
                r.get("query_type", ""),
                r.get("classified_type", "") or "",
                r.get("question", ""),
                dim,
                outcome.get(dim, ""),
            ]
            for j in JUDGES:
                fwd_w = per_judge.get(f"{j}_fwd", {}).get(dim, "")
                fwd_e = per_judge.get(f"{j}_fwd_explanations", {}).get(dim, "")
                rev_w = per_judge.get(f"{j}_rev", {}).get(dim, "")
                rev_e = per_judge.get(f"{j}_rev_explanations", {}).get(dim, "")
                row += [fwd_w, fwd_e, rev_w, rev_e]
            rows.append(row)
    return fieldnames, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--all-domains", action="store_true")
    parser.add_argument("--ultradomain", action="store_true", help="Part A results")
    parser.add_argument("--typed-balanced", action="store_true", help="Part B results")
    parser.add_argument("--typed", action="store_true", help="Legacy informal typed results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.ultradomain:
        suffix = "_ultradomain"
    elif args.typed_balanced:
        suffix = "_typed_balanced"
    elif args.typed:
        suffix = "_typed"
    else:
        suffix = ""

    domains = DOMAINS if args.all_domains else [args.domain]

    all_records = []
    for domain in domains:
        all_records.extend(load_eval_records(domain, suffix))

    if not all_records:
        print("No records loaded. Check that evaluate.py has been run for this domain/suffix.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_records)} judged (question, baseline) records "
          f"from {len(domains)} domain(s).")

    fieldnames, rows = build_rows(all_records)

    if args.output:
        out_path = Path(args.output)
    else:
        tag = "all_domains" if args.all_domains else args.domain
        out_path = RESULTS_DIR / f"eval_explanations_{tag}{suffix}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows ({len(all_records)} questions x {len(DIMENSIONS)} "
          f"dimensions) → {out_path}")


if __name__ == "__main__":
    main()
