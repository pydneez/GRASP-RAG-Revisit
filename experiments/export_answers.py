"""
export_answers.py  —  Convert JSONL answer files to CSV for easy inspection.

Two output formats:
  long  (default) — one row per (question × method)
  wide            — one row per question, one column per method (easier to compare)

Usage examples:
  python experiments/export_answers.py                          # all domains, long
  python experiments/export_answers.py --domain agriculture     # single domain
  python experiments/export_answers.py --format wide            # side-by-side answers
  python experiments/export_answers.py --domain cs --typed      # typed eval file
  python experiments/export_answers.py --truncate 300           # truncate answers
  python experiments/export_answers.py --query-type bridge      # filter by type
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DATA_DIR  = Path(__file__).resolve().parent.parent / "data" / "answers"
DOMAINS   = ["agriculture", "cs", "legal", "mix"]
METHODS   = ["grasp_rag", "naive_rag", "lightrag", "pathrag", "grasp_rag_nocascade"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_answer(text: str, truncate: int | None) -> str:
    """Strip leading markdown headings; optionally truncate."""
    text = text.strip()
    if truncate:
        text = text[:truncate] + ("…" if len(text) > truncate else "")
    return text


def build_long(records: list[dict], truncate: int | None) -> tuple[list[str], list[list]]:
    fieldnames = [
        "domain", "query_type", "classified_type", "classified_anchors",
        "method", "question", "answer",
    ]
    rows = []
    for r in records:
        rows.append([
            r.get("domain", ""),
            r.get("query_type", ""),
            r.get("classified_type", "") or "",
            "; ".join(r.get("classified_anchors") or []),
            r.get("method", ""),
            r.get("question", ""),
            clean_answer(r.get("answer", ""), truncate),
        ])
    return fieldnames, rows


def build_wide(records: list[dict], truncate: int | None) -> tuple[list[str], list[list]]:
    # Pivot: one row per (domain, question), methods become columns
    questions: dict[tuple, dict] = {}

    for r in records:
        key = (r.get("domain", ""), r.get("question", ""))
        if key not in questions:
            questions[key] = {
                "domain":             r.get("domain", ""),
                "query_type":         r.get("query_type", ""),
                "classified_type":    "",
                "classified_anchors": "",
                "question":           r.get("question", ""),
            }
        method = r.get("method", "")
        questions[key][method] = clean_answer(r.get("answer", ""), truncate)

        # Fill metadata from whichever row has it
        if r.get("classified_type"):
            questions[key]["classified_type"] = r["classified_type"]
        if r.get("classified_anchors"):
            questions[key]["classified_anchors"] = "; ".join(r["classified_anchors"])

    # Detect which methods actually appear
    present_methods = []
    for m in METHODS:
        if any(m in q for q in questions.values()):
            present_methods.append(m)

    fieldnames = [
        "domain", "query_type", "classified_type", "classified_anchors",
        "question",
    ] + present_methods

    rows = []
    for q in questions.values():
        rows.append([
            q.get("domain", ""),
            q.get("query_type", ""),
            q.get("classified_type", ""),
            q.get("classified_anchors", ""),
            q.get("question", ""),
            *[q.get(m, "") for m in present_methods],
        ])

    return fieldnames, rows


def main():
    parser = argparse.ArgumentParser(description="Export JSONL answer files to CSV.")
    parser.add_argument("--domain",     choices=DOMAINS + ["all"], default="all",
                        help="Domain to export (default: all)")
    parser.add_argument("--format",     choices=["long", "wide"], default="long",
                        help="Output format (default: long)")
    parser.add_argument("--typed",      action="store_true",
                        help="Use *_typed.jsonl files (default)")
    parser.add_argument("--ultradomain", action="store_true",
                        help="Use *_ultradomain.jsonl files instead")
    parser.add_argument("--truncate",   type=int, default=None,
                        help="Truncate answers to N characters")
    parser.add_argument("--query-type", dest="query_type", default=None,
                        choices=["single_entity", "bridge", "multi_entity", "broad"],
                        help="Filter to a specific query type")
    parser.add_argument("--method",     default=None, choices=METHODS,
                        help="Filter to a specific method")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (default: auto-named in data/)")
    args = parser.parse_args()

    suffix = "_ultradomain" if args.ultradomain else "_typed"
    domains = DOMAINS if args.domain == "all" else [args.domain]

    # Load records
    all_records: list[dict] = []
    for domain in domains:
        path = DATA_DIR / f"{domain}{suffix}.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        all_records.extend(load_jsonl(path))

    if not all_records:
        print("No records loaded. Check file paths.", file=sys.stderr)
        sys.exit(1)

    # Filter
    if args.query_type:
        all_records = [r for r in all_records if r.get("query_type") == args.query_type]
    if args.method:
        all_records = [r for r in all_records if r.get("method") == args.method]

    print(f"Loaded {len(all_records)} records from {len(domains)} domain(s).")

    # Build output
    if args.format == "wide":
        fieldnames, rows = build_wide(all_records, args.truncate)
    else:
        fieldnames, rows = build_long(all_records, args.truncate)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        tag = args.domain if args.domain != "all" else "all_domains"
        qt  = f"_{args.query_type}" if args.query_type else ""
        out_path = DATA_DIR.parent / f"answers_{tag}{suffix}{qt}_{args.format}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
