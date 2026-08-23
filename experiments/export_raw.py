"""
export_raw.py  —  Export raw UltraDomain JSONL to CSV for manual inspection.

Keeps: question, answer, source book, domain.
Drops: context, length, context_id (large / not useful for manual review).

Adds an empty "my_query_type" column so you can fill it in directly in Excel/Numbers.

Usage:
  python experiments/export_raw.py                    # all four domains
  python experiments/export_raw.py --domain legal     # single domain
  python experiments/export_raw.py --output my.csv    # custom output path
"""

import argparse
import csv
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOMAINS = ["agriculture", "cs", "health", "legal", "mix"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="all")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]

    rows = []
    for domain in domains:
        path = RAW_DIR / f"{domain}.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                r = json.loads(line.strip())
                rows.append({
                    "id":            r.get("_id", ""),
                    "domain":        r.get("label", domain),
                    "source":        r.get("meta", {}).get("title", ""),
                    "question":      r.get("input", "").strip(),
                    "answer":        r["answers"][0].strip() if r.get("answers") else "",
                    "my_query_type": "",   # fill this in yourself
                })

    if not rows:
        print("No data found.")
        return

    out = Path(args.output) if args.output else \
          RAW_DIR.parent / f"raw_{args.domain}.csv"

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
