"""
Experiment 3 — Stage 0: corpus preparation.

MASHQA JSON -> (a) corpus of 521 articles, (b) question records carrying gold
evidence as CHARACTER RANGES in their article.

Actual chunking happens in Stage 2 (index_corpus.py --mashqa), via LightRAG's
own default chunker — the same one E1/E2 use, needed for E3's identical-pipeline
requirement.

Outputs (JSONL, in --out):
  corpus.jsonl    {doc_id, title, text}
  questions.jsonl {qid, doc_id, question, answer_text, gold_sent_idx,
                   gold_char_ranges, n_gold_sents, n_evidence_blocks}
"""
import json, os, argparse, hashlib


def sha(s, n=12):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------- stage 0

def load_mashqa(path):
    data = json.load(open(path))["data"]
    corpus, questions = [], []
    for art in data:
        for pi, para in enumerate(art["paragraphs"]):
            text = para["context"]
            sents = para["sent_list"]
            starts = para["sent_starts"]
            doc_id = f"doc_{sha(art['title'] + str(pi))}"
            corpus.append({"doc_id": doc_id, "title": art["title"], "text": text})

            # sent_starts[i] == [char_start, char_length] and aligns with sent_list
            def sent_range(i):
                s, ln = starts[i]
                return [s, s + ln]

            for q in para["qas"]:
                ans = q["answers"][0]
                idx = sorted(ans["answer_span"])
                if not idx:
                    continue
                blocks = 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)
                questions.append({
                    "qid": q["id"],
                    "doc_id": doc_id,
                    "question": q["question"],
                    "answer_text": ans["text"],
                    "gold_sent_idx": idx,
                    "gold_char_ranges": [sent_range(i) for i in idx],
                    "n_gold_sents": len(idx),
                    "n_evidence_blocks": blocks,
                })
    return corpus, questions


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mashqa_json")
    ap.add_argument("--out", default="e3_data")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    corpus, questions = load_mashqa(a.mashqa_json)
    write_jsonl(f"{a.out}/corpus.jsonl", corpus)
    write_jsonl(f"{a.out}/questions.jsonl", questions)
    print(f"docs {len(corpus)} | questions {len(questions)}")
