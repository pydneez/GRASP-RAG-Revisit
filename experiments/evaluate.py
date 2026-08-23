"""
Step 4: Multi-judge LLM-as-judge evaluation following PathRAG's protocol.

For each question, compares GRASP-RAG against each baseline (naive_rag, lightrag, pathrag,
grasp_rag_fixed_*) using THREE independent LLM judges in parallel:
  - Anthropic Claude Haiku 4.5
  - Google    Gemini 3.6 Flash
  - DeepSeek  V4 Flash

OpenAI is deliberately excluded from the judge panel — it is the shared generation
model for every system under test (see grasp_rag/llm_config.py), and having it also
judge inflated GRASP-RAG's win rate vs NaiveRAG/LightRAG by 15-29pp relative to the
other two judges in the original paper (self-preference bias). Removing it from the
panel is the fix, not just a caveat.

Final outcome per dimension = majority vote (2/3 or 3/3 agree).
This eliminates single-model position bias and makes results more credible.
Judges with missing API keys are skipped gracefully.

Usage:
    python experiments/evaluate.py --domain agriculture
    python experiments/evaluate.py --domain agriculture --baseline lightrag
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]
FIXED_MOTIF_METHODS = ["grasp_rag_fixed_ego", "grasp_rag_fixed_bridge",
                        "grasp_rag_fixed_steiner", "grasp_rag_fixed_broad"]
BASELINES = ["naive_rag", "lightrag", "pathrag", "grasp_rag_nocascade",
             "grasp_rag_flat_context", "grasp_rag_oracle", *FIXED_MOTIF_METHODS]

# Evaluation prompt from PathRAG's GitHub repository
JUDGE_PROMPT = """\
You will evaluate two answers to the same question based on five criteria: \
**Comprehensiveness**, **Diversity**, **logicality**, **Coherence**, **Relevance**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **logicality**: How logically does the answer respond to all parts of the question?
- **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?
- **Relevance**: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?

For each criterion, choose the better answer (Answer 1 or Answer 2), or mark "Cannot Evaluate" if both answers fail to address the question meaningfully. Explain your choice.

Here is the question:
{question}

Here are the two answers:
**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "logicality": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Coherence": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Relevance": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }}
}}"""

# Used for ground-truth evaluation (UltraDomain).
# The reference answer anchors the judgment — judges pick which system is closer to it.
JUDGE_PROMPT_GT = """\
You will evaluate two system-generated answers to the same question. \
A reference answer written by a domain expert is provided as the evaluation standard. \
Use the reference to judge which system answer is more accurate, complete, and relevant.

Evaluate on five criteria:
- **Comprehensiveness**: Which answer covers more of the key aspects present in the reference?
- **Diversity**: Which answer provides richer and more varied perspectives consistent with the reference?
- **logicality**: Which answer is more logically structured and complete relative to the reference?
- **Coherence**: Which answer is more internally consistent and well-structured?
- **Relevance**: Which answer stays more on topic and aligns better with the reference?

Here is the question:
{question}

Here is the reference answer:
{reference}

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

For each criterion, choose the answer (Answer 1 or Answer 2) that better matches the reference, or mark "Cannot Evaluate" if neither answer meaningfully addresses the question. Explain your choice.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "logicality": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Coherence": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }},
  "Relevance": {{
    "Winner": "[Answer 1 or Answer 2 or Cannot Evaluate]",
    "Explanation": "[Provide explanation here]"
  }}
}}"""


def _parse_judge_response(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Parse the judge's response into (winners, explanations).

    Tries full JSON parse first. Falls back to per-dimension regex that pulls
    Winner and Explanation separately — robust against unescaped quotes in
    Explanation text that break standard JSON parsing.

    Returns:
        winners:      {dim: "Answer 1" | "Answer 2" | "Cannot Evaluate"}
        explanations: {dim: explanation_text}  (empty dict if unavailable)
    """
    # Attempt 1: standard JSON parse — gives us both Winner and Explanation
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            winners = {dim: parsed[dim]["Winner"] for dim in DIMENSIONS if dim in parsed}
            explanations = {dim: parsed[dim].get("Explanation", "") for dim in DIMENSIONS if dim in parsed}
            return winners, explanations
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Attempt 2: per-dimension regex — extracts Winner and Explanation separately,
    # ignoring structural JSON issues caused by unescaped quotes in free text.
    winners, explanations = {}, {}
    for dim in DIMENSIONS:
        winner_m = re.search(
            rf'"{re.escape(dim)}"\s*:\s*\{{[^{{}}]*?"Winner"\s*:\s*"([^"]+)"',
            text, re.DOTALL
        )
        expl_m = re.search(
            rf'"{re.escape(dim)}"\s*:\s*\{{[^{{}}]*?"Explanation"\s*:\s*"([^"]+)"',
            text, re.DOTALL
        )
        if winner_m:
            winners[dim] = winner_m.group(1)
        if expl_m:
            explanations[dim] = expl_m.group(1)

    return winners, explanations


# ── Per-model judge functions ─────────────────────────────────────────────────

async def judge_deepseek(question: str, answer1: str, answer2: str,
                         prompt_template: str = JUDGE_PROMPT,
                         reference: str = "") -> tuple[dict, dict]:
    """Judge using DeepSeek V4 Flash (OpenAI-compatible API)."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {}, {}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        prompt = prompt_template.format(question=question, answer1=answer1, answer2=answer2,
                                        reference=reference)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=120,
        )
        result = response.choices[0].message.content
        winners, explanations = _parse_judge_response(result)
        if winners:
            return winners, explanations
    except asyncio.TimeoutError:
        print(f"  [deepseek] judge timeout (120s)")
    except Exception as e:
        print(f"  [deepseek] judge error: {e}")
    return {}, {}


async def judge_claude(question: str, answer1: str, answer2: str,
                       prompt_template: str = JUDGE_PROMPT,
                       reference: str = "") -> tuple[dict, dict]:
    """Judge using Anthropic Claude Haiku."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}, {}
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = prompt_template.format(question=question, answer1=answer1, answer2=answer2,
                                    reference=reference)
    for attempt in range(6):
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=120,
            )
            result = response.content[0].text
            winners, explanations = _parse_judge_response(result)
            if winners:
                return winners, explanations
            return {}, {}
        except asyncio.TimeoutError:
            print(f"  [claude] judge timeout (120s, attempt {attempt + 1}/6)")
            if attempt < 5:
                await asyncio.sleep(2 ** attempt)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [claude] overloaded (attempt {attempt + 1}/6), retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  [claude] judge error: {e}")
                return {}, {}
        except Exception as e:
            print(f"  [claude] judge error: {e}")
            return {}, {}
    print("  [claude] gave up after 6 retries")
    return {}, {}


async def judge_gemini(question: str, answer1: str, answer2: str,
                       prompt_template: str = JUDGE_PROMPT,
                       reference: str = "") -> tuple[dict, dict]:
    """Judge using Google Gemini 3.6 Flash."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {}, {}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = prompt_template.format(question=question, answer1=answer1, answer2=answer2,
                                        reference=reference)
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            ),
            timeout=120,
        )
        result = response.text
        winners, explanations = _parse_judge_response(result)
        if winners:
            return winners, explanations
    except asyncio.TimeoutError:
        print(f"  [gemini] judge timeout (120s)")
    except Exception as e:
        print(f"  [gemini] judge error: {e}")
    return {}, {}


# ── Aggregation ───────────────────────────────────────────────────────────────

def majority_vote(judge_winners: list[dict[str, str]]) -> dict[str, str]:
    """
    Aggregate per-dimension verdicts from multiple judges via majority vote.

    A direction wins if at least 2 judges agree on "Answer 1" or "Answer 2".
    Everything else used to collapse into one generic "Undefined" bucket, which
    conflated three genuinely different situations. Now split into:
      - "CannotEvaluate": ≥2 judges' bias-corrected vote was "Cannot Evaluate"
        (neither answer meaningfully addressed the question — the literal,
        narrow meaning of "undefined" — judges couldn't evaluate it, not that
        we don't know who won).
      - "PositionBias": ≥2 judges' vote was "Tie" (their verdict flipped
        inconsistently between the forward/reversed ordering — a judge
        artifact, not a statement about the answers themselves).
      - "NoMajority": judges gave real, meaningful verdicts (Win/Loss/CE/Tie)
        but no single outcome reached a 2-vote majority — genuine disagreement.
    All three are still excluded from Win/Loss rate denominators, same as the
    old "Undefined" was, but are now distinguishable in the saved results.
    Judges that returned an empty dict (failed or skipped) are excluded automatically.
    """
    outcome = {}
    for dim in DIMENSIONS:
        votes = [j[dim] for j in judge_winners if dim in j]
        a1 = votes.count("Answer 1")
        a2 = votes.count("Answer 2")
        ce = votes.count("Cannot Evaluate")
        tie = votes.count("Tie")
        if a1 >= 2:
            outcome[dim] = "Answer 1"
        elif a2 >= 2:
            outcome[dim] = "Answer 2"
        elif ce >= 2:
            outcome[dim] = "CannotEvaluate"
        elif tie >= 2:
            outcome[dim] = "PositionBias"
        else:
            outcome[dim] = "NoMajority"
    return outcome


async def multi_judge(
    question: str, grasp_ans: str, base_ans: str,
    prompt_template: str = JUDGE_PROMPT,
    reference: str = "",
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Run all three judges in BOTH orderings in parallel (6 calls total).

    Each judge sees the question twice:
      - Forward:  answer1 = grasp_ans,  answer2 = base_ans
      - Reverse:  answer1 = base_ans,   answer2 = grasp_ans

    When reference is provided (UltraDomain GT evaluation), JUDGE_PROMPT_GT is used
    and the reference answer is included in every call as the evaluation anchor.

    A vote counts as "Answer 1" (grasp wins) only when:
      forward judge picks Answer 1  OR  reverse judge picks Answer 2.
    This neutralises per-model position bias before majority voting.

    Returns:
      outcome:    {dim: "Answer 1" | "Answer 2" | "CannotEvaluate" | "PositionBias" |
                        "NoMajority"}  (majority vote — see majority_vote() docstring)
      per_judge:  {"claude_fwd": {...}, "claude_rev": {...}, ...}
    """
    judges = [judge_claude, judge_gemini, judge_deepseek]
    names  = ["claude",     "gemini",     "deepseek"]

    # Run all 6 calls in parallel
    fwd_results, rev_results = zip(*await asyncio.gather(*[
        asyncio.gather(
            judge_fn(question, grasp_ans, base_ans, prompt_template, reference),   # forward
            judge_fn(question, base_ans, grasp_ans, prompt_template, reference),   # reverse
        )
        for judge_fn in judges
    ]))

    per_judge = {}
    all_votes: list[dict[str, str]] = []

    for name, (fwd_w, fwd_e), (rev_w, rev_e) in zip(names, fwd_results, rev_results):
        per_judge[f"{name}_fwd"] = fwd_w
        per_judge[f"{name}_fwd_explanations"] = fwd_e
        per_judge[f"{name}_rev"] = rev_w
        per_judge[f"{name}_rev_explanations"] = rev_e

        if not fwd_w and not rev_w:
            continue  # judge failed both — skip

        # Convert each ordering into a bias-corrected vote for GRASP-RAG
        bias_corrected: dict[str, str] = {}
        for dim in DIMENSIONS:
            f = fwd_w.get(dim, "Tie")
            r = rev_w.get(dim, "Tie")
            grasp_fwd_wins = (f == "Answer 1")
            grasp_rev_wins = (r == "Answer 2")
            base_fwd_wins  = (f == "Answer 2")
            base_rev_wins  = (r == "Answer 1")
            ce_fwd = (f == "Cannot Evaluate")
            ce_rev = (r == "Cannot Evaluate")

            if grasp_fwd_wins and grasp_rev_wins:
                bias_corrected[dim] = "Answer 1"        # GRASP genuinely better
            elif base_fwd_wins and base_rev_wins:
                bias_corrected[dim] = "Answer 2"        # baseline genuinely better
            elif ce_fwd and ce_rev:
                bias_corrected[dim] = "Cannot Evaluate" # both orderings unevaluable
            else:
                bias_corrected[dim] = "Tie"             # inconsistent → position bias
        all_votes.append(bias_corrected)

    if not all_votes:
        return {d: "Tie" for d in DIMENSIONS}, per_judge

    outcome = majority_vote(all_votes)
    return outcome, per_judge


def load_reference_answers(domain: str, typed_balanced: bool = False) -> dict[str, str]:
    """Load reference answers from the UltraDomain (Part A) or balanced typed (Part B)
    questions file. Returns {question: ref_text}."""
    suffix = "_typed_balanced" if typed_balanced else "_ultradomain"
    q_path = os.path.join(_ROOT, "data", "questions", f"{domain}{suffix}.json")
    if not os.path.exists(q_path):
        return {}
    with open(q_path, encoding="utf-8") as f:
        items = json.load(f)
    ref_map: dict[str, str] = {}
    for item in items:
        q = item.get("question", "")
        refs = item.get("reference_answers", [])
        if q and refs:
            ref_map[q] = "\n\n".join(refs) if isinstance(refs, list) else str(refs)
    return ref_map


_LOW_N_THRESHOLD = 10  # warn when fewer than this many questions are evaluable


_UNDECIDED_KEYS = ("CannotEvaluate", "PositionBias", "NoMajority")


def _print_dim_table(dim_counts: dict[str, dict[str, int]]) -> None:
    print(f"  {'Dimension':<20} {'Win%':>7} {'Loss%':>7} {'CantEval%':>10} "
          f"{'PosBias%':>9} {'NoMaj%':>8} {'n':>6}")
    print("  " + "-" * 78)
    low_n_dims = []
    for dim in DIMENSIONS:
        c = dim_counts[dim]
        n_undecided = sum(c.get(k, 0) for k in _UNDECIDED_KEYS)
        n_total   = c["Win"] + c["Loss"] + n_undecided
        n_decided = c["Win"] + c["Loss"]
        if n_total == 0:
            continue
        w  = 100 * c["Win"]  / n_decided if n_decided else 0.0
        l  = 100 * c["Loss"] / n_decided if n_decided else 0.0
        ce = 100 * c.get("CannotEvaluate", 0) / n_total
        pb = 100 * c.get("PositionBias", 0) / n_total
        nm = 100 * c.get("NoMajority", 0) / n_total
        print(f"  {dim:<20} {w:>6.1f}% {l:>6.1f}% {ce:>9.1f}% {pb:>8.1f}% {nm:>7.1f}% {n_decided:>5}")
        if n_decided < _LOW_N_THRESHOLD:
            low_n_dims.append((dim, n_decided))
    if low_n_dims:
        for dim, n in low_n_dims:
            print(f"  ⚠  {dim}: only {n} decided question(s) — percentages unreliable")


async def evaluate(domain: str, baselines: list[str], typed: bool = False,
                   ultradomain: bool = False, typed_balanced: bool = False,
                   concurrency: int = 20):
    if ultradomain:
        suffix = "_ultradomain"
    elif typed_balanced:
        suffix = "_typed_balanced"
    elif typed:
        suffix = "_typed"
    else:
        suffix = ""
    gt_anchored = ultradomain or typed_balanced
    answers_path = os.path.join(_ROOT, "data", "answers", f"{domain}{suffix}.jsonl")
    if not os.path.exists(answers_path):
        print(f"Answers not found: {answers_path}")
        sys.exit(1)

    # Load answers, indexed by (question, method)
    answers: dict[tuple[str, str], str] = {}
    # Also load query metadata (type, classified_type) keyed by question
    question_meta: dict[str, dict] = {}
    with open(answers_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[(r["question"], r["method"])] = r["answer"]
            if r["question"] not in question_meta:
                question_meta[r["question"]] = {
                    "query_type": r.get("query_type"),
                    "classified_type": r.get("classified_type"),
                    "classified_anchors": r.get("classified_anchors"),
                }

    questions = list({q for q, _ in answers.keys()})
    print(f"Evaluating {len(questions)} questions on domain={domain}")

    out_dir = os.path.join(_ROOT, "data", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Resumable raw judgment log — one line per (question, baseline, gt_mode) actually
    # judged, written incrementally as each completes. Previously this whole function
    # only wrote to disk once, at the very end of the entire domain run — a crash or
    # interruption (credits, network, laptop sleep) anywhere along the way lost
    # everything, including baselines that had already fully finished. This makes the
    # judge run resumable the same way run_queries.py already is.
    raw_path = os.path.join(out_dir, f"{domain}{suffix}_eval_raw.jsonl")
    done: set[tuple[str, str, bool]] = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["question"], r["baseline"], bool(r.get("gt_comparison"))))
        print(f"Resuming: {len(done)} judgments already saved in {raw_path}")

    raw_f = open(raw_path, "a", encoding="utf-8")
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    done_counter = [0]  # mutable counter shared across coroutines

    async def _judge_one(i, q, baseline, prompt_template, ref_ans="", gt_mode=False):
        if (q, baseline, gt_mode) in done:
            return {"skipped": True, "resumed": True, "i": i, "q": q}

        our_ans  = answers.get((q, "grasp_rag"), "")
        base_ans = answers.get((q, baseline), "")
        skip_reason = None
        if not our_ans:
            skip_reason = "grasp_rag answer missing"
        elif not base_ans:
            skip_reason = f"{baseline} answer missing"
        elif our_ans.startswith("ERROR"):
            skip_reason = f"grasp_rag error: {our_ans[:80]}"
        elif base_ans.startswith("ERROR"):
            skip_reason = f"{baseline} error: {base_ans[:80]}"
        if skip_reason:
            return {"skipped": True, "reason": skip_reason, "i": i, "q": q}

        async with sem:
            outcome, per_judge = await multi_judge(
                q, our_ans, base_ans, prompt_template, reference=ref_ans
            )

        done_counter[0] += 1
        print(f"  [{done_counter[0]}/{len(questions)}] judging...", end="\r")

        dim_outcome = {
            dim: (
                "Win" if v == "Answer 1" else
                "Loss" if v == "Answer 2" else
                v  # already one of CannotEvaluate / PositionBias / NoMajority
            )
            for dim, v in outcome.items()
        }
        meta = question_meta.get(q, {})
        entry = {
            "question": q, "baseline": baseline, "domain": domain,
            "query_type": meta.get("query_type"),
            "classified_type": meta.get("classified_type"),
            "classified_anchors": meta.get("classified_anchors"),
            "outcome": dim_outcome, "per_judge": per_judge,
        }
        if gt_mode:
            entry["gt_comparison"] = True

        async with write_lock:
            raw_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            raw_f.flush()

        return {"skipped": False, "i": i, "q": q}

    async def _run_baseline(baseline, prompt_template, ref_map=None, gt_mode=False):
        done_counter[0] = 0
        tasks = []
        for i, q in enumerate(questions):
            ref_ans = ref_map.get(q, "") if ref_map is not None else ""
            if ref_map is not None and not ref_ans:
                continue  # accounted for as "reference answer missing" in the final aggregation
            tasks.append(_judge_one(i, q, baseline, prompt_template, ref_ans=ref_ans, gt_mode=gt_mode))

        results = await asyncio.gather(*tasks)
        n_new = sum(1 for r in results if not r["skipped"])
        n_resumed = sum(1 for r in results if r.get("resumed"))
        print(f"\n  {baseline}: {n_new} newly judged"
              + (f", {n_resumed} resumed from a prior run" if n_resumed else ""))

    # ── Pairwise: GRASP-RAG vs each baseline (head-to-head, no reference) ──────
    # Skipped in GT-anchored mode (Part A/B) — reference-anchored comparison runs instead.
    for baseline in (baselines if not gt_anchored else []):
        print(f"\n── GRASP-RAG vs {baseline} ──")
        await _run_baseline(baseline, JUDGE_PROMPT)

    # ── Ground-truth anchored comparison (Part A ultradomain / Part B typed_balanced) ──
    ref_map_for_summary = None
    if gt_anchored:
        ref_map = load_reference_answers(domain, typed_balanced=typed_balanced)
        ref_map_for_summary = ref_map
        if ref_map:
            print(f"\n══ Ground-Truth Anchored Comparison (GRASP-RAG vs baseline, judged against reference) ══")
            for baseline in baselines:
                print(f"\n── GRASP-RAG vs {baseline} [+reference] ──")
                await _run_baseline(baseline, JUDGE_PROMPT_GT, ref_map=ref_map, gt_mode=True)
        else:
            print("\n[skip] No reference answers found — skipping ground-truth comparison.")

    raw_f.close()

    # ── Aggregate the raw judgment log (possibly built up across several resumed
    #    runs) into the summary format eval_classifier.py / judge_bias.py expect. ──
    all_results = []
    per_baseline_records: dict[str, list[dict]] = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if bool(r.get("gt_comparison")) != gt_anchored:
                continue  # raw log may hold both modes if this file is reused across runs
            per_baseline_records[r["baseline"]].append(r)
            all_results.append(r)

    for baseline in baselines:
        records = per_baseline_records.get(baseline, [])
        judged_qs = {r["question"] for r in records}
        dim_counts: dict[str, dict[str, int]] = {
            d: {"Win": 0, "Loss": 0, "CannotEvaluate": 0, "PositionBias": 0, "NoMajority": 0}
            for d in DIMENSIONS
        }
        for r in records:
            for dim, verdict in r["outcome"].items():
                dim_counts[dim][verdict] += 1

        skipped = 0
        skipped_log: list[str] = []
        for q in questions:
            if q in judged_qs:
                continue
            skipped += 1
            our_ans  = answers.get((q, "grasp_rag"), "")
            base_ans = answers.get((q, baseline), "")
            if ref_map_for_summary is not None and not ref_map_for_summary.get(q):
                reason = "reference answer missing"
            elif not our_ans:
                reason = "grasp_rag answer missing"
            elif not base_ans:
                reason = f"{baseline} answer missing"
            elif our_ans.startswith("ERROR"):
                reason = f"grasp_rag error: {our_ans[:80]}"
            elif base_ans.startswith("ERROR"):
                reason = f"{baseline} error: {base_ans[:80]}"
            else:
                reason = "not yet judged"
            skipped_log.append(f"    {reason} — {q[:80]}")

        total = len(judged_qs)
        print(f"\n  {baseline} — total evaluated: {total}  (skipped: {skipped})")
        _print_dim_table(dim_counts)
        summary = {
            "type": "summary", "baseline": baseline, "domain": domain,
            "total_evaluated": total, "skipped": skipped,
            "skipped_log": skipped_log,
            "dim_counts": {d: dict(c) for d, c in dim_counts.items()},
        }
        if gt_anchored:
            summary["gt_comparison"] = True
        all_results.append(summary)

    out_path = os.path.join(out_dir, f"{domain}{suffix}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results → {out_path}")
    print(f"Resumable raw log → {raw_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["agriculture", "cs", "legal", "health"],
                        default="agriculture")
    parser.add_argument("--all-domains", action="store_true",
                        help="Run all four domains sequentially (agriculture → cs → legal → health)")
    parser.add_argument("--baseline", nargs="+", choices=BASELINES, default=BASELINES)
    parser.add_argument("--typed", action="store_true",
                        help="Evaluate typed question set (*_typed.jsonl)")
    parser.add_argument("--ultradomain", action="store_true",
                        help="Evaluate UltraDomain question set (*_ultradomain.jsonl)")
    parser.add_argument("--typed-balanced", action="store_true",
                        help="Evaluate Part B balanced typed question set "
                             "(*_typed_balanced.jsonl), reference-anchored like --ultradomain")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max parallel questions being judged at once (default: 20)")
    args = parser.parse_args()

    domains = ["agriculture", "cs", "legal", "health"] if args.all_domains else [args.domain]
    for domain in domains:
        if args.all_domains:
            print(f"\n{'═'*60}")
            print(f"  Domain: {domain.upper()}")
            print(f"{'═'*60}")
        asyncio.run(evaluate(domain, args.baseline, typed=args.typed,
                             ultradomain=args.ultradomain,
                             typed_balanced=args.typed_balanced,
                             concurrency=args.concurrency))


if __name__ == "__main__":
    main()
