"""
Step 2: Generate evaluation questions.

Modes:
  --mode lightrag      (default) — LightRAG's 5×5×5 protocol: broad sensemaking questions
  --mode typed                   — GRASP-RAG typed set: 30 questions per motif type (120 total)
  --mode both                    — generate both and save to separate files
  --mode ultradomain             — use questions already in the UltraDomain raw dataset (input field)

Typed question types:
  single_entity  — asks about one specific named entity (ego-network motif)
  bridge         — asks about the relationship between exactly two entities (bridge-path motif)
  multi_entity   — asks to compare/integrate 3+ entities (Steiner-tree motif)
  broad          — thematic/summarisation, no specific entity anchor (PPR motif)

Usage:
    python experiments/generate_questions.py --domain agriculture
    python experiments/generate_questions.py --domain agriculture --mode typed
    python experiments/generate_questions.py --domain agriculture --mode both
    python experiments/generate_questions.py --domain agriculture --mode ultradomain
    python experiments/generate_questions.py --domain agriculture --no-corpus

Output files:
  {domain}.json             lightrag mode
  {domain}_typed.json       typed mode
  {domain}_ultradomain.json ultradomain mode
"""

import argparse
import asyncio
import json
import os
import random
import sys

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _ROOT)

from grasp_rag.llm_config import generation_complete

DOMAINS = ["agriculture", "cs", "legal", "health"]
N_PER_TYPE = 30  # questions per motif type in typed mode

# ── LightRAG protocol prompts ─────────────────────────────────────────────────

PERSONA_PROMPT = """\
You are helping design evaluation questions for a Retrieval-Augmented Generation system.

The corpus domain is: {domain}

Here is a representative sample of what the corpus actually contains:
{corpus_description}

Generate {n_personas} distinct expert user personas who would realistically use a RAG system \
over this specific corpus. Each persona should have:
- A role/profession
- Their primary goal
- Their level of expertise

Return ONLY a JSON array of objects:
[{{"role": "...", "goal": "...", "expertise": "..."}}]"""

TASK_PROMPT = """\
Given this expert user persona:
Role: {role}
Goal: {goal}
Expertise: {expertise}

Generate {n_tasks} distinct research tasks this person might pursue using a \
knowledge base about {domain}. Each task should require reading across multiple documents.

Return ONLY a JSON array of strings (task descriptions):
["task 1", "task 2", ...]"""

QUESTION_PROMPT = """\
Given this research task in the domain of {domain}:
Task: {task}

Generate {n_questions} specific questions that would help accomplish this task. \
Each question must:
- Require synthesizing information from multiple sources
- Not be answerable from a single sentence
- Be concrete and answerable from a domain knowledge base

Return ONLY a JSON array of strings:
["question 1", "question 2", ...]"""

# ── Typed question prompts ────────────────────────────────────────────────────

SINGLE_ENTITY_PROMPT = """\
Domain: {domain}

Here are real named entities from the knowledge graph for this domain:
{entities}

Generate {n} questions that each ask about EXACTLY ONE of the entities listed above.
Questions should ask for a definition, description, properties, or role of that single entity.
Good formats: "What is [entity]?", "Describe [entity]", "What role does [entity] play in {domain}?",
"What are the key characteristics of [entity]?"

Rules:
- Each question must focus on exactly one named entity from the list
- Questions must be answerable from a {domain} knowledge base
- Do not ask about relationships between multiple entities

Return ONLY a JSON array of strings (the questions):
["question 1", "question 2", ...]"""

BRIDGE_PROMPT = """\
Domain: {domain}

Below are pairs of named entities from the knowledge graph. Each pair has been \
verified to have NO overlapping source text — they appear in completely separate \
parts of the corpus, so their connection can only be discovered by traversing \
the knowledge graph, not by reading a single passage.

Entity pairs (A → B):
{entity_pairs}

Generate {n} questions, one per pair, asking about the hidden connection between \
the two entities. The question must NOT mention any intermediate entity — only A and B.

Good formats:
  "What is the relationship between [A] and [B]?"
  "How does [A] connect to [B] in the context of {domain}?"
  "What links [A] to [B]?"
  "How are [A] and [B] related?"

Rules:
- Use EXACTLY the entity names as given (one question per pair)
- Do NOT reveal or hint at the intermediate connecting entity
- The connection must require cross-document reasoning to answer

Return ONLY a JSON array of strings (one question per pair, same order):
["question 1", "question 2", ...]"""

MULTI_ENTITY_PROMPT = """\
Domain: {domain}

Here are real named entities from the knowledge graph for this domain:
{entities}

Generate {n} questions that require COMPARING or INTEGRATING information about \
THREE OR MORE of the entities listed above.
Good formats: "Compare [X], [Y], and [Z] in terms of...",
"What are the interactions between [X], [Y], and [Z]?",
"How do [X], [Y], and [Z] collectively contribute to [topic]?",
"Analyse the roles of [X], [Y], and [Z] in {domain}"

Rules:
- Each question must mention at least 3 named entities from the list
- Require integrating or comparing information across all mentioned entities
- Questions must be answerable from a {domain} knowledge base

Return ONLY a JSON array of strings (the questions):
["question 1", "question 2", ...]"""

BROAD_PROMPT = """\
Domain: {domain}

Here is a representative sample of what the corpus actually contains:
{corpus_description}

Generate {n} broad thematic or summarisation questions about {domain} that:
- Do NOT ask about any specific named entity
- Require understanding themes, trends, or overarching concepts across the corpus
- Cannot be answered from a single passage
- Are grounded in topics that appear in the corpus sample above

Good formats: "What are the main approaches to [topic] in {domain}?",
"Summarise the key challenges facing {domain} practitioners",
"What trends have emerged in {domain} research?",
"How has [broad concept] evolved in {domain}?"

Return ONLY a JSON array of strings (the questions):
["question 1", "question 2", ...]"""


# ── Helpers ───────────────────────────────────────────────────────────────────

async def call_llm(prompt: str) -> str:
    return await generation_complete(prompt)


def build_corpus_description(domain: str, n_sample: int = 15, chars_each: int = 600) -> str:
    """
    Read the raw corpus for a domain and build a representative description.

    Samples n_sample documents, takes chars_each characters from the start of each
    (which typically contains the title and introduction), and joins them.
    This mirrors LightRAG's Step_0/Step_2 approach of summarising corpus content
    before generating questions, so questions are grounded in what the corpus
    actually covers rather than just the domain name.
    """
    raw_path = os.path.join(_ROOT, "data", "raw", f"{domain}.jsonl")
    if not os.path.exists(raw_path):
        print(f"  WARNING: corpus not found at {raw_path} — using domain name only")
        return f"(corpus not available; domain: {domain})"

    with open(raw_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    random.seed(42)
    sample = random.sample(records, min(n_sample, len(records)))

    snippets = []
    for i, rec in enumerate(sample, 1):
        text = rec.get("context", rec.get("content", "")).strip()
        if not text:
            continue
        # Take chars from beginning (title/intro) and a window from midway through
        head = text[:chars_each // 2]
        mid_start = max(0, len(text) // 2 - chars_each // 4)
        mid = text[mid_start: mid_start + chars_each // 2]
        snippet = f"[Document {i}]\n{head}\n...\n{mid}"
        snippets.append(snippet)

    description = "\n\n".join(snippets)
    print(f"  Corpus description built: {len(sample)} documents sampled, "
          f"{len(description)} chars total")
    return description


async def parse_json(text: str) -> list:
    import re
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return []


_EXCLUDED_TYPES = {
    "person", "location", "content", "UNKNOWN", "data", "document",
    "book", "event", "date", "people", "occupation", "nation",
    "publication", "timeframe", "timeperiod", "time", "currency",
}

def sample_entities(domain: str, n: int = 40) -> list[str]:
    """Pull high-degree node names from the built knowledge graph.

    Filters out: dirty node names (trailing/embedded underscores),
    non-domain entity types (persons, locations, books, etc.), and
    nodes whose description is too short to be meaningful.
    """
    graphml = os.path.join(_ROOT, "data", "working_dir", domain,
                           "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml):
        return []
    G = nx.read_graphml(graphml)

    def _is_clean(node: str) -> bool:
        attrs = G.nodes[node]
        if "_" in node:
            return False
        etype = attrs.get("entity_type", "UNKNOWN")
        if etype in _EXCLUDED_TYPES:
            return False
        desc = attrs.get("description", "")
        if len(desc) < 30:
            return False
        return True

    clean = [n for n in G.nodes() if _is_clean(n)]
    by_degree = sorted(clean, key=lambda v: G.degree(v), reverse=True)
    top = by_degree[:200]
    random.seed(42)
    return random.sample(top, min(n, len(top)))


def find_bridge_candidates(domain: str, n: int = 40,
                            max_hops: int = 3) -> list[tuple[str, str, list[str]]]:
    """
    Find entity pairs that are genuinely hard for NaiveRAG:
      - No shared text chunk  (NaiveRAG can't find their connection in any single passage)
      - Connected via a short graph path  (GRASP-RAG can traverse it)
      - Path does not pass through high-degree hub nodes  (keeps paths meaningful)

    Returns list of (entity_a, entity_b, path_nodes).
    These pairs are the ideal bridge question seeds — the graph path is the only
    way to discover the connection, giving GRASP-RAG a structural advantage.
    """
    from itertools import combinations

    graphml = os.path.join(_ROOT, "data", "working_dir", domain,
                           "graph_chunk_entity_relation.graphml")
    entity_chunks_path = os.path.join(_ROOT, "data", "working_dir", domain,
                                      "kv_store_entity_chunks.json")
    text_chunks_path   = os.path.join(_ROOT, "data", "working_dir", domain,
                                      "kv_store_text_chunks.json")

    if not all(os.path.exists(p) for p in [graphml, entity_chunks_path, text_chunks_path]):
        print(f"  WARNING: working_dir not found for {domain} — skipping bridge candidate search")
        return []

    G = nx.read_graphml(graphml)
    with open(entity_chunks_path, encoding="utf-8") as f:
        entity_chunks = json.load(f)
    with open(text_chunks_path, encoding="utf-8") as f:
        text_chunks = json.load(f)

    def _is_clean(node: str) -> bool:
        attrs = G.nodes[node]
        if "_" in node: return False
        if attrs.get("entity_type", "UNKNOWN") in _EXCLUDED_TYPES: return False
        if len(attrs.get("description", "")) < 30: return False
        return True

    # Top-200 clean entities by degree — same pool as sample_entities()
    top200 = sorted(
        [nd for nd in G.nodes() if _is_clean(nd)],
        key=lambda nd: G.degree(nd), reverse=True
    )[:200]

    # Hub nodes = top-50 by degree; we reject paths passing through them
    hub_set = set(top200[:50])

    # Per-entity chunk sets (for overlap detection)
    chunk_sets = {
        e: set(entity_chunks.get(e, {}).get("chunk_ids", []))
        for e in top200
    }

    candidates = []
    random.seed(42)
    pairs = list(combinations(top200, 2))
    random.shuffle(pairs)

    for a, b in pairs:
        if len(candidates) >= n * 3:   # collect extra, we'll sample down later
            break
        # Must have no shared text chunk
        if chunk_sets[a] & chunk_sets[b]:
            continue
        try:
            path = nx.shortest_path(G, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2 or len(path) > max_hops + 1:
            continue
        # Reject if any intermediary is a hub
        if any(node in hub_set for node in path[1:-1]):
            continue
        candidates.append((a, b, path))

    random.shuffle(candidates)
    result = candidates[:n]
    print(f"  Bridge candidates found: {len(candidates)} total, using {len(result)}")
    return result


def filter_tight_bridges(
    candidates: list[tuple[str, str, list[str]]],
) -> list[tuple[str, str, list[str]]]:
    """
    Tighten find_bridge_candidates()'s output: graph-path existence alone doesn't
    mean the connection is thematically meaningful (a spot-check found ~9% of
    generated bridge questions had reference answers saying the relationship was
    essentially undocumented). Keeps only pairs connected via exactly ONE
    intermediate entity (path length 3: a -> mid -> b) rather than up to 3 hops —
    a true two-hop bridge is far more likely to reflect a real, describable
    connection than a longer incidental chain through the graph. Pure graph
    computation, no extra API calls (an embedding-based semantic check was tried
    first but proved too rate-limit-fragile to run reliably at this volume).
    """
    return [c for c in candidates if len(c[2]) == 3]


def save_questions(questions: list[dict], domain: str, suffix: str = "") -> str:
    out_dir = os.path.join(_ROOT, "data", "questions")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{domain}{suffix}.json"
    out_path = os.path.join(out_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(questions)} questions → {out_path}")
    return out_path


# ── Generation functions ──────────────────────────────────────────────────────

async def generate_lightrag(domain: str, corpus_description: str,
                             n_personas: int = 5,
                             n_tasks: int = 5, n_questions: int = 5) -> list[dict]:
    """LightRAG's 5×5×5 protocol — broad sensemaking questions."""
    print(f"[LightRAG protocol] domain={domain}")
    questions = []

    persona_resp = await call_llm(PERSONA_PROMPT.format(
        domain=domain, n_personas=n_personas, corpus_description=corpus_description,
    ))
    personas = await parse_json(persona_resp)
    print(f"  Generated {len(personas)} personas")

    for persona in personas:
        task_resp = await call_llm(TASK_PROMPT.format(
            role=persona.get("role", "Expert"),
            goal=persona.get("goal", "Research"),
            expertise=persona.get("expertise", "Advanced"),
            n_tasks=n_tasks,
            domain=domain,
        ))
        tasks = await parse_json(task_resp)

        for task in tasks:
            q_resp = await call_llm(QUESTION_PROMPT.format(
                domain=domain, task=task, n_questions=n_questions,
            ))
            for q in await parse_json(q_resp):
                questions.append({
                    "question": str(q),
                    "query_type": "broad",   # LightRAG questions are all broad
                    "persona": persona,
                    "task": task,
                    "domain": domain,
                })

    return questions


async def generate_typed(domain: str, corpus_description: str,
                          n_per_type: int = N_PER_TYPE) -> list[dict]:
    """Generate n_per_type questions for each of the 4 motif types.

    Bridge questions use verified cross-document entity pairs (no shared chunk),
    so they genuinely require graph traversal and cannot be answered by NaiveRAG
    from a single retrieved passage.
    """
    print(f"[Typed protocol] domain={domain}, {n_per_type} per type")

    entities = sample_entities(domain, n=40)
    entity_str = "\n".join(f"- {e}" for e in entities)
    if not entities:
        print("  WARNING: no graph found — entity-based questions may be generic")
        entity_str = f"(use relevant {domain} domain entities)"

    # Bridge: use verified cross-document pairs instead of random entity list
    bridge_candidates = find_bridge_candidates(domain, n=n_per_type * 3)
    if bridge_candidates:
        bridge_pair_str = "\n".join(
            f"- {a}  →  {b}  (connected via: {' → '.join(path[1:-1]) or 'direct'})"
            for a, b, path in bridge_candidates[:n_per_type * 2]
        )
    else:
        # Fallback: use random entity pairs if graph not available
        bridge_pair_str = entity_str
        print("  WARNING: no bridge candidates found — using random entities for bridge")

    questions = []

    type_configs = [
        ("single_entity", SINGLE_ENTITY_PROMPT),
        ("bridge",        BRIDGE_PROMPT),
        ("multi_entity",  MULTI_ENTITY_PROMPT),
        ("broad",         BROAD_PROMPT),
    ]

    for qtype, prompt_template in type_configs:
        if qtype == "broad":
            prompt = prompt_template.format(
                domain=domain, n=n_per_type, corpus_description=corpus_description,
            )
        elif qtype == "bridge":
            prompt = prompt_template.format(
                domain=domain, entity_pairs=bridge_pair_str, n=n_per_type,
            )
        else:
            prompt = prompt_template.format(
                domain=domain, entities=entity_str, n=n_per_type
            )

        resp = await call_llm(prompt)
        qs = await parse_json(resp)

        # Sometimes the LLM returns fewer than requested — ask again for the shortfall
        if len(qs) < n_per_type:
            extra_resp = await call_llm(prompt)
            qs += await parse_json(extra_resp)

        for q in qs[:n_per_type]:
            questions.append({
                "question": str(q),
                "query_type": qtype,
                "domain": domain,
            })

        print(f"  {qtype}: {min(len(qs), n_per_type)} questions")

    return questions


# ── UltraDomain loader ────────────────────────────────────────────────────────

def load_ultradomain(domain: str) -> list[dict]:
    """
    Extract questions directly from the UltraDomain raw dataset.

    The raw .jsonl files have an 'input' field (the question) and an 'answers'
    field (reference answers). We save questions in the standard format so
    run_queries.py can treat them identically to generated questions.
    The reference answers are stored alongside for potential future use.
    """
    raw_path = os.path.join(_ROOT, "data", "raw", f"{domain}.jsonl")
    if not os.path.exists(raw_path):
        print(f"  ERROR: raw corpus not found at {raw_path}")
        return []

    questions = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("input", "").strip()
            if not q:
                continue
            questions.append({
                "question": q,
                "query_type": "broad",
                "domain": domain,
                "reference_answers": rec.get("answers", []),
            })

    print(f"  Loaded {len(questions)} questions from UltraDomain ({domain})")
    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(domain: str, mode: str, use_corpus: bool = True,
              n_per_type: int = N_PER_TYPE,
              n_personas: int = 5, n_tasks: int = 5, n_questions: int = 5,
              typed_suffix: str = "_typed"):
    if mode == "ultradomain":
        qs = load_ultradomain(domain)
        save_questions(qs, domain, suffix="_ultradomain")
        return

    if use_corpus:
        print(f"Reading corpus for domain={domain}...")
        corpus_description = build_corpus_description(domain)
    else:
        corpus_description = f"(domain: {domain})"

    if mode in ("lightrag", "both"):
        qs = await generate_lightrag(domain, corpus_description,
                                     n_personas=n_personas,
                                     n_tasks=n_tasks,
                                     n_questions=n_questions)
        save_questions(qs, domain, suffix="")

    if mode in ("typed", "both"):
        qs = await generate_typed(domain, corpus_description, n_per_type=n_per_type)
        save_questions(qs, domain, suffix=typed_suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--mode", choices=["lightrag", "typed", "both", "ultradomain"],
                        default="lightrag",
                        help="lightrag=5×5×5 protocol, typed=motif-stratified, "
                             "both=save both, ultradomain=use dataset questions directly")
    parser.add_argument("--no-corpus", action="store_true",
                        help="Skip reading the corpus (old behaviour — use domain name only)")
    # Size controls — use small values (e.g. 3) for quick smoke tests
    parser.add_argument("--n-per-type", type=int, default=N_PER_TYPE,
                        help="Questions per motif type in typed mode (default 30)")
    parser.add_argument("--n-personas", type=int, default=5,
                        help="Personas in LightRAG 5×5×5 cascade (default 5)")
    parser.add_argument("--n-tasks", type=int, default=5,
                        help="Tasks per persona in LightRAG cascade (default 5)")
    parser.add_argument("--n-questions", type=int, default=5,
                        help="Questions per task in LightRAG cascade (default 5)")
    parser.add_argument("--typed-suffix", default="_typed",
                        help="Output filename suffix for typed mode (default _typed). "
                             "Use e.g. _typed_balanced to avoid overwriting the existing set.")
    args = parser.parse_args()
    asyncio.run(run(
        args.domain, args.mode,
        use_corpus=not args.no_corpus,
        n_per_type=args.n_per_type,
        n_personas=args.n_personas,
        n_tasks=args.n_tasks,
        n_questions=args.n_questions,
        typed_suffix=args.typed_suffix,
    ))


if __name__ == "__main__":
    main()
