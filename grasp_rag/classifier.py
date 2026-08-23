import json
import re

CLASSIFY_PROMPT = """Analyze this query for a knowledge graph retrieval system.

Query: {query}

Candidate graph nodes (real names from the knowledge graph):
{candidates}

Identify:
1. query_type — choose exactly one:
   - "single_entity": asks about one specific named thing (definition, description, properties)
   - "bridge": asks about the relationship, connection, or causation between exactly 2 named entities
   - "multi_entity": asks to compare, integrate, or reason across 3 or more named entities
   - "broad": general question with no specific named entity focus

2. anchors — pick the named entities the query is genuinely asking ABOUT (the subject(s) of
   inquiry), not domain/context phrases. You MUST use EXACT names from the candidate list
   only. Do NOT invent or paraphrase names. Empty list if broad.

   IMPORTANT: domain or context phrases like "in agriculture", "in legal matters", "in
   application development" are NOT anchors — they describe the setting, not a second entity.
   A "bridge" query requires exactly 2 distinct named entities where the question asks about
   the relationship or connection between them specifically. If only one entity is the subject
   of inquiry, classify as "single_entity" even if a domain word appears in the question.

Respond ONLY in JSON, no other text:
{{"query_type": "...", "anchors": ["exact node name 1", "exact node name 2"]}}"""


class QueryClassifier:
    def __init__(self, llm_func):
        self._llm = llm_func

    async def classify(self, query: str, candidates: list[str] | None = None) -> tuple[str, list[str]]:
        candidate_str = "\n".join(f"- {c}" for c in candidates) if candidates else "(none provided)"
        prompt = CLASSIFY_PROMPT.format(query=query, candidates=candidate_str)
        try:
            result = await self._llm(prompt)
            m = re.search(r"\{.*\}", result, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                qtype = data.get("query_type", "broad")
                anchors = [str(a) for a in data.get("anchors", [])]
                return qtype, anchors
        except Exception:
            pass
        return "broad", []
