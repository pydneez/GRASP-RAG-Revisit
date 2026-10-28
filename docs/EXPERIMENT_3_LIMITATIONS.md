# Experiment 3 — Known Limitations and Future Improvements (MASHQA)

Source material for the paper's Limitations / Future Work section. Everything
below is labelled by evidence strength: **[measured]** = computed over the full
MASHQA graph or the full pilot; **[observed]** = verified on a specific worked
example; **[reasoned]** = inferred from known structure, not directly measured
here.

All measurements are against `data/working_dir/mashqa/` (521 articles, 5,772
entities, 8,224 relationships) and a 30-question answer-quality pilot run
that preceded the full 1,000-question run (`data/e3_data/answer_quality.jsonl`).

---

## 1. Corpus structure: many short independent documents, not one coherent corpus

GRASP-RAG's graph construction inherits LightRAG's assumption that the corpus is
one coherent body of text, where merging repeated mentions of an entity across
the corpus is semantically correct — two mentions of the same entity in
different chapters of one book generally *are* the same referent in the same
context.

MASHQA violates that assumption. It is 521 independent WebMD articles on
mutually unrelated topics, and every question is grounded in exactly one of them
(`doc_id` in `questions.jsonl`).

- **[measured]** 77.0% of articles (401/521) are a single LightRAG chunk;
  mean 1.25 chunks/article (652 chunks over 521 articles).
- **[measured]** UltraDomain, by contrast, supplies the entire source book as
  one document (median 590 KB health / 228 KB legal), which at the same
  1200-token chunk size is on the order of 100+ chunks per document.

**Consequence.** In MASHQA the chunk is effectively the whole article, so
retrieval is all-or-nothing: retrieving the right chunk delivers the complete
evidence, and retrieving the wrong chunk delivers content from an entirely
unrelated topic. There is no partial-credit regime in between, and no
topically-adjacent fallback, because neighbouring chunks belong to different
articles about different diseases.

**Deliberately not mitigated.** Chunk size was held identical to E1/E2 (1200
tokens, 100 overlap) as a control variable, so that E3 remains a test of the
same configuration rather than a re-tuned one. Reducing chunk size for
short-article corpora is listed under Future Improvements below.

---

## 2. Entity merging produces cross-document hub nodes

Because entity resolution is corpus-wide and document-blind, a generic entity
name accumulates chunk references from every article that mentions it,
regardless of topic.

- **[measured]** Mean 1.83 chunks per entity, maximum 166.
- **[measured]** 100% of chunks (652/652) are linked by more than one entity;
  mean fan-in 16.2 entities per chunk, maximum 43.
- **[observed]** The entity `Child` links to 17 chunks drawn from 17 different
  articles, spanning special-education law, chronic fatigue syndrome, diaper
  rash, HIV, dehydration, and paediatric emergencies — topics with no mutual
  relationship beyond the word "child."
- **[observed]** `Swimming Pool` and `Pool` exist as two separate, unmerged
  nodes despite referring to the same concept, each linked to a different
  single article — so anchor selection between near-synonymous node names
  determines which article's evidence becomes reachable.

**Consequence.** Anchoring on a correct-sounding generic entity does not imply
retrieving the correct article. The node carries a mixture of unrelated
documents, and nothing in the node's own representation records which document
the current question concerns.

---

## 3. Context assembly discards available document provenance

Every chunk record in `kv_store_text_chunks.json` carries a `full_doc_id`
identifying its source article. This field is never consulted during context
assembly.

- **[measured]** `_node_chunks()` (`grasp_rag/grasp_rag.py:507-513`) iterates a
  node's `chunk_ids` with no document filter, no relevance ranking, and no
  ordering criterion — chunks are consumed in stored order, which reflects
  indexing/insertion sequence rather than relevance to the query.
- **[measured]** `_multi_entity_context()` (`grasp_rag.py:827`) displays the
  first three chunks per entity (`chunks[:3]`). "First three" is therefore an
  artefact of extraction order, not a top-3 by any relevance criterion.
- **[measured]** Passages are emitted to the generator as bare `[Passage]`
  blocks with no source attribution, so the generator cannot distinguish a
  passage drawn from the question's own article from one drawn from an
  unrelated article. (LightRAG's own naive/hybrid answer formatting does emit
  document references; GRASP-RAG's structured-context path does not.)

---

## 4. Defect: deduplication is applied over the full chunk list, display is truncated afterwards

This is a concrete implementation defect rather than a design trade-off, and it
converts the dilution described above into total evidence loss.

**Behaviour.** `_multi_entity_context()` creates one shared `seen` set
(`grasp_rag.py:807`) for the whole context-building pass and calls
`_node_chunks(node, seen)` for each anchor entity in turn
(`grasp_rag.py:818-821`). `_node_chunks()` walks the entity's **entire**
`chunk_ids` list, adding every not-yet-seen chunk to `seen`
(`grasp_rag.py:511`) as it collects it. Only afterwards does the caller
truncate the result for display to the first three (`grasp_rag.py:827`).

Deduplication is therefore applied over all N of an entity's chunks, while
display is limited to 3. Any chunk at position 4..N is marked consumed but never
shown — and is consequently unavailable to every entity processed later in the
same pass.

**Failure condition.** The defect fires when a chunk is linked to two anchor
entities, the earlier-processed entity does not display that chunk within its
own first three, and the later entity would have displayed it. Independent,
non-overlapping entities are unaffected.

**Worked example [observed].** For *"How can you help your child avoid swimming
pool and water parks germs?"* (qid `fbc2239a...`, anchors `Child`,
`Swimming Pool`, `Water Park`, `Germs`, routed `multi_entity`):

1. The question's own source article is `doc_d132fc059a43`; its chunk
   `doc_d132fc059a43-chunk-001` contains the gold evidence, including the
   "chlorination does not kill all bacteria" fact.
2. That chunk is correctly linked to the `Swimming Pool` node — which has
   exactly one linked chunk, this one.
3. `Child` is processed first and links to 17 chunks. The correct chunk sits at
   position 9 of 17 — beyond the display cut-off, so it is never shown under
   `Child`, but it *is* added to `seen`.
4. `Swimming Pool` is processed next, finds its only chunk already in `seen`,
   and contributes nothing.
5. **Verified outcome:** the string "chlorina" does not appear anywhere in the
   68,075-character context actually passed to the generator. The generated
   answer explicitly states that the context "does not give pool-specific or
   water-park-specific cleaning steps," and all three judges scored it below the
   NaiveRAG answer on all five dimensions, citing the missing chlorination fact
   by name.

**Severity is amplified by corpus structure.**

- **[measured]** 75.5% of entities (4,357/5,772) have exactly one linked chunk,
  so for three quarters of entities a single collision removes 100% of their
  evidence with no fallback.
- **[measured]** 100% of chunks are multi-entity linked, so the collision
  precondition is satisfied corpus-wide rather than rarely.
- **[reasoned]** The same defect exists for UltraDomain, but is materially
  less damaging there: entities in a single large book typically link to many
  chunks, so losing one to an earlier anchor degrades passage quality rather
  than eliminating the evidence, and the substitute chunks remain topically
  coherent because they come from the same document.

**Fix direction.** Truncate before marking rather than marking before
truncating — i.e. have `_node_chunks()` stop once it has collected the number of
chunks that will actually be displayed, so an entity claims only what it shows.
Note `_node_chunks()` is also called from `_build_flat_context()`
(`grasp_rag.py:641`) under different truncation semantics, so the limit should
be passed by the caller rather than hard-coded.

---

## 5. Fallback chunk retrieval underperforms a plain dense baseline on the same store

When entity-linked passages are exhausted, `_secondary_chunks()`
(`grasp_rag.py:524-538`) fills the remaining budget with a vector search over
the full chunk corpus — architecturally similar to NaiveRAG's retrieval.

- **[observed]** On the worked example above, this fallback also failed to
  surface the correct article, returning instead chunks on bug repellent,
  bedwetting, ear infections, diaper rash, dehydration, E. coli, HIV and
  paediatric emergencies. NaiveRAG, querying the **same** underlying chunk
  vector store (`vdb_chunks.json`, same `working_dir`), did retrieve the correct
  article and answered with the gold facts.
- **[reasoned]** The likely mechanism is that `_secondary_chunks()` is invoked
  with only the leftover budget (`_CHUNK_LIMIT` minus chunks already consumed)
  and is therefore truncated before reaching the correctly-ranked chunk. Not
  confirmed; worth isolating before citing as a cause.

---

## 6. Classifier instability across repeated calls

Independent of retrieval, the query classifier is not self-consistent.

- **[measured]** Re-classifying the same 30 pilot questions a second time
  (once during the sampling pre-pass, once during the real generation run)
  produced a different predicted type for **11 of 30 questions (37%)**.
- **[measured]** The `broad` class is least stable: only 2 of 8 questions
  sampled as `broad` were still labelled `broad` at generation time; the rest
  moved to `multi_entity` or `single_entity`.

**Consequence.** Any per-predicted-type breakdown is reported against a label
that is itself unstable, and a sample stratified on a pre-pass label will not
retain that balance in the final results. Self-inconsistency requires no
ground truth to demonstrate.

---

## Future improvements

Ordered roughly by expected impact per unit of effort.

1. **Truncate-before-mark in `_node_chunks()`** — removes the defect in §4; a
   small, local change with no change to the dedup concept, which is itself
   sound (mean fan-in 16.2 means undeduplicated contexts would repeat the same
   passage several times and exhaust the budget).
2. **Document-scoped or document-aware chunk selection** — `full_doc_id` is
   already present; ranking or filtering an entity's chunks by agreement with
   the other anchors' documents would prefer the article the anchors jointly
   point at.
3. **Relevance-ranked rather than insertion-ordered chunk selection** — replace
   `chunks[:3]` with a ranking against the query.
4. **Source labelling in the generated context** — emit the source document
   alongside each passage so the generator can weigh provenance, as LightRAG's
   own naive/hybrid formatting already does.
5. **Entity-name normalisation** — `Swimming Pool` / `Pool` remaining unmerged
   suggests near-synonymous node names fragment evidence across nodes.
6. **Chunk size tuned to short-article corpora** — deliberately excluded from
   E3 as a control variable (§1); appropriate as a follow-up study rather than a
   mid-experiment change.

**Methodological caution.** Items 1–5 alter GRASP-RAG's behaviour. E3 is framed
as external validation of the *same* pipeline used in E1/E2, so applying any of
them invalidates that framing unless E1/E2 are re-run on the modified system.
The cheaper and more defensible option is to report E3 against the unmodified
system and present these as findings with the fixes as future work.
