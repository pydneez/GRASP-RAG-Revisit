require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });

const express = require('express');
const path    = require('path');
const fs      = require('fs');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Static dashboard files
app.use(express.static(path.join(__dirname, 'public')));

// Paper figure PDFs
app.use('/paper-assets', express.static(
  path.join(__dirname, '..', 'paper', 'assets')
));

// Raw experiment data files (read-only, served as-is for the "view source" buttons)
function safeServe(baseDir) {
  return (req, res) => {
    const file = path.basename(req.params.file); // no path traversal
    const full = path.join(baseDir, file);
    if (!fs.existsSync(full)) return res.status(404).json({ error: 'not found' });
    const ext = path.extname(file).toLowerCase();
    const ct  = ext === '.json' ? 'application/json' :
                ext === '.jsonl' ? 'text/plain' :
                ext === '.py'   ? 'text/plain' : 'text/plain';
    res.setHeader('Content-Type', ct + '; charset=utf-8');
    res.sendFile(full);
  };
}

const DATA = path.join(__dirname, '..', 'data');
app.get('/raw/results/:file',   safeServe(path.join(DATA, 'results')));
app.get('/raw/answers/:file',   safeServe(path.join(DATA, 'answers')));
app.get('/raw/questions/:file', safeServe(path.join(DATA, 'questions')));
app.get('/raw/source/:file',    safeServe(path.join(__dirname, '..', 'grasp_rag')));

// ── /api/classify ─────────────────────────────────────────────────────────────
// Uses the actual GRASP-RAG classifier prompt with top entity names from the
// pre-sampled graph as candidate nodes.
app.post('/api/classify', async (req, res) => {
  const { question, domain = 'agriculture' } = req.body || {};
  if (!question || !question.trim()) {
    return res.status(400).json({ error: 'question is required' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === 'sk-...') {
    return res.status(500).json({ error: 'OPENAI_API_KEY not configured in .env' });
  }

  // Load candidate entity names from the pre-sampled graph
  let candidates = [];
  try {
    const graphPath = path.join(__dirname, 'public', 'data', `graph_${domain}.json`);
    const graphData = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
    // Top 40 entities by degree as candidates (mirrors what vector search returns)
    candidates = graphData.nodes.slice(0, 40).map(n => n.label);
  } catch (_) {
    candidates = [];
  }

  // The exact prompt used by GRASP-RAG's QueryClassifier
  const candidateStr = candidates.length
    ? candidates.map(c => `- ${c}`).join('\n')
    : '(none available)';

  const classifyPrompt = `Analyze this query for a knowledge graph retrieval system.

Query: ${question}

Candidate graph nodes (real names from the knowledge graph):
${candidateStr}

Identify:
1. query_type — choose exactly one:
   - "single_entity": asks about one specific named thing (definition, description, properties)
   - "bridge": asks about the relationship, connection, or causation between exactly 2 named entities
   - "multi_entity": asks to compare, integrate, or reason across 3 or more named entities
   - "broad": general question with no specific named entity focus

2. anchors — pick the most relevant node names from the candidate list above that the query
   is asking about. You MUST use EXACT names from the candidate list only. Do NOT invent
   or paraphrase names. Empty list if broad.

Respond ONLY in JSON, no other text:
{"query_type": "...", "anchors": ["exact node name 1", "exact node name 2"]}`;

  // Also request a step-by-step reasoning explanation
  const reasoningPrompt = `You are analyzing a query for the GRASP-RAG system.

Query: "${question}"
Classification result: this is a {TYPE} query with anchors: {ANCHORS}

Explain in 2-3 sentences WHY this classification is correct and what GRASP-RAG will do:
- Why it is this type (not the others)
- Which graph motif will be used (ego-network / bridge path / Steiner tree / PPR diffusion)
- What information the system will retrieve from the knowledge graph`;

  try {
    // Step 1: classify
    const classifyResp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: classifyPrompt }],
        temperature: 0,
        max_tokens: 200,
      }),
    });

    if (!classifyResp.ok) {
      const err = await classifyResp.text();
      return res.status(502).json({ error: `OpenAI error: ${err.slice(0, 200)}` });
    }

    const classifyJson = await classifyResp.json();
    const rawText = classifyJson.choices?.[0]?.message?.content || '{}';

    let queryType = 'broad';
    let anchors   = [];
    try {
      const m = rawText.match(/\{.*\}/s);
      if (m) {
        const parsed = JSON.parse(m[0]);
        queryType = parsed.query_type || 'broad';
        anchors   = parsed.anchors    || [];
      }
    } catch (_) {}

    // Step 2: reasoning explanation
    const reasonFilled = reasoningPrompt
      .replace('{TYPE}',    queryType)
      .replace('{ANCHORS}', anchors.length ? anchors.join(', ') : 'none');

    const reasonResp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: reasonFilled }],
        temperature: 0.3,
        max_tokens: 300,
      }),
    });

    let reasoning = '';
    if (reasonResp.ok) {
      const reasonJson = await reasonResp.json();
      reasoning = reasonJson.choices?.[0]?.message?.content || '';
    }

    return res.json({
      question,
      domain,
      query_type: queryType,
      anchors,
      reasoning,
      candidates_used: candidates.slice(0, 10), // first 10 for display
      motif: MOTIFS[queryType] || MOTIFS.broad,
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

const MOTIFS = {
  single_entity: {
    name: 'Ego-Network',
    description: 'Retrieves the 1-hop neighbourhood of the anchor entity — its description, properties, and direct connections to other nodes.',
    algorithm: 'BFS depth-1 from anchor node',
    context_format: 'Entity Profile with connected nodes and source passages',
    color: '#2563eb',
  },
  bridge: {
    name: 'Bridge Path',
    description: 'Finds the shortest path(s) between two anchor entities in the knowledge graph. The intermediate nodes on this path are the "bridge" entities that connect the two anchors.',
    algorithm: 'Bidirectional shortest-path search (BFS from both anchors)',
    context_format: 'Step-by-step hop chain: [Entity] → [Connection] → [Entity] → …',
    color: '#ea580c',
  },
  multi_entity: {
    name: 'Steiner Tree',
    description: 'Finds a minimum-cost connected subgraph spanning all anchor entities. Steiner intermediate nodes that connect the anchors are discovered and included.',
    algorithm: 'Greedy Steiner tree approximation on the knowledge graph',
    context_format: 'Per-entity profiles + connection map between all entities',
    color: '#16a34a',
  },
  broad: {
    name: 'PPR Diffusion',
    description: 'Runs Personalized PageRank seeded at query-relevant nodes, retrieving the highest-scoring nodes across the graph. Falls back to LightRAG hybrid mode.',
    algorithm: 'Personalized PageRank (α=0.85) + LightRAG hybrid fallback',
    context_format: 'Top-k high-PPR nodes with passages (no fixed structure)',
    color: '#7c3aed',
  },
};

app.listen(PORT, () => {
  console.log(`GRASP-RAG Dashboard → http://localhost:${PORT}`);
  if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY === 'sk-...') {
    console.log('  WARN: OPENAI_API_KEY not set — /api/classify will not work');
  }
});
