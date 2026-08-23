/* ═══════════════════════════════════════════════════════════════════════════
   GRASP-RAG Dashboard — app.js
   Loads all static JSON, then renders each section.
══════════════════════════════════════════════════════════════════════════ */

// ── Constants ────────────────────────────────────────────────────────────────
const METHODS = ['grasp_rag', 'naive_rag', 'lightrag', 'pathrag'];
const METHOD_LABELS = {
  grasp_rag: 'GRASP-RAG', naive_rag: 'NaiveRAG',
  lightrag: 'LightRAG', pathrag: 'PathRAG',
};
const METHOD_COLORS = {
  grasp_rag: '#2563eb', naive_rag: '#6b7280',
  lightrag:  '#16a34a', pathrag:   '#ea580c',
};
const DIMS     = ['comprehensiveness','diversity','logicality','relevance','coherence'];
const DOMAINS  = ['agriculture','cs','legal','mix'];
const DOMAIN_LABELS = { agriculture:'Agriculture', cs:'CS', legal:'Legal', mix:'Mix' };

const ENTITY_COLORS = {
  concept:      '#3b82f6',
  method:       '#f59e0b',
  organisation: '#10b981',
  organization: '#10b981',
  person:       '#ec4899',
  technology:   '#8b5cf6',
  event:        '#ef4444',
  location:     '#06b6d4',
  unknown:      '#94a3b8',
};

// ── Global state ─────────────────────────────────────────────────────────────
let berData     = null;
let judgeData   = null;
let answersData = {};   // domain → question map
let graphData   = {};   // domain → {stats, nodes, edges}

let berChart      = null;
let judgeSimChart = null;
let judgeTypeChart= null;

let currentBerDomain   = 'agriculture';
let currentGraphDomain = 'agriculture';
let graphSimulation    = null;

// ── Fetch helper ─────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
async function init() {
  try {
    [berData, judgeData] = await Promise.all([
      fetchJSON('data/ber_results.json'),
      fetchJSON('data/judge_results.json'),
    ]);

    // Load agriculture answers immediately; others lazy-loaded
    answersData['agriculture'] = await fetchJSON('data/answers_agriculture.json')
      .catch(() => ({}));

    renderOverviewStats();
    setupBerTabs();
    renderBerSection(currentBerDomain);
    updateBerSrcBtn(currentBerDomain);
    renderJudgeSection();
    setupGraphTabs();
    loadGraph('agriculture');

  } catch (err) {
    console.error('Init error:', err);
    document.body.insertAdjacentHTML('afterbegin',
      `<div class="bg-red-100 text-red-700 p-4 text-sm">
        Error loading data: ${err.message}<br>
        Make sure you ran <code>python3 dashboard/preprocess.py</code> first.
      </div>`
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION A — Overview stats
// ═══════════════════════════════════════════════════════════════════════════
function renderOverviewStats() {
  const container = document.getElementById('overview-stats');
  if (!berData) return;

  // Compute 4-domain averages
  const avg = {};
  METHODS.forEach(m => {
    let sum = 0, n = 0;
    DOMAINS.forEach(d => {
      if (berData[d]?.results?.[m]) {
        sum += berData[d].results[m].ber; n++;
      }
    });
    avg[m] = n ? (sum / n).toFixed(1) : '—';
  });

  const cards = [
    { value: '87.5–100%', label: 'GRASP-RAG BER range (4 domains)', color: 'text-blue-600' },
    { value: `${avg.grasp_rag}%`, label: 'GRASP-RAG 4-domain avg BER', color: 'text-blue-600' },
    { value: '78%',  label: 'LLM-judge loss rate vs NaiveRAG', color: 'text-red-600' },
    { value: '4',    label: 'Knowledge graph domains tested',  color: 'text-green-600' },
  ];

  container.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="value ${c.color}">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION B — BER Results
// ═══════════════════════════════════════════════════════════════════════════
function setupBerTabs() {
  document.querySelectorAll('#ber-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#ber-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentBerDomain = btn.dataset.domain;
      renderBerSection(currentBerDomain);
      updateBerSrcBtn(currentBerDomain);
    });
  });
}

function renderBerSection(domain) {
  const data = berData?.[domain];
  if (!data) return;

  // Bar chart
  const labels  = METHODS.map(m => METHOD_LABELS[m]);
  const values  = METHODS.map(m => data.results[m]?.ber ?? 0);
  const bgColors = METHODS.map(m => METHOD_COLORS[m] + 'cc');
  const bdColors = METHODS.map(m => METHOD_COLORS[m]);

  if (berChart) berChart.destroy();
  berChart = new Chart(document.getElementById('ber-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'BER (%)',
        data: values,
        backgroundColor: bgColors,
        borderColor: bdColors,
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0, max: 100,
          title: { display: true, text: 'BER (%)' },
          grid: { color: '#f1f5f9' },
        },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `Bridging Entity Recall — ${DOMAIN_LABELS[domain]}`,
          font: { size: 13, weight: '600' },
          color: '#1e293b',
        },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.raw.toFixed(1)}%  (${data.results[METHODS[ctx.dataIndex]]?.hit ?? '?'} / ${data.results[METHODS[ctx.dataIndex]]?.total ?? '?'})`,
          },
        },
      },
    },
  });

  // Stats table
  const tbody = document.getElementById('ber-table-body');
  tbody.innerHTML = METHODS.map(m => {
    const r = data.results[m];
    if (!r) return '';
    const ber = r.ber.toFixed(1);
    const isGrasp = m === 'grasp_rag';
    return `<tr>
      <td class="font-medium ${isGrasp ? 'text-blue-700' : ''}">${METHOD_LABELS[m]}</td>
      <td class="text-right font-mono ${isGrasp ? 'font-bold text-blue-700' : ''}">${ber}%</td>
      <td class="text-right text-slate-500">${r.hit}</td>
      <td class="text-right text-slate-400">${r.total}</td>
    </tr>`;
  }).join('');

  // Miss explorer (GRASP-RAG misses)
  const graspMisses = data.misses?.grasp_rag ?? [];
  const missContainer = document.getElementById('ber-miss-container');
  if (graspMisses.length === 0) {
    missContainer.innerHTML = '<p class="text-xs text-slate-400 italic">No misses in this domain (100% BER).</p>';
  } else {
    missContainer.innerHTML = graspMisses.map(m => `
      <div class="miss-card">
        <div class="miss-q">${escapeHtml(m.question)}</div>
        <div class="miss-meta">Bridge: <strong>${escapeHtml(m.ground_truth || m.bridge || '?')}</strong></div>
        <div class="mt-1 text-slate-500">${escapeHtml((m.answer_snippet || '').slice(0, 180))}...</div>
      </div>
    `).join('');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION C — LLM-as-Judge
// ═══════════════════════════════════════════════════════════════════════════
function renderJudgeSection() {
  if (!judgeData) return;
  const { summary, records } = judgeData;

  // Overall totals
  const tot = summary.overall;
  const total = (tot.Win||0) + (tot.Loss||0) + (tot.Tie||0);
  document.getElementById('judge-totals').innerHTML = [
    { label: 'Wins',   val: tot.Win||0,  color: 'text-green-600' },
    { label: 'Losses', val: tot.Loss||0, color: 'text-red-600'   },
    { label: 'Ties',   val: tot.Tie||0,  color: 'text-slate-500' },
  ].map(c => `
    <div class="stat-card text-center">
      <div class="value ${c.color}">${c.val}</div>
      <div class="label">${c.label} / ${total} dim-outcomes</div>
    </div>
  `).join('');

  // By-dimension chart
  const dimLabels = DIMS.map(d => d.charAt(0).toUpperCase() + d.slice(1, 5) + '.');
  const dimWin  = DIMS.map(d => summary.by_dim[d]?.Win  ?? 0);
  const dimLoss = DIMS.map(d => summary.by_dim[d]?.Loss ?? 0);
  const dimTie  = DIMS.map(d => summary.by_dim[d]?.Tie  ?? 0);

  if (judgeSimChart) judgeSimChart.destroy();
  judgeSimChart = new Chart(document.getElementById('judge-dim-chart'), {
    type: 'bar',
    data: {
      labels: dimLabels,
      datasets: [
        { label: 'Win',  data: dimWin,  backgroundColor: '#86efac', borderColor: '#16a34a', borderWidth:1 },
        { label: 'Tie',  data: dimTie,  backgroundColor: '#e2e8f0', borderColor: '#94a3b8', borderWidth:1 },
        { label: 'Loss', data: dimLoss, backgroundColor: '#fca5a5', borderColor: '#dc2626', borderWidth:1 },
      ],
    },
    options: stackedBarOptions('GRASP-RAG vs NaiveRAG (per dimension)'),
  });

  // By-type chart
  const qtypes = ['single_entity','bridge','multi_entity','broad'];
  const typeLabels = qtypes.map(t => t.replace('_',' '));
  const typeWin  = qtypes.map(t => summary.by_type[t]?.Win  ?? 0);
  const typeLoss = qtypes.map(t => summary.by_type[t]?.Loss ?? 0);
  const typeTie  = qtypes.map(t => summary.by_type[t]?.Tie  ?? 0);

  if (judgeTypeChart) judgeTypeChart.destroy();
  judgeTypeChart = new Chart(document.getElementById('judge-type-chart'), {
    type: 'bar',
    data: {
      labels: typeLabels,
      datasets: [
        { label: 'Win',  data: typeWin,  backgroundColor: '#86efac', borderColor: '#16a34a', borderWidth:1 },
        { label: 'Tie',  data: typeTie,  backgroundColor: '#e2e8f0', borderColor: '#94a3b8', borderWidth:1 },
        { label: 'Loss', data: typeLoss, backgroundColor: '#fca5a5', borderColor: '#dc2626', borderWidth:1 },
      ],
    },
    options: stackedBarOptions('GRASP-RAG vs NaiveRAG (by query type)'),
  });

  // Answer explorer
  renderJudgeExplorer(records);
}

function stackedBarOptions(title) {
  return {
    responsive: true,
    scales: {
      x: { stacked: true, grid: { display: false } },
      y: { stacked: true, grid: { color: '#f1f5f9' } },
    },
    plugins: {
      legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
      title: { display: true, text: title, font: { size: 12, weight: '600' }, color: '#1e293b' },
    },
  };
}

// Answer explorer ─────────────────────────────────────────────────────────────
let judgeRecordsAll = [];
let selectedJudgeQ  = null;

function renderJudgeExplorer(records) {
  // Deduplicate: keep unique questions with their first outcome record
  const seen = new Map();
  records.forEach(r => { if (!seen.has(r.question)) seen.set(r.question, r); });
  judgeRecordsAll = [...seen.values()];

  renderJudgeList(judgeRecordsAll);

  document.getElementById('judge-search').addEventListener('input', filterJudgeList);
  document.getElementById('judge-type-filter').addEventListener('change', filterJudgeList);
}

function filterJudgeList() {
  const search = document.getElementById('judge-search').value.toLowerCase();
  const type   = document.getElementById('judge-type-filter').value;
  const filtered = judgeRecordsAll.filter(r =>
    (!search || r.question.toLowerCase().includes(search)) &&
    (!type   || r.query_type === type)
  );
  renderJudgeList(filtered);
}

function renderJudgeList(records) {
  const list = document.getElementById('judge-question-list');
  list.innerHTML = records.map((r, i) => {
    const overallOutcome = dominantOutcome(r.outcome);
    return `
      <div class="judge-q-item ${r.question === selectedJudgeQ ? 'selected' : ''}"
           data-q="${escapeHtml(r.question)}">
        <span class="qtype-badge qtype-${r.query_type}">${r.query_type?.replace('_',' ')}</span>
        <span class="outcome-badge outcome-${overallOutcome}">${overallOutcome}</span><br>
        ${escapeHtml(r.question)}
      </div>
    `;
  }).join('');

  list.querySelectorAll('.judge-q-item').forEach(el => {
    el.addEventListener('click', () => {
      selectedJudgeQ = el.dataset.q;
      showJudgeAnswer(el.dataset.q);
      list.querySelectorAll('.judge-q-item').forEach(e => e.classList.remove('selected'));
      el.classList.add('selected');
    });
  });
}

function dominantOutcome(outcome) {
  if (!outcome) return 'Tie';
  const counts = { Win: 0, Loss: 0, Tie: 0 };
  Object.values(outcome).forEach(v => { if (counts[v] !== undefined) counts[v]++; });
  if (counts.Loss > counts.Win) return 'Loss';
  if (counts.Win  > counts.Loss) return 'Win';
  return 'Tie';
}

async function showJudgeAnswer(question) {
  // Ensure agriculture answers loaded
  if (!answersData['agriculture']) {
    answersData['agriculture'] = await fetchJSON('data/answers_agriculture.json').catch(() => ({}));
  }
  const qaMap = answersData['agriculture'];
  const qa    = qaMap[question];

  const record = judgeRecordsAll.find(r => r.question === question);
  const outcomeHTML = record ? DIMS.map(d => {
    const o = record.outcome?.[d] || 'Tie';
    return `<span class="outcome-badge outcome-${o} mr-1">${d.slice(0,4)}: ${o}</span>`;
  }).join('') : '';

  const panel = document.getElementById('judge-answer-panel');

  if (!qa) {
    panel.innerHTML = `<div class="text-slate-400 text-sm italic py-8 text-center">
      Answer data not found for this question.
    </div>`;
    return;
  }

  const gt = qa.ground_truth || '';
  const graspAns  = qa.methods?.grasp_rag  || '(no answer)';
  const naiveAns  = qa.methods?.naive_rag  || '(no answer)';

  panel.innerHTML = `
    <div class="mb-3">
      <div class="font-semibold text-slate-700 text-sm">${escapeHtml(question)}</div>
      <div class="text-xs text-slate-400 mt-1">${outcomeHTML}</div>
      ${gt ? `<div class="text-xs mt-1"><span class="font-medium text-orange-600">Bridge entity:</span> <strong>${escapeHtml(gt)}</strong></div>` : ''}
    </div>
    <div class="grid grid-cols-2 gap-3">
      <div>
        <div class="answer-col">
          <div class="method-header text-blue-600">GRASP-RAG</div>
          ${highlightEntity(graspAns, gt)}
        </div>
      </div>
      <div>
        <div class="answer-col">
          <div class="method-header text-slate-400">NaiveRAG</div>
          ${highlightEntity(naiveAns, gt)}
        </div>
      </div>
    </div>
  `;
}

function highlightEntity(text, entity) {
  if (!entity || !text) return escapeHtml(text);
  const escaped = escapeHtml(text);
  const escapedEntity = escapeHtml(entity);
  const re = new RegExp(`(${escapedEntity.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`, 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION D — Knowledge Graph (D3 force layout)
// ═══════════════════════════════════════════════════════════════════════════
function setupGraphTabs() {
  document.querySelectorAll('#graph-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#graph-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGraphDomain = btn.dataset.domain;
      loadGraph(currentGraphDomain);
    });
  });

  document.getElementById('graph-node-count').addEventListener('change', () => {
    renderGraph(currentGraphDomain);
  });
}

async function loadGraph(domain) {
  if (!graphData[domain]) {
    try {
      graphData[domain] = await fetchJSON(`data/graph_${domain}.json`);
    } catch (e) {
      document.getElementById('graph-note').textContent =
        `Graph data not found for ${domain}. Run preprocess.py.`;
      return;
    }
  }
  renderGraphStats(domain);
  renderGraph(domain);
}

function renderGraphStats(domain) {
  const s = graphData[domain]?.stats;
  if (!s) return;
  const density = s.total_edges / Math.max(s.total_nodes * (s.total_nodes - 1) / 2, 1);

  const berRow   = berData?.[domain]?.results;
  const graspBer = berRow?.grasp_rag?.ber?.toFixed(1) ?? '—';

  document.getElementById('graph-stats').innerHTML = [
    { label: 'Total Nodes',     value: s.total_nodes.toLocaleString(),       color: 'text-blue-600'  },
    { label: 'Total Edges',     value: s.total_edges.toLocaleString(),       color: 'text-slate-700' },
    { label: 'Graph Density',   value: (density * 100).toFixed(3)+'%', color: 'text-slate-700' },
    { label: 'GRASP BER',       value: graspBer+'%',                   color: 'text-green-600' },
  ].map(c => `
    <div class="stat-card">
      <div class="value ${c.color}">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');
}

function renderGraph(domain) {
  const data = graphData[domain];
  if (!data) return;

  const { stats } = data;
  const nodeCount = parseInt(document.getElementById('graph-node-count').value, 10);

  // Slice top-N nodes (already sorted by degree desc from preprocess.py)
  const nodes = data.nodes.slice(0, nodeCount);
  const nodeSet = new Set(nodes.map(n => n.id));
  // Keep only edges where both endpoints are in the visible set
  const edges = data.edges.filter(e => nodeSet.has(e.source) && nodeSet.has(e.target));

  document.getElementById('graph-note').textContent =
    `Showing top ${nodes.length} of ${stats.total_nodes.toLocaleString()} nodes (by degree)`;

  // Clear previous simulation
  if (graphSimulation) graphSimulation.stop();
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  const el  = document.getElementById('graph-svg');
  const W   = el.clientWidth  || 700;
  const H   = el.clientHeight || 520;

  // Zoom container
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => g.attr('transform', e.transform)));

  // Degree scale for node radius
  const maxDeg = Math.max(...nodes.map(n => n.degree));
  const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([4, 18]);

  // Build link/node maps
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const links   = edges
    .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map(e => ({ ...e }));

  // Draw links
  const link = g.append('g').selectAll('line')
    .data(links).join('line')
    .attr('class', 'link');

  // Draw nodes
  const nodeG = g.append('g').selectAll('g.node')
    .data(nodes).join('g').attr('class', 'node');

  nodeG.append('circle')
    .attr('r', d => rScale(d.degree))
    .attr('fill', d => ENTITY_COLORS[d.entity_type] || ENTITY_COLORS.unknown)
    .attr('fill-opacity', 0.85)
    .attr('stroke', d => ENTITY_COLORS[d.entity_type] || ENTITY_COLORS.unknown);

  nodeG.append('text')
    .text(d => d.label.length > 14 ? d.label.slice(0, 12) + '...' : d.label)
    .attr('x', d => rScale(d.degree) + 3)
    .attr('y', 4)
    .attr('font-size', '8px')
    .style('display', d => d.degree > maxDeg * 0.15 ? 'block' : 'none');

  // Tooltip + click
  const tooltip = document.getElementById('graph-tooltip');

  nodeG
    .on('mousemove', (event, d) => {
      tooltip.innerHTML = `
        <div class="font-semibold text-slate-800">${escapeHtml(d.label)}</div>
        <div class="text-slate-500 mt-1"><span class="font-medium">${d.entity_type}</span> · degree ${d.degree}</div>
        ${d.description ? `<div class="text-slate-600 mt-1 text-xs">${escapeHtml(d.description.slice(0,120))}...</div>` : ''}
      `;
      tooltip.classList.remove('hidden');
      tooltip.style.left = (event.clientX + 14) + 'px';
      tooltip.style.top  = (event.clientY - 10) + 'px';
    })
    .on('mouseleave', () => tooltip.classList.add('hidden'))
    .on('click', (event, d) => {
      event.stopPropagation();
      showNodeDetail(d);
      // Highlight
      nodeG.select('circle').attr('stroke-width', n => n.id === d.id ? 3 : 1.5);
    });

  svg.on('click', () => {
    nodeG.select('circle').attr('stroke-width', 1.5);
    document.getElementById('graph-detail').innerHTML =
      '<span class="text-slate-400 italic">Click a node to see details.</span>';
  });

  // Force simulation
  graphSimulation = d3.forceSimulation(nodes)
    .force('link',   d3.forceLink(links).id(d => d.id).distance(60).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(d => rScale(d.degree) + 2))
    .on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeG.attr('transform', d => `translate(${d.x},${d.y})`);
    });

  // Drag
  nodeG.call(d3.drag()
    .on('start', (e, d) => { if (!e.active) graphSimulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end',   (e, d) => { if (!e.active) graphSimulation.alphaTarget(0); d.fx = null; d.fy = null; })
  );

  // Legend
  const entityTypes = [...new Set(nodes.map(n => n.entity_type))].sort();
  document.getElementById('graph-legend').innerHTML = entityTypes.map(t => `
    <div class="flex items-center gap-2">
      <div class="w-3 h-3 rounded-full flex-shrink-0"
           style="background:${ENTITY_COLORS[t] || ENTITY_COLORS.unknown}"></div>
      <span class="text-xs text-slate-600">${t}</span>
    </div>
  `).join('');
}

function showNodeDetail(d) {
  document.getElementById('graph-detail').innerHTML = `
    <div class="font-semibold text-slate-700">${escapeHtml(d.label)}</div>
    <div class="text-slate-400 mt-1">Type: <strong>${d.entity_type}</strong></div>
    <div class="text-slate-400">Degree: <strong>${d.degree}</strong></div>
    ${d.description ? `<div class="text-slate-600 mt-2">${escapeHtml(d.description)}</div>` : ''}
  `;
}

// Lazy-load answer data for other domains
async function getAnswers(domain) {
  if (!answersData[domain]) {
    answersData[domain] = await fetchJSON(`data/answers_${domain}.json`).catch(() => ({}));
  }
  return answersData[domain];
}

// ── Utility ──────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION F — Pipeline diagram + Interactive demo
// ═══════════════════════════════════════════════════════════════════════════

const PIPELINE_STEPS = [
  {
    title: 'User Query',
    desc: 'Natural-language question is entered.',
    color: '#6366f1', bg: '#eef2ff', border: '#a5b4fc',
  },
  {
    title: 'Candidate Retrieval',
    desc: 'Vector search finds the top-k most relevant entity names.',
    color: '#0891b2', bg: '#ecfeff', border: '#67e8f9',
  },
  {
    title: 'LLM Classifier',
    desc: 'GPT-4o-mini assigns one of 4 query types and identifies anchor entities.',
    color: '#7c3aed', bg: '#f5f3ff', border: '#c4b5fd',
  },
  {
    title: 'Motif Selection',
    desc: 'Query type maps to a retrieval motif: ego-network / bridge / Steiner / PPR.',
    color: '#b45309', bg: '#fffbeb', border: '#fcd34d',
  },
  {
    title: 'Graph Traversal',
    desc: 'Run the motif algorithm on the knowledge graph to extract a subgraph.',
    color: '#15803d', bg: '#f0fdf4', border: '#86efac',
  },
  {
    title: 'PGR Context',
    desc: 'Path-Guided Reasoning formats the subgraph as a step-by-step chain.',
    color: '#ea580c', bg: '#fff7ed', border: '#fdba74',
  },
  {
    title: 'LLM Answer',
    desc: 'LLM generates an answer grounded in the structured graph context.',
    color: '#2563eb', bg: '#eff6ff', border: '#93c5fd',
  },
];

const EXAMPLE_QUESTIONS = {
  agriculture: [
    'What connects Rational Grazing to Polyface Farm?',
    'What is Soil Organic Matter?',
    'Compare cover crops, crop rotation, and composting.',
    'How do regenerative agriculture practices impact climate?',
  ],
  cs: [
    'What is the relationship between Linux Kernel and IPv4?',
    'What is a Binary Search Tree?',
    'How are Machine Learning, Neural Networks, and Backpropagation related?',
    'How does the internet work?',
  ],
  legal: [
    'What connects Habeas Corpus to Due Process?',
    'What is Mens Rea?',
    'Compare civil law, criminal law, and constitutional law.',
    'What are the main principles of contract law?',
  ],
  mix: [
    'What connects Media to Communication Theory?',
    'What is Quantum Computing?',
    'How are Economics, Supply Chain, and Logistics related?',
    'What are the key trends in artificial intelligence?',
  ],
};

function renderPipelineDiagram() {
  const container = document.getElementById('pipeline-steps');
  container.innerHTML = PIPELINE_STEPS.map((s, i) => `
    <div class="pipeline-step">
      <div class="pipeline-box" style="border-color:${s.border}; background:${s.bg};">
        <div class="step-num" style="color:${s.color}">STEP ${i + 1}</div>
        <div class="step-title" style="color:${s.color}">${s.title}</div>
        <div class="step-desc">${s.desc}</div>
      </div>
    </div>
    ${i < PIPELINE_STEPS.length - 1 ? '<div class="pipeline-arrow">&#8594;</div>' : ''}
  `).join('');
}

function renderExamplePills() {
  const container = document.getElementById('demo-examples');
  const domain = document.getElementById('demo-domain').value;
  const examples = EXAMPLE_QUESTIONS[domain] || [];
  container.innerHTML = '<span class="text-xs text-slate-400 self-center mr-1">Try:</span>' +
    examples.map(q => `
      <button class="example-pill" onclick="setDemoQuestion(${JSON.stringify(q)})">${q}</button>
    `).join('');
}

function setDemoQuestion(q) {
  document.getElementById('demo-question').value = q;
}

function setupDemoSection() {
  renderPipelineDiagram();
  renderExamplePills();

  document.getElementById('demo-domain').addEventListener('change', renderExamplePills);

  document.getElementById('demo-run').addEventListener('click', runDemo);
  document.getElementById('demo-question').addEventListener('keydown', e => {
    if (e.key === 'Enter') runDemo();
  });
}

async function runDemo() {
  const question = document.getElementById('demo-question').value.trim();
  const domain   = document.getElementById('demo-domain').value;
  if (!question) return;

  const btn = document.getElementById('demo-run');
  btn.disabled = true;
  btn.textContent = 'Running...';

  const trace = document.getElementById('demo-trace');
  trace.innerHTML = '';

  // Step definitions for the trace
  const steps = [
    { id: 'step-query',      label: '1. Query Received'     },
    { id: 'step-candidates', label: '2. Candidate Entities' },
    { id: 'step-classify',   label: '3. LLM Classification' },
    { id: 'step-motif',      label: '4. Motif Selected'     },
    { id: 'step-traversal',  label: '5. Graph Traversal'    },
    { id: 'step-pgr',        label: '6. PGR Context Format' },
    { id: 'step-answer',     label: '7. Answer Generation'  },
  ];

  // Create all step cards up front
  steps.forEach(s => {
    trace.insertAdjacentHTML('beforeend', `
      <div class="trace-step" id="${s.id}">
        <div class="trace-step-header">
          <span class="step-label">${s.label}</span>
          <span class="step-status" id="${s.id}-status">waiting</span>
        </div>
        <div class="trace-step-body hidden" id="${s.id}-body"></div>
      </div>
    `);
  });

  function setStepStatus(id, status, html) {
    const el     = document.getElementById(id);
    const sEl    = document.getElementById(`${id}-status`);
    const bEl    = document.getElementById(`${id}-body`);
    el.className = `trace-step ${status}`;
    sEl.innerHTML = status === 'active' ? '<div class="spinner"></div>' :
                    status === 'done'   ? 'done' :
                    status === 'error'  ? 'error' : 'waiting';
    if (html) {
      bEl.innerHTML = html;
      bEl.classList.remove('hidden');
    }
  }

  async function pause(ms) { return new Promise(r => setTimeout(r, ms)); }

  try {
    // ── Step 1: Query received ────────────────────────────────────────────
    setStepStatus('step-query', 'active');
    await pause(300);
    setStepStatus('step-query', 'done', `
      <div class="bg-slate-50 rounded-lg px-3 py-2 border border-slate-200">
        <span class="font-medium">Query:</span> "${escapeHtml(question)}"
      </div>
      <div class="text-slate-400 text-xs mt-1">Domain: ${domain}</div>
    `);

    // ── Step 2: Candidates (from graph data) ──────────────────────────────
    setStepStatus('step-candidates', 'active');
    await pause(400);

    let candidateHtml = '';
    if (graphData[domain]) {
      const top = graphData[domain].nodes.slice(0, 10).map(n => n.label);
      candidateHtml = `
        <div class="text-slate-500 text-xs mb-1">Top entities by degree (used as classifier candidates):</div>
        <div class="flex flex-wrap gap-1">
          ${top.map(c => `<span class="bg-blue-50 text-blue-700 text-xs px-2 py-0.5 rounded-full border border-blue-100">${escapeHtml(c)}</span>`).join('')}
          <span class="text-slate-400 text-xs self-center">+ 30 more</span>
        </div>`;
    } else {
      await fetchJSON(`data/graph_${domain}.json`)
        .then(d => { graphData[domain] = d; })
        .catch(() => {});
      const top = (graphData[domain]?.nodes || []).slice(0, 10).map(n => n.label);
      candidateHtml = top.length
        ? `<div class="flex flex-wrap gap-1">
            ${top.map(c => `<span class="bg-blue-50 text-blue-700 text-xs px-2 py-0.5 rounded-full border border-blue-100">${escapeHtml(c)}</span>`).join('')}
           </div>`
        : '<span class="text-slate-400 text-xs">Graph data not loaded yet.</span>';
    }
    setStepStatus('step-candidates', 'done', candidateHtml);

    // ── Step 3: LLM classification (real API call) ─────────────────────────
    setStepStatus('step-classify', 'active');
    const resp = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, domain }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      setStepStatus('step-classify', 'error',
        `<div class="text-red-600 text-xs">API error: ${escapeHtml(err.error || 'unknown')}</div>`);
      btn.disabled = false; btn.textContent = 'Run Pipeline ▶';
      return;
    }

    const result = await resp.json();
    const { query_type, anchors, reasoning, motif } = result;

    const typeColors = {
      single_entity: 'bg-blue-50 text-blue-700 border-blue-200',
      bridge:        'bg-orange-50 text-orange-700 border-orange-200',
      multi_entity:  'bg-green-50 text-green-700 border-green-200',
      broad:         'bg-purple-50 text-purple-700 border-purple-200',
    };
    const tc = typeColors[query_type] || 'bg-slate-50 text-slate-700 border-slate-200';

    setStepStatus('step-classify', 'done', `
      <div class="flex flex-wrap items-center gap-3 mb-2">
        <span class="text-xs font-medium text-slate-500">Query type:</span>
        <span class="text-sm font-bold px-3 py-1 rounded-full border ${tc}">${query_type}</span>
        ${anchors.length ? `
          <span class="text-xs font-medium text-slate-500">Anchors:</span>
          ${anchors.map(a => `<span class="bg-slate-100 text-slate-700 text-xs px-2 py-1 rounded border border-slate-200">${escapeHtml(a)}</span>`).join('')}
        ` : '<span class="text-xs text-slate-400">No specific anchors (broad)</span>'}
      </div>
      ${reasoning ? `<div class="text-xs text-slate-600 bg-slate-50 rounded-lg p-3 border border-slate-100 leading-relaxed">${escapeHtml(reasoning)}</div>` : ''}
      <div class="text-xs text-slate-400 mt-1">Model: gpt-4o-mini — actual GRASP-RAG classifier prompt used</div>
    `);

    // ── Step 4: Motif selected ────────────────────────────────────────────
    setStepStatus('step-motif', 'active');
    await pause(500);
    setStepStatus('step-motif', 'done', `
      <div class="motif-card" style="border-color:${motif.color}20; background:${motif.color}08;">
        <div>
          <div class="font-bold text-sm" style="color:${motif.color}">${motif.name}</div>
          <div class="text-xs text-slate-600 mt-0.5">${escapeHtml(motif.description)}</div>
          <div class="text-xs text-slate-400 mt-1"><span class="font-medium">Algorithm:</span> ${escapeHtml(motif.algorithm)}</div>
        </div>
      </div>
    `);

    // ── Step 5: Graph traversal (simulated — we don't run it live) ─────────
    setStepStatus('step-traversal', 'active');
    await pause(600);

    const traversalDescriptions = {
      single_entity: anchors.length
        ? `BFS from <strong>${escapeHtml(anchors[0])}</strong> → collect all 1-hop neighbours with their edge keywords and source passages.`
        : 'BFS from the anchor entity → collect 1-hop neighbourhood.',
      bridge: anchors.length >= 2
        ? `Bidirectional shortest-path search: <strong>${escapeHtml(anchors[0])}</strong> ↔ <strong>${escapeHtml(anchors[1])}</strong>.<br>Each intermediate node on the path is a candidate bridge entity.`
        : 'Bidirectional shortest-path between the two anchor entities.',
      multi_entity: anchors.length
        ? `Greedy Steiner tree spanning: <strong>${anchors.map(a => escapeHtml(a)).join('</strong>, <strong>')}</strong>.<br>Minimum-cost subgraph connecting all anchors; intermediate Steiner nodes are discovered.`
        : 'Greedy Steiner tree connecting all anchor entities.',
      broad: 'Personalized PageRank seeded at query-relevant nodes. Top-scoring nodes across the full graph are retrieved, then LightRAG hybrid mode is used as fallback.',
    };

    setStepStatus('step-traversal', 'done', `
      <div class="text-xs text-slate-600 bg-slate-50 rounded-lg p-3 border border-slate-100">
        ${traversalDescriptions[query_type] || traversalDescriptions.broad}
      </div>
      <div class="text-xs text-slate-400 mt-1">
        (Live traversal not run in demo — would require loading the full knowledge graph)
      </div>
    `);

    // ── Step 6: PGR context format ────────────────────────────────────────
    setStepStatus('step-pgr', 'active');
    await pause(500);

    const pgrDescriptions = {
      bridge: `
        <div class="font-mono text-xs bg-slate-800 text-green-300 rounded-lg p-3 leading-relaxed">
          [Step 1 — Entity: ${escapeHtml(anchors[0] || 'Anchor A')} (type)]<br>
          &nbsp;&nbsp;Description: …<br>
          &nbsp;&nbsp;Source Passages: [Passage] …<br><br>
          [Step 2 — Connection: ${escapeHtml(anchors[0] || 'A')} → <strong class="text-yellow-300">Bridge</strong>]<br>
          &nbsp;&nbsp;Relationship keywords: …<br><br>
          [Step 3 — Entity: <strong class="text-yellow-300">Bridge Entity</strong> (type)]<br>
          &nbsp;&nbsp;Description: …<br><br>
          [Step 4 — Connection: <strong class="text-yellow-300">Bridge</strong> → ${escapeHtml(anchors[1] || 'Anchor B')}]<br>
          &nbsp;&nbsp;Relationship keywords: …<br><br>
          [Step 5 — Entity: ${escapeHtml(anchors[1] || 'Anchor B')} (type)]<br>
          &nbsp;&nbsp;Description: …
        </div>
        <div class="text-xs text-slate-400 mt-1">Bridge entity is placed at a labelled step → structurally unavoidable for the LLM.</div>`,
      single_entity: `
        <div class="font-mono text-xs bg-slate-800 text-green-300 rounded-lg p-3 leading-relaxed">
          [Entity Profile: ${escapeHtml(anchors[0] || 'Entity')}]<br>
          &nbsp;&nbsp;Type: …<br>
          &nbsp;&nbsp;Description: …<br>
          &nbsp;&nbsp;Connected to:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;- Neighbour A: keywords…<br>
          &nbsp;&nbsp;&nbsp;&nbsp;- Neighbour B: keywords…<br>
          &nbsp;&nbsp;Source Passages: [Passage] …
        </div>`,
      multi_entity: `
        <div class="font-mono text-xs bg-slate-800 text-green-300 rounded-lg p-3 leading-relaxed">
          [Entity 1: ${escapeHtml(anchors[0] || 'A')}]  Description + Passages<br>
          [Entity 2: ${escapeHtml(anchors[1] || 'B')}]  Description + Passages<br>
          ${anchors[2] ? `[Entity 3: ${escapeHtml(anchors[2])}]  Description + Passages<br>` : ''}
          [Connections between entities]<br>
          &nbsp;&nbsp;- A ↔ B: keywords…<br>
          [Connection Source Passages] …
        </div>`,
      broad: `
        <div class="font-mono text-xs bg-slate-800 text-green-300 rounded-lg p-3 leading-relaxed">
          ← Falls back to LightRAG hybrid mode →<br><br>
          Top-k passages from vector search + graph<br>
          No fixed step structure (PPR diffusion output)
        </div>`,
    };

    setStepStatus('step-pgr', 'done',
      pgrDescriptions[query_type] || pgrDescriptions.broad);

    // ── Step 7: Answer generation ─────────────────────────────────────────
    setStepStatus('step-answer', 'active');
    await pause(400);

    const promptNames = {
      bridge:        'Bridge Prompt — "Follow the chain step by step, then synthesise"',
      single_entity: 'Entity Profile Prompt — "Describe using only the provided profile"',
      multi_entity:  'Multi-Entity Prompt — "Describe each entity, then explain connections"',
      broad:         'LightRAG Hybrid — standard hybrid retrieval prompt',
    };

    setStepStatus('step-answer', 'done', `
      <div class="text-xs text-slate-600 bg-slate-50 rounded p-3 border border-slate-100">
        <span class="font-medium">Prompt used:</span> ${escapeHtml(promptNames[query_type] || 'Hybrid')}
      </div>
      <div class="text-xs text-slate-400 mt-2">
        The LLM receives the structured PGR context above and generates an answer
        grounded in the knowledge graph — no external knowledge injection.
      </div>
    `);

  } catch (err) {
    const trace = document.getElementById('demo-trace');
    trace.insertAdjacentHTML('beforeend',
      `<div class="text-red-600 text-sm p-3 bg-red-50 rounded-lg border border-red-200">
        Error: ${escapeHtml(err.message)}
      </div>`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Pipeline';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SOURCE-FILE BUTTONS
// ═══════════════════════════════════════════════════════════════════════════

function setupSourceButtons() {
  let activePopover = null;

  document.querySelectorAll('.src-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();

      // Close existing
      if (activePopover && activePopover !== btn._popover) {
        activePopover.remove();
        activePopover = null;
      }
      if (btn._popover) {
        btn._popover.remove();
        btn._popover = null;
        activePopover = null;
        return;
      }

      const files = JSON.parse(btn.dataset.files || '[]');
      const pop = document.createElement('div');
      pop.className = 'src-popover';
      pop.innerHTML = files.map(f => {
        const ext = f.url.split('.').pop().split('?')[0].toUpperCase();
        return `<a href="${f.url}" target="_blank" rel="noopener">
          <span class="file-icon">${ext}</span>${escapeHtml(f.label)}
        </a>`;
      }).join('');

      btn.style.position = 'relative';
      btn.appendChild(pop);
      btn._popover  = pop;
      activePopover = pop;
    });
  });

  document.addEventListener('click', () => {
    if (activePopover) {
      activePopover.remove();
      // clear the reference on all buttons
      document.querySelectorAll('.src-btn').forEach(b => { b._popover = null; });
      activePopover = null;
    }
  });
}

// Update BER source-file button when domain tab changes
function updateBerSrcBtn(domain) {
  const btn = document.getElementById('ber-src-btn');
  if (!btn) return;
  btn.dataset.files = JSON.stringify([
    { label: 'ber_results.json (all domains)',       url: '/data/ber_results.json' },
    { label: `${domain}_bridge_gt_eval.json`,        url: `/raw/results/${domain}_bridge_gt_eval.json` },
    { label: `${domain}_bridge_gt.json (questions)`, url: `/raw/questions/${domain}_bridge_gt.json` },
    { label: `${domain}_bridge_gt.jsonl (answers)`,  url: `/raw/answers/${domain}_bridge_gt.jsonl` },
  ]);
}


// ═══════════════════════════════════════════════════════════════════════════
// HOW PGR WORKS — section rendering
// ═══════════════════════════════════════════════════════════════════════════

const PGR_MOTIFS = {
  bridge: {
    label: 'Bridge Path',
    color: '#ea580c',
    desc: 'For a query like "What connects A to C?", the bidirectional shortest path A → B → C is serialised into numbered steps. Every entity and every edge gets its own labelled step, so the bridge node B is structurally unavoidable — the LLM cannot skip it without breaking the chain.',
    code: `<span class="pgr-step-entity">[Step 1 — Entity: Rational Grazing (method)]</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">A systematic approach to managing pasture through controlled
  grazing periods that enhance productivity and pasture health.</span>
<span class="pgr-key">Source Passages:</span> <span class="pgr-dim">[Passage] Rational Grazing is developed by André Voisin...</span>

<span class="pgr-step-conn">[Step 2 — Connection: Rational Grazing → Cows]</span>
<span class="pgr-key">Relationship keywords:</span> <span class="pgr-dim">livestock management, rotational grazing, pasture recovery</span>
<span class="pgr-key">Source Passages:</span> <span class="pgr-dim">[Passage] Cows are moved across paddocks according to...</span>

<span class="pgr-step-entity">[Step 3 — Entity: <span class="pgr-bridge">Cows</span> (livestock)]</span>       <span class="pgr-comment">&lt;-- bridge entity, at a labelled step</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">Cows are central to regenerative farming systems.
  At Polyface Farm, Joel Salatin uses cows as the primary animal...</span>

<span class="pgr-step-conn">[Step 4 — Connection: Cows → Polyface Farm]</span>
<span class="pgr-key">Relationship keywords:</span> <span class="pgr-dim">rotational livestock, farm operations</span>

<span class="pgr-step-entity">[Step 5 — Entity: Polyface Farm (organisation)]</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">Joel Salatin's regenerative farm in Virginia, known for
  multi-species rotational operations...</span>`,
    prompt: `You are an expert AI assistant. The context below is structured as a step-by-step
reasoning chain through a knowledge graph, tracing the connection between entities.

Instructions:
1. Follow the chain from the first entity to the last, step by step
2. At each step, explain the connection using ONLY the provided facts
3. Synthesize the complete pathway into a coherent causal explanation
4. DO NOT introduce external knowledge

<span class="ph">---Reasoning Chain---
{context}
---Question---
{query}
---Answer (trace the chain step by step, then synthesise)---</span>`,
    insight: 'The bridge entity (Cows) is at Step 3 — a labelled position. The LLM prompt explicitly asks it to "follow the chain step by step." It cannot produce a fluent answer without naming the entity at each step. Flat retrieval gives the same passages but no chain topology — the bridge can be silently omitted.',
  },

  single_entity: {
    label: 'Single Entity',
    color: '#2563eb',
    desc: 'For a query like "What is Soil Organic Matter?", GRASP retrieves the 1-hop ego-network of the anchor node and formats it as an entity profile with its type, description, direct connections, and source passages.',
    code: `<span class="pgr-step-entity">[Entity Profile: Soil Organic Matter]</span>
<span class="pgr-key">Type:</span>        <span class="pgr-dim">concept</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">Soil organic matter consists of plant and animal residues at various
  stages of decomposition, cells and tissues of soil microbes, and
  substances synthesised by soil organisms.</span>
<span class="pgr-key">Connected to:</span>
  <span class="pgr-dim">- Soil Health:       nutrient cycling, microbial activity
  - Carbon Sequestration: climate regulation, carbon storage
  - Compost:           decomposition, soil amendment
  - Crop Yield:        agricultural productivity, soil fertility
  - Cover Crops:       organic matter input, erosion prevention</span>
<span class="pgr-key">Source Passages:</span>
  <span class="pgr-dim">[Passage] Soil organic matter is a key indicator of soil health...
  [Passage] Studies show SOM improves water retention and...</span>`,
    prompt: `You are an expert AI assistant. The context below contains a structured
profile of a specific entity from a knowledge graph.

Instructions:
1. Describe the entity using ONLY the provided profile and source passages
2. Cover its type, key properties, role in the domain, and how it relates to others
3. DO NOT introduce external knowledge

<span class="ph">---Entity Profile---
{context}
---Question---
{query}
---Answer---</span>`,
    insight: 'The ego-network format gives the LLM a ready-made profile. It does not need to infer connections — they are explicitly listed with edge keywords. This prevents hallucination of connections that do not exist in the graph.',
  },

  multi_entity: {
    label: 'Multi-Entity',
    color: '#16a34a',
    desc: 'For a query comparing or relating 3+ entities, GRASP computes a Steiner tree — the minimum-cost subgraph spanning all anchors. Each entity gets its own profile section, followed by an explicit connection map.',
    code: `<span class="pgr-step-entity">[Entity 1: Cover Crops (method)]</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">Plants grown primarily to manage soil health, erosion, water, weeds,
  pests, diseases, and biodiversity in an agroecosystem...</span>
<span class="pgr-key">Source Passages:</span> <span class="pgr-dim">[Passage] Cover crops are incorporated into crop rotations to...</span>

<span class="pgr-step-entity">[Entity 2: Crop Rotation (method)]</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">The practice of growing different crops sequentially on the same
  land to improve soil health and optimize nutrients in the soil...</span>

<span class="pgr-step-entity">[Entity 3: Composting (process)]</span>
<span class="pgr-key">Description:</span> <span class="pgr-dim">Biological decomposition and recycling of organic material into
  a rich soil amendment known as compost...</span>

<span class="pgr-step-conn">[Connections between entities]</span>
<span class="pgr-dim">  - Cover Crops ↔ Crop Rotation: soil health, nitrogen fixation, weed suppression
  - Cover Crops ↔ Composting:  organic matter input, decomposition
  - Crop Rotation ↔ Composting: nutrient cycling, soil amendment</span>
<span class="pgr-key">Connection Source Passages:</span> <span class="pgr-dim">[Passage]...</span>`,
    prompt: `You are an expert AI assistant. The context below contains structured
profiles of multiple entities and their interconnections.

Instructions:
1. Describe each entity individually using the provided facts
2. Explain how the entities connect and interact based on the graph structure
3. Synthesize or compare them as the question requires
4. DO NOT introduce external knowledge

<span class="ph">---Entity Comparison Context---
{context}
---Question---
{query}
---Answer---</span>`,
    insight: 'The Steiner tree finds the minimum-cost subgraph connecting all named anchors. Intermediate Steiner nodes (not in the original query) are discovered automatically — these are the connecting concepts that explain how the queried entities relate to each other.',
  },

  broad: {
    label: 'Broad',
    color: '#7c3aed',
    desc: 'For general questions with no specific named entity, GRASP uses Personalized PageRank (PPR) seeded at query-relevant nodes. The top-scoring nodes across the full graph are retrieved. If confidence is below threshold, it falls back to LightRAG hybrid mode.',
    code: `<span class="pgr-comment">// No fixed PGR structure for broad queries.</span>
<span class="pgr-comment">// PPR diffusion returns top-k nodes scored by:</span>
<span class="pgr-comment">//   π(v) = α · M^T · π(v) + (1-α) · seed_distribution</span>
<span class="pgr-comment">//   where α = 0.85 (restart probability)</span>

<span class="pgr-step-entity">Seed nodes</span> (from query vector search):
  <span class="pgr-dim">Regenerative Agriculture  π=0.21
  Soil Health               π=0.18
  Carbon Sequestration      π=0.14
  Biodiversity              π=0.11</span>

<span class="pgr-step-entity">Top PPR nodes retrieved:</span>
  <span class="pgr-dim">Cover Crops, Composting, No-Till Farming,
  Agroforestry, Mycorrhizal Networks, ...</span>

<span class="pgr-comment">// Falls back to LightRAG hybrid (local + global) if:</span>
<span class="pgr-comment">//   query_type == "broad"  OR  confidence &lt; 0.04  OR  paths empty</span>`,
    prompt: `<span class="pgr-comment">// LightRAG hybrid mode prompt (standard RAG, no PGR scaffold)</span>

<span class="pgr-dim">Uses LightRAG's built-in hybrid retrieval:
  - Local mode:  retrieves entity/relation chunks near the query
  - Global mode: retrieves community-level summaries
  - Hybrid:      combines both, ranked by relevance score</span>

<span class="pgr-comment">// No step-by-step chain — broad questions do not have a
// well-defined graph topology to mirror.</span>`,
    insight: 'Broad queries have no natural graph path to follow, so PGR cannot impose structure. The system falls back to LightRAG hybrid mode, which is well-suited for open-ended questions. The confidence threshold (0.04) also triggers this fallback for any motif where graph retrieval is uncertain.',
  },
};

function setupPGRSection() {
  const content = document.getElementById('pgr-content');
  if (!content) return;

  function renderMotif(motif) {
    const m = PGR_MOTIFS[motif];
    content.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">

        <!-- Context format -->
        <div class="space-y-3">
          <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
            <div class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">What it does</div>
            <p class="text-sm text-slate-600 leading-relaxed">${m.desc}</p>
          </div>
          <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
            <div class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Context format fed to LLM</div>
            <div class="pgr-code">${m.code}</div>
          </div>
        </div>

        <!-- Prompt + insight -->
        <div class="space-y-3">
          <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
            <div class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Matching prompt template</div>
            <div class="pgr-prompt-box">${m.prompt}</div>
          </div>
          <div class="rounded-xl border p-4 text-sm text-slate-700 leading-relaxed"
               style="background:${m.color}08; border-color:${m.color}30;">
            <div class="font-semibold mb-1" style="color:${m.color}">Key insight</div>
            ${m.insight}
          </div>
        </div>

      </div>
    `;
  }

  renderMotif('bridge');

  document.querySelectorAll('#pgr-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#pgr-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderMotif(btn.dataset.motif);
    });
  });
}


// ── Start ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  init();
  setupDemoSection();
  setupPGRSection();
  setupSourceButtons();
});
