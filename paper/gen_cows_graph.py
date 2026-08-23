"""
Generate cows_graph.jpg — knowledge graph showing bridge paths
from Cows to Agriculture.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(_ROOT, "data", "report", "cows_graph.jpg")

# ── Graph ─────────────────────────────────────────────────────────────────────
G = nx.DiGraph()

# Short IDs → display labels
node_labels = {
    "Cows":        "Cows",
    "Brown":       "Brown\n(farmer)",
    "Soil":        "Soil",
    "Farmers":     "Farmers",
    "Farm":        "Farm",
    "Weeds":       "Weeds",
    "Polyface":    "Polyface\nFarm",
    "FoodSec":     "Food\nSecurity",
    "Agriculture": "Agriculture",
}

# 4 representative paths
edges_raw = [
    # Path 1 (best-ranked): Cows → Brown → Soil → Agriculture
    ("Cows",     "Brown",       "grazing practices,\nsoil restoration"),
    ("Brown",    "Soil",        "agriculture,\nhealth assessment"),
    ("Soil",     "Agriculture", "ecological role,\nresource significance"),
    # Path 2: Cows → Farm → Farmers → Agriculture
    ("Cows",     "Farm",        "livestock management,\nagriculture"),
    ("Farm",     "Farmers",     "agricultural\nproduction"),
    ("Farmers",  "Agriculture", "food production,\neconomic contribution"),
    # Path 3: Cows → Weeds → Soil → Agriculture  (Soil already placed)
    ("Cows",     "Weeds",       "eating behavior,\nnutritional selection"),
    ("Weeds",    "Soil",        "crop interference,\nfarming issue"),
    # Path 4: Cows → Polyface Farm → Food Security → Agriculture
    ("Cows",     "Polyface",    "farming method,\nlivestock"),
    ("Polyface", "FoodSec",     "sustainability,\nnatural practices"),
    ("FoodSec",  "Agriculture", "communal\nresponsibility"),
]

for n in node_labels:
    G.add_node(n)
for src, dst, lbl in edges_raw:
    G.add_edge(src, dst, label=lbl)

# ── Layout ────────────────────────────────────────────────────────────────────
pos = {
    "Cows":        (-5.2,  0.0),
    # Path 1 — top lane
    "Brown":       (-2.4,  2.2),
    "Soil":        ( 0.5,  2.2),
    # Path 2 — upper-mid lane
    "Farm":        (-2.4,  0.7),
    "Farmers":     ( 0.5,  0.7),
    # Path 3 — lower-mid (Weeds connects to Soil, already placed)
    "Weeds":       (-2.4, -0.7),
    # Path 4 — bottom lane
    "Polyface":    (-2.4, -2.2),
    "FoodSec":     ( 0.5, -2.2),
    # Anchor right
    "Agriculture": ( 4.0,  0.0),
}

# ── Colors ────────────────────────────────────────────────────────────────────
anchors = {"Cows", "Agriculture"}
node_colors = ["#3a7ebf" if n in anchors else "#b0b8c1" for n in G.nodes()]
node_sizes  = [3400 if n in anchors else 2200 for n in G.nodes()]

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 9))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=node_colors, node_size=node_sizes, alpha=0.92)

nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
    font_size=8.0, font_color="white", font_weight="bold")

nx.draw_networkx_edges(G, pos, ax=ax,
    edge_color="#6c757d", arrows=True,
    arrowstyle="-|>", arrowsize=18,
    connectionstyle="arc3,rad=0.08",
    width=1.6, alpha=0.75,
    min_source_margin=22, min_target_margin=22)

edge_labels = {(s, d): lbl for s, d, lbl in edges_raw}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
    font_size=6.5, font_color="#343a40",
    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
    label_pos=0.42)

# ── Path legend ───────────────────────────────────────────────────────────────
path_labels = [
    "Path 1 (best): Cows → Brown → Soil → Agriculture",
    "Path 2: Cows → Farm → Farmers → Agriculture",
    "Path 3: Cows → Weeds → Soil → Agriculture",
    "Path 4: Cows → Polyface Farm → Food Security → Agriculture",
]
legend_handles = [
    mpatches.Patch(color="#3a7ebf", label="Anchor entity"),
    mpatches.Patch(color="#b0b8c1", label="Intermediate entity"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
          framealpha=0.85, edgecolor="#ced4da")

# Path annotation
ax.text(0.5, -0.02,
    " | ".join(f"P{i+1}: {p.split(': ')[1]}" for i, p in enumerate(path_labels)),
    transform=ax.transAxes, ha="center", va="top",
    fontsize=6.5, color="#555", style="italic")

ax.set_title(
    'Bridge Paths: "Cows" → "Agriculture"  (GRASP-RAG bridge motif, 10 paths found)',
    fontsize=13, fontweight="bold", pad=12, color="#212529"
)
ax.set_xlim(-6.8, 5.6)
ax.set_ylim(-3.2, 3.2)
ax.axis("off")

plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)
plt.savefig(OUT, dpi=150, format="jpeg", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
