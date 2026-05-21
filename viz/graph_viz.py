"""
viz/graph_viz.py — Interactive knowledge-graph visualiser for LightRAG/RAGAnything.

Colour strategy (no hardcoded entity-type lists)
-------------------------------------------------
1. KNOWN_TYPE_OVERRIDES — explicit colours/shapes/icons for well-known types
   (standard LightRAG + RAGAnything multimodal). Extend this dict freely.
2. Auto-assignment — any entity type found in the graph that is NOT in the
   overrides gets a colour automatically drawn from AUTO_PALETTE (a curated
   sequence of visually distinct hues). The mapping is computed fresh each run
   from the actual graph data, so it adapts to whatever entity_types you have
   configured in lightrag_init.py without any code changes.

A colour legend is printed to stdout after rendering so you can see exactly
which colour was assigned to each discovered type.

Usage
-----
Standalone:
    python -m viz.graph_viz
    python -m viz.graph_viz --out my_graph.html
    python -m viz.graph_viz --graphml path/to/file.graphml

Programmatic:
    from viz.graph_viz import render_graph
    render_graph()
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Known-type overrides
# Add / edit entries here for any type you want to pin to a specific style.
# All other types discovered in the graph get auto-assigned from AUTO_PALETTE.
# ---------------------------------------------------------------------------

KNOWN_TYPE_OVERRIDES: dict[str, dict] = {
    # ── Standard LightRAG entity types ─────────────────────────────────────
    "PERSON":       {"color": "#4fc3f7", "shape": "dot",      "icon": ""},
    "ORG":          {"color": "#aed581", "shape": "dot",      "icon": ""},
    "ORGANIZATION": {"color": "#aed581", "shape": "dot",      "icon": ""},
    "LOCATION":     {"color": "#f06292", "shape": "dot",      "icon": ""},
    "CONCEPT":      {"color": "#ff8a65", "shape": "dot",      "icon": ""},
    "EVENT":        {"color": "#ce93d8", "shape": "dot",      "icon": ""},
    "TECHNOLOGY":   {"color": "#80cbc4", "shape": "dot",      "icon": ""},

    # ── RAGAnything multimodal node types ───────────────────────────────────
    "IMAGE":        {"color": "#ffe082", "shape": "square",   "icon": "🖼 "},
    "FIGURE":       {"color": "#ffe082", "shape": "square",   "icon": "🖼 "},
    "TABLE":        {"color": "#a5d6a7", "shape": "diamond",  "icon": "📊 "},
    "EQUATION":     {"color": "#ef9a9a", "shape": "triangle", "icon": "∑ "},
    "CHUNK":        {"color": "#b0bec5", "shape": "dot",      "icon": "📄 "},
}

# Palette for auto-assigning colours to any type NOT in KNOWN_TYPE_OVERRIDES.
# Chosen for contrast against a dark (#0f0f0f) background and against each other.
AUTO_PALETTE: list[str] = [
    "#ffcc80",  # warm amber
    "#80deea",  # cyan
    "#c5e1a5",  # lime
    "#ffab91",  # peach
    "#b39ddb",  # violet
    "#80cbc4",  # teal
    "#f48fb1",  # rose
    "#fff59d",  # pale yellow
    "#90caf9",  # light blue
    "#a1887f",  # brown
    "#dce775",  # yellow-green
    "#4db6ac",  # medium teal
    "#ff8a65",  # deep orange
    "#7986cb",  # indigo
    "#4dd0e1",  # aqua
    "#e6ee9c",  # yellow-lime
]

DEFAULT_COLOR  = "#ffcc80"
DEFAULT_SHAPE  = "dot"
DEFAULT_ICON   = ""

# Cross-modal relation types from RAGAnything — rendered as dashed edges
CROSS_MODAL_RELATIONS: frozenset[str] = frozenset(
    {"IMAGE_DESCRIBES", "TABLE_SUPPORTS", "EQUATION_DEFINES"}
)


# ---------------------------------------------------------------------------
# Type-style registry (built at render time from the actual graph)
# ---------------------------------------------------------------------------

class TypeStyleRegistry:
    """Assigns colours/shapes/icons to entity types, auto-discovering unknowns.

    Call ``register_all(graph)`` once to scan the graph, then use
    ``get(entity_type)`` to retrieve styles for individual nodes.
    """

    def __init__(self) -> None:
        self._auto_map: dict[str, str] = {}   # type → auto-assigned colour
        self._palette_idx: int = 0

    def _next_color(self) -> str:
        color = AUTO_PALETTE[self._palette_idx % len(AUTO_PALETTE)]
        self._palette_idx += 1
        return color

    def register_all(self, G: nx.Graph) -> None:
        """Pre-scan the graph and assign colours to all entity types found."""
        for _, data in G.nodes(data=True):
            et = data.get("entity_type", "UNKNOWN").upper()
            if et not in KNOWN_TYPE_OVERRIDES and et not in self._auto_map:
                self._auto_map[et] = self._next_color()

    def get(self, entity_type: str) -> dict:
        """Return the style dict for an entity type."""
        et = entity_type.upper()
        if et in KNOWN_TYPE_OVERRIDES:
            return KNOWN_TYPE_OVERRIDES[et]
        return {
            "color": self._auto_map.get(et, DEFAULT_COLOR),
            "shape": DEFAULT_SHAPE,
            "icon":  DEFAULT_ICON,
        }

    def legend(self) -> dict[str, str]:
        """Return a {type: colour} dict covering both known and auto types."""
        result = {k: v["color"] for k, v in KNOWN_TYPE_OVERRIDES.items()}
        result.update(self._auto_map)
        return result


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_networkx_graph(
    G: nx.Graph,
    output_path: str = "lightrag_graph.html",
) -> str:
    """Render a NetworkX graph using pyvis and save it as an interactive HTML file."""
    print(f"Graph rendering — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    degrees = dict(G.degree())

    # ── Build type-style registry from actual graph data ───────────────────
    registry = TypeStyleRegistry()
    registry.register_all(G)   # auto-assigns colours for all discovered types

    # ── Pyvis network ──────────────────────────────────────────────────────
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0f0f0f",
        font_color="white",
        notebook=False,
        directed=G.is_directed(),
    )

    # ── Add nodes ──────────────────────────────────────────────────────────
    for node, data in G.nodes(data=True):
        entity_type     = data.get("entity_type", "UNKNOWN").upper()
        description     = data.get("description", "No description available.")
        source_modality = data.get("source_modality", "")
        file_path       = data.get("file_path", "")
        source_id       = data.get("source_id", "")

        style = registry.get(entity_type)
        icon  = style["icon"]

        raw_label = node if len(node) <= 28 else node[:26] + "…"
        label     = icon + raw_label

        # Tooltip
        tooltip_lines = [node, entity_type]
        if source_modality:
            tooltip_lines.append(f"Modality: {source_modality}")
        if file_path:
            tooltip_lines.append(f"Source: {Path(file_path).name}")
        if source_id:
            tooltip_lines.append(f"src_id: {source_id[:80]}")
        tooltip_lines += [
            "",
            description[:300] + ("…" if len(description) > 300 else ""),
        ]
        title = "\n".join(filter(None, tooltip_lines))

        is_multimodal = style["shape"] != DEFAULT_SHAPE
        base_size = 16 if is_multimodal else 12
        size = base_size + min(degrees[node] * 4, 40)

        net.add_node(
            node,
            label=label,
            title=title,
            size=size,
            color=style["color"],
            shape=style["shape"],
            borderWidth=3 if is_multimodal else 2,
            borderWidthSelected=4,
            font={"size": 11, "face": "Inter, Arial, sans-serif"},
        )

    # ── Add edges ──────────────────────────────────────────────────────────
    for src, dst, data in G.edges(data=True):
        keywords      = data.get("keywords", "")
        description   = data.get("description", "")
        weight        = float(data.get("weight", 1.0))
        relation_type = data.get("relation_type", "")

        tooltip_lines = []
        if keywords:
            tooltip_lines.append(keywords)
        if relation_type:
            tooltip_lines.append(f"Relation: {relation_type}")
        if description:
            tooltip_lines += [
                "",
                description[:200] + ("…" if len(description) > 200 else ""),
            ]
        title = "\n".join(filter(None, tooltip_lines))

        is_cross_modal = relation_type in CROSS_MODAL_RELATIONS
        edge_color = "#888800" if is_cross_modal else "#444444"

        net.add_edge(
            src, dst,
            title=title,
            width=0.5 + weight,
            color={"color": edge_color, "highlight": "#ffffff", "hover": "#aaaaaa"},
            dashes=is_cross_modal,
        )

    # ── Physics + interaction ──────────────────────────────────────────────
    net.set_options("""
{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 200,
      "springConstant": 0.05,
      "damping": 0.09
    },
    "stabilization": { "iterations": 200 }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "zoomView": true,
    "dragView": true,
    "navigationButtons": true,
    "keyboard": true
  },
  "nodes": {
    "scaling": { "min": 10, "max": 50 }
  },
  "edges": {
    "smooth": { "type": "continuous" },
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
  }
}
""")

    # ── Save ───────────────────────────────────────────────────────────────
    output_abs = str(Path(output_path).resolve())
    net.save_graph(output_abs)
    print(f"✅ Saved → {output_abs}")

    # ── Print discovered type legend ───────────────────────────────────────
    legend = registry.legend()
    print("\n── Entity-type colour legend ─────────────────────────────")
    for et, color in sorted(legend.items()):
        src = "known" if et in KNOWN_TYPE_OVERRIDES else "auto "
        print(f"  [{src}]  {color}  {et}")
    print("──────────────────────────────────────────────────────────\n")

    return output_abs


def render_graph(
    graphml_path: str | None = None,
    output_path: str = "lightrag_graph.html",
) -> str:
    """Load a LightRAG/RAGAnything GraphML file and save an interactive HTML viz.

    GraphML path resolution order:
        1. ``graphml_path`` argument (if provided)
        2. ``WORKING_DIR`` env var + ``graph_chunk_entity_relation.graphml``
        3. Fallback: ``./storage/dickens_v1/graph_chunk_entity_relation.graphml``

    Args:
        graphml_path: Explicit path to .graphml file. Optional.
        output_path:  Destination HTML file. Default: ``lightrag_graph.html``

    Returns:
        Absolute path of the saved HTML file.

    Raises:
        FileNotFoundError: If the GraphML file does not exist.
    """
    load_dotenv(dotenv_path=".env", override=False)

    # ── Resolve GraphML path ───────────────────────────────────────────────
    if graphml_path is None:
        working_dir = os.getenv("WORKING_DIR", "./storage/dickens_v1")
        graphml_path = str(
            Path(working_dir) / "graph_chunk_entity_relation.graphml"
        )

    graphml_path = str(Path(graphml_path).resolve())
    if not Path(graphml_path).exists():
        raise FileNotFoundError(
            f"GraphML file not found: {graphml_path}\n"
            "Run the ingest pipeline first (python main.py) to generate it."
        )

    # ── Load graph ─────────────────────────────────────────────────────────
    G = nx.read_graphml(graphml_path)
    return render_networkx_graph(G, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Render LightRAG/RAGAnything knowledge graph as interactive HTML."
    )
    parser.add_argument(
        "--graphml",
        default=None,
        help=(
            "Path to the GraphML file. "
            "Defaults to <WORKING_DIR>/graph_chunk_entity_relation.graphml"
        ),
    )
    parser.add_argument(
        "--out",
        default="lightrag_graph.html",
        help="Output HTML file path. Default: lightrag_graph.html",
    )
    args = parser.parse_args()

    try:
        render_graph(graphml_path=args.graphml, output_path=args.out)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
