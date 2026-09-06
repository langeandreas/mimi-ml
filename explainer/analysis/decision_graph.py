"""Decision-narrowing graph construction and rendering.

Turns the ``scope -> decider`` co-occurrence signal produced by
:class:`~explainer.analysis.decision_context_analyzer.DecisionContextAnalyzer`
into a directed graph of *how the model narrows decisions*:

- **Nodes** are features, annotated with their late-decider behaviour
  (``decision_impact_score``, ``late_decider_rate``, behavioral quadrant).
- **Edges** ``scope_feature -> decider_feature`` mean "the scope feature
  repeatedly narrows the population *before* the decider makes its late split",
  weighted by how often they co-occur along root-to-leaf paths.

The JSON builder (:func:`build_decision_narrowing_graph`) has no third-party
dependencies, so it can be used headlessly. The renderer
(:func:`decision_narrowing_to_graphviz`) produces a ``graphviz.Digraph`` whose
DOT source can be displayed by any Graphviz-compatible frontend (including
Streamlit's ``st.graphviz_chart``) without requiring the ``dot`` binary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import graphviz  # Local import keeps the JSON builder dependency-free.

from .trajectory_utils import round_float

# Behavioral-quadrant colour palette (shared with the rest of the analysis app).
QUADRANT_COLORS: Dict[str, str] = {
    "CONFLICT RESOLVER": "#d62728",
    "HIGH-VARIANCE PATCH": "#ff7f0e",
    "OUTLIER SPECIALIST": "#9467bd",
    "EASY": "#2ca02c",
}
_DEFAULT_NODE_COLOR = "#7f7f7f"


def build_decision_narrowing_graph(
    decision_context: Dict[str, Any],
    *,
    top_deciders: int = 15,
    max_scope_per_decider: int = 5,
    min_cooccurrence: int = 1,
    max_edges: int = 60,
) -> Dict[str, Any]:
    """Build a directed decision-narrowing graph from decision-context output.

    Args:
        decision_context: The dict returned by
            :meth:`DecisionContextAnalyzer.analyze` (must contain
            ``feature_profiles``).
        top_deciders: Keep incoming ``scope -> decider`` edges only for the
            top-N features ranked by ``decision_impact_score``.
        max_scope_per_decider: Maximum incoming scope edges kept per decider.
        min_cooccurrence: Minimum ``scope -> decider`` co-occurrence count for
            an edge to be retained.
        max_edges: Global cap on the number of edges (highest co-occurrence
            counts are kept).

    Returns:
        A JSON-serializable graph dict with ``schema``, ``metadata``,
        ``nodes`` and ``edges`` keys.
    """
    if top_deciders < 1:
        raise ValueError("top_deciders must be >= 1")
    if max_scope_per_decider < 1:
        raise ValueError("max_scope_per_decider must be >= 1")
    if max_edges < 1:
        raise ValueError("max_edges must be >= 1")

    profiles: List[Dict[str, Any]] = list(decision_context.get("feature_profiles", []))
    profile_by_idx: Dict[int, Dict[str, Any]] = {
        int(p["feature_idx"]): p for p in profiles
    }

    outlier_ids = {
        int(c["feature_idx"])
        for c in decision_context.get("outlier_specialist_candidates", [])
    }

    # Rank deciders by decision impact; only those acting as late deciders matter.
    ranked_deciders = sorted(
        (p for p in profiles if float(p.get("late_decider_rate", 0.0)) > 0.0),
        key=lambda p: float(p.get("decision_impact_score", 0.0)),
        reverse=True,
    )[:top_deciders]

    # Collect candidate edges: scope_feature -> decider_feature.
    candidate_edges: List[Dict[str, Any]] = []
    for decider in ranked_deciders:
        decider_idx = int(decider["feature_idx"])
        scope_features = decider.get("top_scope_features", [])[:max_scope_per_decider]
        for scope in scope_features:
            scope_idx = int(scope["feature_idx"])
            if scope_idx == decider_idx:
                continue
            count = int(scope.get("cooccurrence_count", 0))
            if count < min_cooccurrence:
                continue
            candidate_edges.append(
                {
                    "source": scope_idx,
                    "target": decider_idx,
                    "cooccurrence_count": count,
                    "cooccurrence_rate": round_float(
                        float(scope.get("cooccurrence_rate", 0.0))
                    ),
                }
            )

    # Keep the strongest edges up to the global cap.
    candidate_edges.sort(
        key=lambda e: (-e["cooccurrence_count"], e["source"], e["target"])
    )
    kept_edges = candidate_edges[:max_edges]

    # Assemble node set from the endpoints of kept edges.
    node_ids = {e["source"] for e in kept_edges} | {e["target"] for e in kept_edges}

    def _node_name(idx: int) -> str:
        profile = profile_by_idx.get(idx)
        if profile is not None:
            return str(profile.get("feature_name", idx))
        return str(idx)

    decider_ids = {int(d["feature_idx"]) for d in ranked_deciders}
    nodes: List[Dict[str, Any]] = []
    for idx in sorted(node_ids):
        profile = profile_by_idx.get(idx, {})
        nodes.append(
            {
                "id": int(idx),
                "feature_name": _node_name(idx),
                "decision_impact_score": round_float(
                    float(profile.get("decision_impact_score", 0.0))
                ),
                "late_decider_rate": round_float(
                    float(profile.get("late_decider_rate", 0.0))
                ),
                "quadrant_category": profile.get("quadrant_category"),
                "mean_abs_leaf_when_late_decider": round_float(
                    float(profile.get("mean_abs_leaf_when_late_decider", 0.0))
                ),
                "path_count": int(profile.get("path_count", 0)),
                "is_decider": bool(idx in decider_ids),
                "is_outlier_specialist": bool(idx in outlier_ids),
            }
        )

    edges: List[Dict[str, Any]] = [
        {
            "source": int(e["source"]),
            "source_name": _node_name(int(e["source"])),
            "target": int(e["target"]),
            "target_name": _node_name(int(e["target"])),
            "cooccurrence_count": int(e["cooccurrence_count"]),
            "cooccurrence_rate": float(e["cooccurrence_rate"]),
        }
        for e in kept_edges
    ]

    return {
        "schema": "decision_narrowing_graph_v1",
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "top_deciders": int(top_deciders),
            "max_scope_per_decider": int(max_scope_per_decider),
            "min_cooccurrence": int(min_cooccurrence),
            "max_edges": int(max_edges),
            "edge_semantics": "source scope-feature narrows the decision scope before target decider-feature's late split",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _edge_penwidth(count: int, max_count: int) -> float:
    """Scale a co-occurrence count to a Graphviz pen width in [1.0, 6.0]."""
    if max_count <= 0:
        return 1.0
    return round(1.0 + 5.0 * (count / max_count), 2)


def _node_penwidth(score: float, max_score: float) -> float:
    """Scale a decision-impact score to a Graphviz pen width in [1.0, 5.0]."""
    if max_score <= 0:
        return 1.0
    return round(1.0 + 4.0 * (score / max_score), 2)


def decision_narrowing_to_graphviz(
    graph: Dict[str, Any],
    *,
    highlight_feature_id: Optional[int] = None,
    rankdir: str = "LR",
    node_fontsize: int = 20,
    edge_fontsize: int = 10,
):
    """Render a decision-narrowing graph as a ``graphviz.Digraph``.

    The returned object exposes ``.source`` (DOT text) that can be displayed
    without the Graphviz ``dot`` binary being installed — e.g. through
    ``streamlit.graphviz_chart``.

    Args:
        graph: Output of :func:`build_decision_narrowing_graph`.
        highlight_feature_id: When set, dims every node and edge not connected
            to this feature to spotlight its narrowing context.
        rankdir: Graph orientation passed to Graphviz (``"LR"`` or ``"TB"``).

    Returns:
        A ``graphviz.Digraph`` instance.
    """

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    max_count = max((int(e["cooccurrence_count"]) for e in edges), default=0)
    max_score = max((float(n["decision_impact_score"]) for n in nodes), default=0.0)

    # Determine the neighbourhood of the highlighted feature, if any.
    highlighted_nodes = set()
    if highlight_feature_id is not None:
        highlighted_nodes.add(int(highlight_feature_id))
        for edge in edges:
            if int(edge["target"]) == int(highlight_feature_id):
                highlighted_nodes.add(int(edge["source"]))
            elif int(edge["source"]) == int(highlight_feature_id):
                highlighted_nodes.add(int(edge["target"]))

    if node_fontsize < 1:
        raise ValueError("node_fontsize must be >= 1")
    if edge_fontsize < 1:
        raise ValueError("edge_fontsize must be >= 1")

    dot = graphviz.Digraph("decision_narrowing", graph_attr={"rankdir": rankdir})
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize=str(node_fontsize),
    )
    dot.attr("edge", color="#555555", fontname="Helvetica", fontsize=str(edge_fontsize))

    for node in nodes:
        idx = int(node["id"])
        color = QUADRANT_COLORS.get(
            str(node.get("quadrant_category")), _DEFAULT_NODE_COLOR
        )
        penwidth = _node_penwidth(float(node["decision_impact_score"]), max_score)

        dimmed = highlight_feature_id is not None and idx not in highlighted_nodes
        fillcolor = "#ececec" if dimmed else _lighten(color)
        border = "#bbbbbb" if dimmed else color

        badge = " ★" if node.get("is_outlier_specialist") else ""
        label = (
            f"{node['feature_name']}{badge}\n"
            f"impact={node['decision_impact_score']} · late={node['late_decider_rate']}"
        )
        dot.node(
            str(idx),
            label=label,
            fillcolor=fillcolor,
            color=border,
            penwidth=str(penwidth),
            fontsize=str(node_fontsize),
            tooltip=(
                f"{node['feature_name']} | quadrant={node.get('quadrant_category')} | "
                f"path_count={node['path_count']}"
            ),
        )

    for edge in edges:
        src = int(edge["source"])
        tgt = int(edge["target"])
        count = int(edge["cooccurrence_count"])
        penwidth = _edge_penwidth(count, max_count)
        dimmed = highlight_feature_id is not None and (
            src not in highlighted_nodes or tgt not in highlighted_nodes
        )
        dot.edge(
            str(src),
            str(tgt),
            penwidth=str(penwidth),
            color="#cccccc" if dimmed else "#555555",
            tooltip=f"co-occurs {count}x (rate={edge['cooccurrence_rate']})",
        )

    return dot


def _lighten(hex_color: str) -> str:
    """Return a lightened pastel variant of a hex colour for node fills."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#eeeeee"
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * 0.65)
    g = int(g + (255 - g) * 0.65)
    b = int(b + (255 - b) * 0.65)
    return f"#{r:02x}{g:02x}{b:02x}"
