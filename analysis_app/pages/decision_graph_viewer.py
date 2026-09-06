"""Decision-Narrowing Graph page for the Streamlit app.

Visualizes how the XGBoost model narrows its decisions: directed edges run from
*scope* features (which repeatedly narrow the population early on a root-to-leaf
path) to *decider* features (which make the decisive late split before a leaf).
This exposes globally-quiet features that act as contextual "final deciders".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from explainer.analysis.decision_graph import (
    QUADRANT_COLORS,
    decision_narrowing_to_graphviz,
)
from explainer.analysis.trajectory_utils import substitute_feature_names
from explainer.analysis.tree_analyzer import TreeAnalyzer


def _resolve_booster(model: Any):
    """Return the underlying XGBoost booster from a model or GridSearchCV."""
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_.get_booster()
    return model.get_booster()


def _render_legend() -> None:
    """Render a small colour legend for the behavioral quadrants."""
    swatches = " &nbsp; ".join(
        f"<span style='color:{color};font-size:18px'>&#9632;</span> {name.title()}"
        for name, color in QUADRANT_COLORS.items()
    )
    st.markdown(
        f"{swatches} &nbsp; <span style='color:#7f7f7f;font-size:18px'>&#9632;</span> Unclassified"
        " &nbsp;&nbsp; (&#9733; = outlier-specialist candidate)",
        unsafe_allow_html=True,
    )


def _apply_readable_feature_names(
    graph: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Replace node/edge feature labels with human-readable names."""
    node_ids = {int(node["id"]) for node in graph.get("nodes", [])}
    edge_rows = graph.get("edges", [])
    for edge in edge_rows:
        node_ids.add(int(edge["source"]))
        node_ids.add(int(edge["target"]))

    lookup_df = pd.DataFrame(
        {
            "feature_id": sorted(node_ids),
            "feature_name": sorted(node_ids),
        }
    )
    lookup_df = substitute_feature_names(
        lookup_df,
        feature_index_col="feature_name",
        feature_names=feature_names,
    )
    readable_map = {
        int(row["feature_id"]): str(row["feature_name"])
        for _, row in lookup_df.iterrows()
    }

    readable_nodes = []
    for node in graph.get("nodes", []):
        node_copy = dict(node)
        feature_id = int(node_copy["id"])
        node_copy["feature_name"] = readable_map.get(
            feature_id, str(node_copy.get("feature_name", feature_id))
        )
        readable_nodes.append(node_copy)

    readable_edges = []
    for edge in edge_rows:
        edge_copy = dict(edge)
        source_id = int(edge_copy["source"])
        target_id = int(edge_copy["target"])
        edge_copy["source_name"] = readable_map.get(source_id, str(edge_copy.get("source_name", source_id)))
        edge_copy["target_name"] = readable_map.get(target_id, str(edge_copy.get("target_name", target_id)))
        readable_edges.append(edge_copy)

    return {
        "schema": graph.get("schema"),
        "metadata": graph.get("metadata", {}),
        "nodes": readable_nodes,
        "edges": readable_edges,
    }


def render_decision_graph_page() -> None:
    """Render the Decision-Narrowing Graph page."""
    st.header("Decision-Narrowing Graph")
    st.caption(
        "How the model narrows decisions: **scope** features (sources) narrow the "
        "population before **decider** features (targets) make the decisive late split. "
        "Edge thickness = co-occurrence strength; node border = decision-impact score."
    )

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info(
            "No trained model available. Please run the trajectory analysis in the "
            "**Setup** page first."
        )
        return

    summary_traj = st.session_state.summary_traj

    if not getattr(summary_traj, "model", None):
        st.info("Model not available in session state. Run setup first.")
        return

    tree_entries = getattr(getattr(summary_traj, "explain", None), "trees", None)
    if not tree_entries:
        st.warning("No in-memory tree objects available for decision-path analysis.")
        return

    # Controls.
    c1, c2, c3 = st.columns(3)
    with c1:
        top_deciders = st.slider("Top deciders", 3, 30, 8, key="dgraph_top_deciders")
    with c2:
        max_scope_per_decider = st.slider("Scope edges per decider", 1, 10, 4, key="dgraph_scope_per")
    with c3:
        max_edges = st.slider("Max total edges", 10, 150, 60, key="dgraph_max_edges")

    min_cooccurrence = st.slider(
        "Minimum co-occurrence count for an edge", 1, 50, 25, key="dgraph_min_cooc"
    )
    node_fontsize = st.slider(
        "Node font size", 10, 40, 10, key="dgraph_node_fontsize"
    )
    fit_to_container = st.checkbox(
        "Fit graph to container width",
        value=False,
        key="dgraph_fit_container",
        help="Disable this to preserve true font scaling.",
    )

    try:
        with st.spinner("Computing trajectory metrics and decision-path context..."):
            tree_analyzer = TreeAnalyzer(summary_traj.classification)
            booster = _resolve_booster(summary_traj.model)
            trajectory_metrics = tree_analyzer.compute_trajectory_metrics(booster)

            graph = tree_analyzer.build_decision_narrowing_graph(
                tree_entries=tree_entries,
                trajectory_metrics=trajectory_metrics,
                top_deciders=top_deciders,
                max_scope_per_decider=max_scope_per_decider,
                min_cooccurrence=min_cooccurrence,
                max_edges=max_edges,
            )
            graph = _apply_readable_feature_names(graph, tree_analyzer.feature_names)
    except Exception as exc:  # noqa: BLE001 - surface any analysis error to the UI
        st.error(f"Error building decision-narrowing graph: {exc}")
        return

    nodes: List[Dict[str, Any]] = graph.get("nodes", [])
    edges: List[Dict[str, Any]] = graph.get("edges", [])

    if not edges:
        st.info(
            "No scope-to-decider edges met the current thresholds. Try lowering the "
            "minimum co-occurrence count or increasing the number of deciders."
        )
        return

    m1, m2 = st.columns(2)
    m1.metric("Feature nodes", graph["metadata"]["node_count"])
    m2.metric("Narrowing edges", graph["metadata"]["edge_count"])

    # Optional highlight of a single decider's narrowing context.
    decider_options = ["(show all)"] + [
        f"{n['feature_name']} ({n['id']})" for n in nodes if n.get("is_decider")
    ]
    highlight_choice = st.selectbox(
        "Highlight the narrowing context of a decider",
        options=decider_options,
        key="dgraph_highlight",
    )
    highlight_id = None
    if highlight_choice != "(show all)":
        highlight_id = int(highlight_choice.rsplit("(", 1)[1].rstrip(")"))

    _render_legend()

    dot = decision_narrowing_to_graphviz(
        graph,
        highlight_feature_id=highlight_id,
        node_fontsize=node_fontsize,
    )
    st.graphviz_chart(dot, use_container_width=fit_to_container)

    with st.expander("Edges (scope → decider)", expanded=False):
        edges_df = pd.DataFrame(edges)[
            ["source_name", "target_name", "cooccurrence_count", "cooccurrence_rate"]
        ].rename(
            columns={
                "source_name": "scope_feature",
                "target_name": "decider_feature",
                "cooccurrence_count": "cooccurrence",
                "cooccurrence_rate": "rate",
            }
        )
        st.dataframe(edges_df, width="content")

    with st.expander("Nodes (feature roles)", expanded=False):
        nodes_df = pd.DataFrame(nodes)[
            [
                "feature_name",
                "quadrant_category",
                "decision_impact_score",
                "late_decider_rate",
                "mean_abs_leaf_when_late_decider",
                "path_count",
                "is_decider",
                "is_outlier_specialist",
            ]
        ]
        st.dataframe(nodes_df, width="content")

    st.download_button(
        "Download graph JSON",
        data=json.dumps(graph, ensure_ascii=False),
        file_name="decision_narrowing_graph.json",
        mime="application/json",
        key="dgraph_download",
    )
