"""Tree Viewer page for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis_app.config import DEFAULT_TREE_JSON_PATH
from analysis_app.utils import get_llm_config
from explainer.analysis.trajectory_utils import substitute_feature_names


def _load_tree_json(path: str) -> Dict[str, Any]:
    """Load tree JSON from file."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _index_to_name(idx: Any, feature_names: List[str]) -> str:
    """Resolve a numeric feature index to a readable name when possible."""
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return str(idx)

    if 0 <= i < len(feature_names):
        return str(feature_names[i])
    return str(idx)


def _substitute_feature_column(df: pd.DataFrame, col: str, feature_names: List[str]) -> pd.DataFrame:
    """Apply readable-name substitution to a feature-index column."""
    if col not in df.columns:
        return df
    out = df.copy()
    out = substitute_feature_names(out, feature_index_col=col, feature_names=feature_names)
    out[col] = out[col].astype(str)
    return out


def render_tree_viewer_page() -> None:
    """Render the Tree Viewer page."""
    st.header("Tree Trajectory Viewer")
    st.caption("Visualize model tree structure dynamics across training iterations.")

    config = get_llm_config()

    st.subheader("Tree Insights")
    st.caption("Load a tree trajectory JSON and explore complexity, split usage, and interactions.")

    tree_json_path = st.text_input(
        "(Optional) Custom Path to Tree JSON",
        value=str(config.get("tree_json_path", DEFAULT_TREE_JSON_PATH)),
        key="tree_json_path_input",
    )
    config["tree_json_path"] = tree_json_path

    try:
        tree_json = _load_tree_json(tree_json_path)
    except FileNotFoundError:
        st.error(f"Tree JSON file not found: {tree_json_path}")
        return
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON format in tree file: {exc}")
        return

    progress = tree_json.get("iteration_progress", [])
    if not progress:
        st.warning("No tree iteration data found in the provided JSON.")
        return

    feature_names = [str(name) for name in tree_json.get("feature_names", [])]

    metric_rows: List[Dict[str, Any]] = []
    leaf_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    interaction_rows: List[Dict[str, Any]] = []

    for step in progress:
        iteration = int(step.get("iteration", 0))
        metric_rows.append(
            {
                "iteration": iteration,
                "split_count": int(step.get("split_count", 0)),
                "leaf_count": int(step.get("leaf_count", 0)),
                "max_depth": int(step.get("max_depth", 0)),
            }
        )

        leaf_stats = step.get("leaf_value_stats", {})
        leaf_rows.append(
            {
                "iteration": iteration,
                "leaf_mean": float(leaf_stats.get("mean", 0.0)),
                "leaf_std": float(leaf_stats.get("std", 0.0)),
                "leaf_min": float(leaf_stats.get("min", 0.0)),
                "leaf_max": float(leaf_stats.get("max", 0.0)),
            }
        )

        for item in step.get("top_split_features", []):
            feature_rows.append(
                {
                    "iteration": iteration,
                    "feature_index": item.get("feature_index"),
                    "split_count": int(item.get("split_count", 0)),
                    "avg_split_depth": float(item.get("avg_split_depth", 0.0)),
                }
            )

        for item in step.get("top_feature_interactions", []):
            interaction_rows.append(
                {
                    "iteration": iteration,
                    "left_feature_index": item.get("left_feature_index"),
                    "right_feature_index": item.get("right_feature_index"),
                    "cooccurrence_count": int(item.get("cooccurrence_count", 0)),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("iteration")
    leaf_df = pd.DataFrame(leaf_rows).sort_values("iteration")
    features_df = pd.DataFrame(feature_rows)
    interactions_df = pd.DataFrame(interaction_rows)

    st.subheader("Tree Complexity Over Iterations")
    complexity_options = ["split_count", "leaf_count", "max_depth"]
    selected_metrics = st.multiselect(
        "Metrics to plot",
        options=complexity_options,
        default=["split_count", "max_depth"],
        key="tree_complexity_metrics",
    )

    if selected_metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        for metric in selected_metrics:
            ax.plot(metrics_df["iteration"], metrics_df[metric], label=metric, alpha=0.9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title("Tree Complexity Trajectory")
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, width="content")
    else:
        st.info("Select at least one metric to display the complexity plot.")

    st.subheader("Leaf Value Statistics Over Iterations")
    leaf_metric_options = ["leaf_mean", "leaf_std", "leaf_min", "leaf_max"]
    selected_leaf_metrics = st.multiselect(
        "Leaf stats to plot",
        options=leaf_metric_options,
        default=["leaf_mean", "leaf_std"],
        key="tree_leaf_metrics",
    )

    if selected_leaf_metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        for metric in selected_leaf_metrics:
            ax.plot(leaf_df["iteration"], leaf_df[metric], label=metric, alpha=0.9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title("Leaf Value Dynamics")
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, width="content")

    st.subheader("Most Frequent Split Features")
    top_k_global = st.slider(
        "Top features to show",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
        key="tree_top_k_global",
    )

    if not features_df.empty:
        global_df = (
            features_df.groupby("feature_index", as_index=False)
            .agg(split_count=("split_count", "sum"))
            .sort_values(by="split_count", ascending=False)
            .head(top_k_global)
        )
        global_df = _substitute_feature_column(global_df, "feature_index", feature_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(global_df["feature_index"], global_df["split_count"], alpha=0.9)
        ax.set_xlabel("Total split count")
        ax.set_ylabel("Feature")
        ax.set_title("Global Top Split Features")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, width="content")

        with st.expander("Show aggregated split-feature table"):
            st.dataframe(global_df, width="content")

    st.subheader("Windowed Iteration Details")
    iterations = sorted(metrics_df["iteration"].astype(int).unique().tolist())
    selected_window = st.select_slider(
        "Select iteration window",
        options=iterations,
        value=(iterations[0], iterations[-1]),
        key="tree_selected_iteration_window",
        format_func=lambda iteration: f"Iter {iteration}",
    )
    window_start, window_end = selected_window

    top_k_window = st.slider(
        "Top features to show in window",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
        key="tree_top_k_window",
    )

    window_features_df = features_df[
        (features_df["iteration"] >= window_start) & (features_df["iteration"] <= window_end)
    ].copy()
    window_features_df = (
        window_features_df.groupby("feature_index", as_index=False)
        .agg(split_count=("split_count", "sum"))
        .sort_values(by="split_count", ascending=False)
        .head(top_k_window)
    )
    window_features_df = _substitute_feature_column(
        window_features_df, "feature_index", feature_names
    )

    if not window_features_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(window_features_df["feature_index"], window_features_df["split_count"], alpha=0.9)
        ax.set_xlabel("Total split count in window")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top Split Features (Iterations {window_start} to {window_end})")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, width="content")
    else:
        st.info("No split-feature data found in the selected iteration window.")

    window_interactions_df = interactions_df[
        (interactions_df["iteration"] >= window_start)
        & (interactions_df["iteration"] <= window_end)
    ].copy()
    if not window_interactions_df.empty:
        window_interactions_df["left_feature"] = window_interactions_df["left_feature_index"].apply(
            lambda value: _index_to_name(value, feature_names)
        )
        window_interactions_df["right_feature"] = window_interactions_df["right_feature_index"].apply(
            lambda value: _index_to_name(value, feature_names)
        )
        window_interactions_df["pair"] = (
            window_interactions_df["left_feature"] + " x " + window_interactions_df["right_feature"]
        )
        window_interactions_df = (
            window_interactions_df.groupby("pair", as_index=False)
            .agg(cooccurrence_count=("cooccurrence_count", "sum"))
            .sort_values(by="cooccurrence_count", ascending=False)
            .head(top_k_window)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(
            window_interactions_df["pair"],
            window_interactions_df["cooccurrence_count"],
            alpha=0.9,
        )
        ax.set_xlabel("Total co-occurrence count in window")
        ax.set_ylabel("Feature pair")
        ax.set_title(f"Top Feature Interactions (Iterations {window_start} to {window_end})")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, width="content")

    phase_summaries = tree_json.get("phase_summaries", [])
    if phase_summaries:
        st.subheader("Phase Summaries")
        cols = st.columns(len(phase_summaries))
        for col, phase in zip(cols, phase_summaries):
            with col:
                phase_name = str(phase.get("phase_name", "phase")).title()
                range_values = phase.get("iteration_range", [None, None])
                st.markdown(
                    f"**{phase_name}**  \\nIterations: {range_values[0]} - {range_values[1]}"
                )
                phase_df = pd.DataFrame(phase.get("top_phase_features", []))
                if not phase_df.empty:
                    phase_df = _substitute_feature_column(
                        phase_df,
                        "feature_index",
                        feature_names,
                    )
                    st.dataframe(phase_df, width="content")

    with st.expander("Show raw tree metrics data"):
        st.dataframe(metrics_df, width="content")
