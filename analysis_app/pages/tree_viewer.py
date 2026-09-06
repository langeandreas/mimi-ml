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
from explainer.analysis.tree_analyzer import TreeAnalyzer


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


def _feature_idx_to_readable_name(feature_idx: int, feature_names: List[str]) -> str:
    """Resolve feature index to readable feature label."""
    tmp_df = pd.DataFrame({"feature_index": [int(feature_idx)]})
    tmp_df = _substitute_feature_column(tmp_df, "feature_index", feature_names)
    if tmp_df.empty:
        return str(feature_idx)
    return str(tmp_df.iloc[0]["feature_index"])


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
    _leaf_df = pd.DataFrame(leaf_rows).sort_values("iteration")
    features_df = pd.DataFrame(feature_rows)
    interactions_df = pd.DataFrame(interaction_rows)


    # Behavioral Signatures Section (requires model access from session state)
    st.subheader("Feature Behavioral Signatures")
    confusion_metric = st.radio(
        "Confusion metric",
        options=["hessian", "fci"],
        index=0,
        horizontal=True,
        key="behavioral_confusion_metric",
    )
    confusion_label = (
        "Raw Hessian mean (HVI signal)" if confusion_metric == "hessian" else "Feature Confusion Index (FCI)"
    )
    st.caption(
        f"Analyze Feature Acceleration and {confusion_label} to reveal model learning patterns."
    )
    
    if "summary_traj" in st.session_state and st.session_state.summary_traj is not None:
        summary_traj = st.session_state.summary_traj
        
        if hasattr(summary_traj, "model") and summary_traj.model is not None:
            try:
                # Compute trajectory metrics from the booster
                with st.spinner("Computing trajectory metrics..."):
                    tree_analyzer = TreeAnalyzer(summary_traj.classification)
                    
                    # Get the booster (handle both GridSearchCV and direct XGBClassifier)
                    model = summary_traj.model
                    if hasattr(model, "best_estimator_"):
                        booster = model.best_estimator_.get_booster()
                    else:
                        booster = model.get_booster()
                    
                    trajectory_metrics = tree_analyzer.compute_trajectory_metrics(booster)
                
                if trajectory_metrics:
                    # Display behavioral signatures table
                    behavioral_data = []
                    for feature_idx in sorted(trajectory_metrics.keys()):
                        signature = tree_analyzer.get_behavioral_signature(
                            trajectory_metrics,
                            feature_idx,
                            confusion_metric=confusion_metric,
                        )
                        behavioral_data.append(signature)
                    
                    sig_df = pd.DataFrame(behavioral_data)
                    if not sig_df.empty:
                        sig_df = sig_df[
                            [
                                'feature_name',
                                'signature',
                                'total_gain',
                                'avg_fci',
                                'early_acceleration_mean',
                                'late_acceleration_mean',
                                'num_iterations',
                            ]
                        ].rename(columns={"avg_fci": f"avg_confusion ({confusion_metric})"})
                        with st.expander("Show behavioral signatures table", expanded=False):
                            st.dataframe(sig_df, width="content")
                    
                    # Feature selection controls with human-readable names
                    st.subheader("Feature Selection")
                    
                    # Create DataFrame for substitution
                    feature_df = pd.DataFrame({
                        'feature_idx': list(trajectory_metrics.keys()),
                        'codename': [metrics['feature_name'] for metrics in trajectory_metrics.values()]
                    })
                    
                    # Substitute codenames with human-readable names
                    feature_df = substitute_feature_names(
                        feature_df,
                        feature_index_col='codename',
                        feature_names=feature_df['codename'].unique().tolist()
                    )
                    
                    # Create mappings
                    feature_idx_to_readable = dict(zip(feature_df['feature_idx'], feature_df['codename']))
                    readable_to_idx = {v: k for k, v in feature_idx_to_readable.items()}
                    
                    # Create sorted display options
                    feature_display_options = sorted(set(feature_df['codename'].tolist()))
                    
                    all_features_option = st.checkbox("Show all features", value=True, key="behavioral_show_all_features")
                    
                    if all_features_option:
                        selected_feature_indices = list(trajectory_metrics.keys())
                    else:
                        selected_feature_display = st.multiselect(
                            "Select features to display",
                            options=feature_display_options,
                            default=feature_display_options[:5],  # Default to first 5
                            key="behavioral_selected_features",
                        )
                        # Map selected readable names back to feature indices
                        selected_feature_indices = [readable_to_idx[readable] for readable in selected_feature_display]

                    metrics_for_sampling = (
                        trajectory_metrics.values()
                        if all_features_option
                        else [trajectory_metrics[idx] for idx in selected_feature_indices]
                    )
                    max_points_per_feature = max(
                        (len(metrics.get("iterations", [])) for metrics in metrics_for_sampling),
                        default=1,
                    )
                    points_per_feature = st.slider(
                        "Behavioral signature dots per feature",
                        min_value=1,
                        max_value=max_points_per_feature,
                        value=max_points_per_feature,
                        step=1,
                        key="behavioral_points_per_feature",
                        help=(
                            "Select how many phase-contracted points to show per feature. "
                            "For example, 3 points show representative points from the first, "
                            "second, and third thirds of training."
                        ),
                    )
                    
                    # Display the behavioral signatures plot with selected features
                    with st.spinner("Generating behavioral signatures visualization..."):
                        fig, _ = tree_analyzer.plot_behavioral_signatures(
                            trajectory_metrics,
                            figsize=(20, 12),
                            save_path=None,
                            selected_features=selected_feature_indices if not all_features_option else None,
                            points_per_feature=points_per_feature,
                            confusion_metric=confusion_metric,
                        )
                        st.pyplot(fig, width="content", use_container_width=True)
                        
                        # Display interpretation guide
                        with st.expander("How to interpret these plots", expanded=False):
                            st.markdown("""
                            **Plot 1: Behavioral Signatures (Acceleration vs Confusion)**
                            - **Each dot is one training-phase sample** for a feature (not an all-time average).
                            - X-axis (**Confusion signal**): currently selected metric in the controls above.
                              - High values: split is resolving broader, noisier uncertainty mass.
                              - Low values: split is refining narrower or cleaner local slices.
                            - Y-axis (**Acceleration**): slope change in cumulative gain at that moment.
                              - Positive: contribution is ramping up.
                              - Negative: contribution is tapering off or consolidating.
                            - Dot color encodes **time** (lighter = early, darker = late), so timing must be read jointly with position.
                            
                            **Quadrants (read with color/time)**
                            - **CONFLICT RESOLVER** (high confusion, +acceleration): feature is increasingly used to resolve hard/global uncertainty.
                              - Light dots: early emergence.
                              - Dark dots: late-stage conflict escalation.
                            - **HIGH-VARIANCE PATCH** (low confusion, +acceleration): feature is increasingly active on local/heterogeneous pockets.
                              - Light dots: early local patching.
                              - Dark dots: late residual cleanup.
                            - **EASY** (high confusion, -acceleration): feature was used on broad uncertainty but is now stabilizing/decaying.
                              - Light dots: early coarse-fit settling.
                              - Dark dots: late global stabilization.
                            - **OUTLIER SPECIALIST** (low confusion, -acceleration): feature is tapering on local edge cases.
                              - Light dots: early pruning of weak local effects.
                              - Dark dots: late tail calibration/saturation.
                            
                            **Plot 2: Feature Acceleration Over Time**
                            - Shows where each feature ramps up versus cools down across training.
                            - Persistent sign flips suggest role-switching or collinearity.
                            
                            **Plot 3: Confusion Signal Over Time**
                            - Tracks whether each feature is acting more globally (higher confusion) or locally (lower confusion) over time.
                            - Rising curve: migration toward broader uncertainty resolution.
                            - Falling curve: migration toward local/tail refinement.
                            
                            **Plot 4: Cumulative Gain Over Training**
                            - Shows total optimization contribution of each feature
                            - Steeper curves: More important features
                            - Flat curves: Marginal features
                            """)

                    st.subheader("Feature Decision-Path Context Explorer")
                    st.caption(
                        "Follow each feature across individual tree paths to see where it acts as a late decider "
                        "and which earlier scope-defining features precede it."
                    )

                    if hasattr(summary_traj, "explain") and getattr(summary_traj.explain, "trees", None):
                        with st.spinner("Analyzing individual decision-tree paths..."):
                            path_context = tree_analyzer.analyze_feature_decision_contexts(
                                tree_entries=summary_traj.explain.trees,
                                trajectory_metrics=trajectory_metrics,
                                top_k=10,
                            )

                        profiles_df = pd.DataFrame(path_context.get("feature_profiles", []))
                        if profiles_df.empty:
                            st.info("No feature path-context profiles were extracted.")
                        else:
                            profiles_df["feature_idx"] = pd.to_numeric(
                                profiles_df["feature_idx"], errors="coerce"
                            ).fillna(-1).astype(int)
                            profiles_df["feature_readable"] = profiles_df["feature_idx"].apply(
                                lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                            )
                            profiles_df["selector_label"] = profiles_df.apply(
                                lambda row: f"{row['feature_readable']} ({int(row['feature_idx'])})",
                                axis=1,
                            )

                            selected_label = st.selectbox(
                                "Inspect feature path behavior",
                                options=profiles_df.sort_values(
                                    ["late_decider_rate", "path_count"], ascending=False
                                )["selector_label"].tolist(),
                                key="tree_path_context_feature_selector",
                            )

                            selected_row = profiles_df[
                                profiles_df["selector_label"] == selected_label
                            ].iloc[0]

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Path count", int(selected_row["path_count"]))
                            c2.metric("Late decider rate", f"{float(selected_row['late_decider_rate']):.3f}")
                            c3.metric("Decision impact score", f"{float(selected_row.get('decision_impact_score', 0.0)):.3f}")
                            c4.metric("Scope entropy", f"{float(selected_row['scope_entropy']):.3f}")

                            d1, d2, d3, d4 = st.columns(4)
                            d1.metric("Late-decider in late training", f"{float(selected_row['late_in_late_training_rate']):.3f}")
                            d2.metric("Mean |leaf| when late decider", f"{float(selected_row.get('mean_abs_leaf_when_late_decider', 0.0)):.4f}")
                            d3.metric("Impact magnitude ratio", f"{float(selected_row.get('impact_magnitude_ratio', 0.0)):.3f}")
                            d4.metric("Mean leaf when present", f"{float(selected_row.get('mean_leaf_value_when_present', 0.0)):.4f}")

                            st.caption(
                                "Top scope-defining features are those that repeatedly appear before the selected "
                                "feature's late split point along root-to-leaf paths."
                            )

                            scope_rows = selected_row.get("top_scope_features", [])
                            scope_df = pd.DataFrame(scope_rows)
                            if not scope_df.empty:
                                scope_df["feature_idx"] = pd.to_numeric(scope_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
                                scope_df["scope_feature"] = scope_df["feature_idx"].apply(
                                    lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                                )
                                scope_df = scope_df[["scope_feature", "cooccurrence_count", "cooccurrence_rate"]]
                                st.dataframe(scope_df, width="content")
                            else:
                                st.info("No dominant scope-defining features found for this feature.")

                            partner_rows = selected_row.get("top_near_leaf_partners", [])
                            partner_df = pd.DataFrame(partner_rows)
                            if not partner_df.empty:
                                partner_df["feature_idx"] = pd.to_numeric(partner_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
                                partner_df["near_leaf_partner"] = partner_df["feature_idx"].apply(
                                    lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                                )
                                partner_df = partner_df[["near_leaf_partner", "cooccurrence_count", "cooccurrence_rate"]]
                                with st.expander("Near-leaf co-decider partners", expanded=False):
                                    st.dataframe(partner_df, width="content")

                            candidate_rows = path_context.get("outlier_specialist_candidates", [])
                            candidates_df = pd.DataFrame(candidate_rows)
                            if not candidates_df.empty:
                                candidates_df["feature_idx"] = pd.to_numeric(candidates_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
                                candidates_df["feature_name"] = candidates_df["feature_idx"].apply(
                                    lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                                )
                                st.markdown("**Outlier specialist candidates**")
                                st.dataframe(
                                    candidates_df[
                                        [
                                            "feature_name",
                                            "path_count",
                                            "late_decider_rate",
                                            "decision_impact_score",
                                            "mean_abs_leaf_when_late_decider",
                                            "late_in_late_training_rate",
                                            "scope_entropy",
                                        ]
                                    ],
                                    width="content",
                                )

                            pair_rows = path_context.get("top_scope_to_decider_pairs", [])
                            pairs_df = pd.DataFrame(pair_rows)
                            if not pairs_df.empty:
                                pairs_df["scope_feature"] = pairs_df["scope_feature_idx"].apply(
                                    lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                                )
                                pairs_df["decider_feature"] = pairs_df["decider_feature_idx"].apply(
                                    lambda idx: _feature_idx_to_readable_name(int(idx), tree_analyzer.feature_names)
                                )
                                with st.expander("Global scope → decider co-occurrence pairs", expanded=False):
                                    st.dataframe(
                                        pairs_df[["scope_feature", "decider_feature", "cooccurrence_count"]],
                                        width="content",
                                    )
                    else:
                        st.info("No in-memory tree objects available for path-context analysis.")
                else:
                    st.warning("No trajectory metrics computed. Try re-running the setup.")
            except Exception as e:
                st.error(f"Error computing trajectory metrics: {e}")
        else:
            st.info("Model not available in session state. Run setup first to compute behavioral signatures.")
    else:
        st.info("Summary trajectory not loaded. Run the Setup page first to analyze behavioral signatures.")




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

    window_interactions_df = _substitute_feature_column(
        window_interactions_df, "left_feature_index", feature_names)
    window_interactions_df = _substitute_feature_column(
        window_interactions_df, "right_feature_index", feature_names)
    
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
