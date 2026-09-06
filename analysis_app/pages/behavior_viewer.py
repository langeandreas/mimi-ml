"""Behavioral correlations viewer page for SHAP and tree dynamics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_app.config import DEFAULT_BEHAVIOR_JSON_PATH
from analysis_app.utils import get_llm_config
from explainer.analysis.trajectory_utils import substitute_feature_names


def _normalize_shap_matrix(shap_values_raw: object) -> np.ndarray:
    """Normalize SHAP outputs to a 2D matrix (samples x features)."""
    if isinstance(shap_values_raw, list):
        shap_matrix = np.asarray(shap_values_raw[-1])
    else:
        shap_matrix = np.asarray(shap_values_raw)

    if shap_matrix.ndim == 1:
        return shap_matrix.reshape(1, -1)
    if shap_matrix.ndim == 2:
        return shap_matrix
    if shap_matrix.ndim == 3:
        return np.asarray(shap_matrix[:, :, -1])
    raise ValueError(f"Unsupported SHAP value shape: {shap_matrix.shape}")


def _load_behavior_json(path: str) -> Dict[str, Any]:
    """Load behavior correlation JSON from file."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _to_numeric(value: Any) -> float:
    """Convert mixed scalar values to float for plotting."""
    if pd.isna(value):
        return 0.0
    try:
        return float(np.real(np.asarray(value).reshape(-1)[0]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _render_selected_feature_metrics(feature_row: pd.Series) -> None:
    """Render key metrics for the selected feature."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Conditional Importance", f"{_to_numeric(feature_row.get('conditional_importance_score')):.4f}")
    with col2:
        st.metric("Mean |SHAP|", f"{_to_numeric(feature_row.get('mean_abs_shap')):.4f}")
    with col3:
        st.metric("SHAP Variation (std)", f"{_to_numeric(feature_row.get('std_abs_shap')):.4f}")
    with col4:
        st.metric("Late Acceleration", f"{_to_numeric(feature_row.get('late_acceleration_mean')):.4f}")


def _plot_global_scatter_with_selection(
    metrics_df: pd.DataFrame,
    selected_feature_idx: int,
    confusion_metric: str,
) -> None:
    """Plot global feature landscape and highlight selected feature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

    x1 = metrics_df["std_abs_shap"].astype(float)
    y1 = metrics_df["late_acceleration_mean"].astype(float)
    c1 = metrics_df["avg_fci"].astype(float)

    scatter1 = axes[0].scatter(x1, y1, c=c1, cmap="viridis", alpha=0.75, edgecolors="black", lw=0.4)
    selected = metrics_df[metrics_df["feature_idx"] == selected_feature_idx]
    if not selected.empty:
        srow = selected.iloc[0]
        axes[0].scatter(
            [float(srow["std_abs_shap"])],
            [float(srow["late_acceleration_mean"])],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=10,
            label="Selected feature",
        )
        axes[0].legend(loc="best")
    axes[0].set_xlabel("Final SHAP variation (std |SHAP|)")
    axes[0].set_ylabel("Late acceleration mean")
    axes[0].set_title("SHAP Variation vs Late Acceleration")
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], pad=0.02)
    cbar1.set_label(_confusion_axis_label(confusion_metric))

    x2 = metrics_df["mean_abs_shap"].astype(float)
    y2 = metrics_df["avg_fci"].astype(float)
    s2 = 30.0 + 170.0 * (metrics_df["conditional_importance_score"].astype(float) / (metrics_df["conditional_importance_score"].astype(float).max() + 1e-9))

    axes[1].scatter(x2, y2, s=s2, alpha=0.65, color="#1f77b4", edgecolors="black", lw=0.4)
    if not selected.empty:
        srow = selected.iloc[0]
        axes[1].scatter(
            [float(srow["mean_abs_shap"])],
            [float(srow["avg_fci"])],
            s=260,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=10,
        )
        axes[1].annotate(
            str(srow["feature_name"]),
            xy=(float(srow["mean_abs_shap"]), float(srow["avg_fci"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
    axes[1].set_xlabel("Final mean |SHAP|")
    axes[1].set_ylabel(f"Average {_confusion_axis_label(confusion_metric)}")
    axes[1].set_title(f"Mean |SHAP| vs {_confusion_axis_label(confusion_metric)} (size=Conditional Importance)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, width="content")


def _plot_selected_feature_live_trajectories(
    summary_traj: Any,
    feature_idx: int,
    confusion_metric: str,
) -> None:
    """Plot selected feature dynamics over iterations from live model artifacts."""
    tree_analyzer = summary_traj.tree_analyzer
    model = summary_traj.model
    if model is None:
        st.info("Model is not available in session state for live trajectory plotting.")
        return

    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if estimator is None or not hasattr(estimator, "get_booster"):
        st.info("Booster is not available for live trajectory plotting.")
        return

    booster = estimator.get_booster()
    trajectory_metrics = tree_analyzer.compute_trajectory_metrics(booster)
    feature_metrics = trajectory_metrics.get(feature_idx)
    if not feature_metrics:
        st.info("Selected feature not present in trajectory metrics.")
        return

    iterations = np.asarray(feature_metrics.get("iterations", []), dtype=float)
    accelerations = np.asarray(feature_metrics.get("acceleration", []), dtype=float)
    fcis = np.asarray(_feature_confusion_series(feature_metrics, confusion_metric), dtype=float)
    cumulative = np.asarray(feature_metrics.get("cumulative_gain", []), dtype=float)

    if iterations.size == 0:
        st.info("No iteration points available for this feature.")
        return

    final_shap_raw = summary_traj.explain.shap_values_history[-1]["shap_values"]
    shap_matrix = _normalize_shap_matrix(final_shap_raw)
    if feature_idx >= shap_matrix.shape[1]:
        st.info("Selected feature index is outside final SHAP matrix columns.")
        return

    selected_abs_shap = np.abs(shap_matrix[:, feature_idx]).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=130)

    axes[0, 0].plot(iterations, accelerations, marker="o", color="#1f77b4", lw=2)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Acceleration by Iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Acceleration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iterations, fcis, marker="s", color="#2ca02c", lw=2)
    axes[0, 1].set_title(f"{_confusion_axis_label(confusion_metric)} by Iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel(_confusion_axis_label(confusion_metric))
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iterations, cumulative, marker="^", color="#9467bd", lw=2)
    axes[1, 0].set_title("Cumulative Gain")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Cumulative gain")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(selected_abs_shap, bins=25, color="#ff7f0e", alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Final |SHAP| Distribution")
    axes[1, 1].set_xlabel("|SHAP|")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, width="content")


def _to_feature_df(summary_payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert summary payload to DataFrame for filtering and plotting."""
    rows = summary_payload.get("feature_metrics", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "feature_idx",
        "mean_abs_shap",
        "std_abs_shap",
        "p90_abs_shap",
        "p10_abs_shap",
        "shap_tail_span",
        "avg_fci",
        "last_fci",
        "early_acceleration_mean",
        "late_acceleration_mean",
        "acceleration_volatility",
        "total_gain",
        "conditional_importance_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "feature_idx" in df.columns:
        df["feature_idx"] = df["feature_idx"].astype(int)
    if "feature_name" in df.columns:
        df["feature_name"] = df["feature_name"].astype(str)

    return df


def _compute_live_behavior_summary(summary_traj: Any, top_k_features: int) -> Optional[Dict[str, Any]]:
    """Compute behavior summary from live model/session state."""
    return _compute_live_behavior_summary_with_metric(
        summary_traj,
        top_k_features=top_k_features,
        confusion_metric="hessian",
    )


def _compute_live_behavior_summary_with_metric(
    summary_traj: Any,
    top_k_features: int,
    confusion_metric: str,
) -> Optional[Dict[str, Any]]:
    """Compute behavior summary from live model/session state."""
    try:
        return summary_traj.generate_shap_behavior_correlation_json(
            save=False,
            top_k_features=top_k_features,
            confusion_metric=confusion_metric,
        )
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Could not compute live behavioral correlations: {exc}")
        return None


def _confusion_axis_label(confusion_metric: str) -> str:
    return "Raw Hessian Mean (HVI signal)" if confusion_metric == "hessian" else "FCI"


def _feature_confusion_series(feature_metrics: Dict[str, Any], confusion_metric: str) -> np.ndarray:
    if confusion_metric == "hessian" and feature_metrics.get("hessian_mean_raw"):
        return np.asarray(feature_metrics.get("hessian_mean_raw"), dtype=float)
    return np.asarray(feature_metrics.get("fci", []), dtype=float)


def _build_selector_df(metrics_df: pd.DataFrame, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
    """Build selector rows with human-readable names and unique labels."""
    selector_df = metrics_df.copy()
    selector_df["feature_index"] = selector_df["feature_idx"]
    selector_df["display_name"] = selector_df["feature_name"].astype(str)

    if feature_names:
        readable_df = substitute_feature_names(
            selector_df[["feature_index"]].copy(),
            feature_index_col="feature_index",
            feature_names=feature_names,
        )
        selector_df["display_name"] = readable_df["feature_index"].astype(str)

    selector_df["selector_label"] = selector_df.apply(
        lambda row: f"{row['display_name']} ({int(row['feature_idx'])})",
        axis=1,
    )
    return selector_df


def _resolve_quadrant_payload(
    summary_payload: Dict[str, Any],
    summary_traj: Any,
    confusion_metric: str,
) -> Optional[Dict[str, Any]]:
    """Resolve quadrant categorization payload from summary or live model."""
    payload = summary_payload.get("quadrant_categorization")
    if isinstance(payload, dict):
        return payload

    if summary_traj is None or summary_traj.model is None:
        return None

    estimator = summary_traj.model.best_estimator_ if hasattr(summary_traj.model, "best_estimator_") else summary_traj.model
    if estimator is None or not hasattr(estimator, "get_booster"):
        return None

    booster = estimator.get_booster()
    trajectory_metrics = summary_traj.tree_analyzer.compute_trajectory_metrics(booster)
    return summary_traj.tree_analyzer.categorize_features_into_quadrants(
        trajectory_metrics,
        confusion_metric=confusion_metric,
    )


def _render_quadrant_categorization_section(
    quadrant_payload: Optional[Dict[str, Any]],
    feature_names: Optional[list[str]] = None,
    confusion_metric: str = "hessian",
) -> None:
    """Render feature quadrant categorization and underlying aggregation data."""
    st.subheader("Quadrant Feature Categorization")
    st.caption(
        f"Categorization is phase-aware: it combines acceleration/{_confusion_axis_label(confusion_metric)} levels with whether those high-signal events "
        "occur mostly early or late in training."
    )
    st.markdown(
        """
Data used for categorization per feature:
- early_acceleration_mean and late_acceleration_mean
- early_high_accel_rate and late_high_accel_rate (share of iterations above high_accel_threshold)
- early_high_fci_rate and late_high_fci_rate (share of iterations above high confusion threshold)
- acceleration_score and fci_score (phase-aware aggregate scores)
- accel_phase_dominance and fci_phase_dominance (early/late/balanced)

Interpretation notes:
- Quadrants encode the sign/level regime (high-vs-low confusion and positive-vs-non-positive acceleration).
- `accel_phase_dominance` and `fci_phase_dominance` encode timing (whether that regime is concentrated early, late, or balanced).
- Always read category + dominance together.
        """
    )

    if not quadrant_payload:
        st.info("No quadrant categorization payload is available.")
        return

    counts = quadrant_payload.get("category_counts", {})
    if counts:
        counts_df = pd.DataFrame(
            [{"category": str(k), "count": int(v)} for k, v in counts.items()]
        ).sort_values("count", ascending=False)
        st.dataframe(counts_df, width="content")

    thresholds = quadrant_payload.get("thresholds", {})
    with st.expander("Thresholds used for categorization", expanded=False):
        if thresholds:
            st.json(thresholds)
        else:
            st.info("No threshold metadata available.")

    features = quadrant_payload.get("features", [])
    features_df = pd.DataFrame(features)
    if features_df.empty:
        st.info("No feature-level category rows available.")
        return

    features_df["feature_idx"] = pd.to_numeric(features_df.get("feature_idx"), errors="coerce").fillna(-1).astype(int)

    if feature_names:
        features_df["feature_index"] = features_df["feature_idx"]
        mapped_df = substitute_feature_names(
            features_df[["feature_index"]].copy(),
            feature_index_col="feature_index",
            feature_names=feature_names,
        )
        features_df["feature_name"] = mapped_df["feature_index"].astype(str)

    category_order = [
        "CONFLICT RESOLVER",
        "HIGH-VARIANCE PATCH",
        "EASY",
        "OUTLIER SPECIALIST",
    ]
    selected_category = st.selectbox(
        "Filter by category",
        options=["All", *category_order],
        index=0,
        key="quadrant_category_filter",
    )

    if selected_category != "All":
        features_df = features_df[features_df["quadrant_category"] == selected_category]

    display_cols = [
        "feature_name",
        "feature_idx",
        "quadrant_category",
        "acceleration_score",
        "fci_score",
        "early_acceleration_mean",
        "late_acceleration_mean",
        "early_high_accel_rate",
        "late_high_accel_rate",
        "early_high_fci_rate",
        "late_high_fci_rate",
        "accel_phase_dominance",
        "fci_phase_dominance",
        "quadrant_reason",
        "quadrant_interpretation",
    ]
    display_cols = [col for col in display_cols if col in features_df.columns]
    st.dataframe(features_df[display_cols].sort_values(["quadrant_category", "feature_name"]), width="content")


def render_behavior_viewer_page() -> None:
    """Render behavioral correlations page with single-feature selection."""
    st.header("Behavioral Correlations")
    confusion_metric = st.radio(
        "Confusion metric",
        options=["hessian", "fci"],
        index=0,
        horizontal=True,
        key="behavior_confusion_metric",
    )
    st.caption(
        "Inspect correlations between final SHAP variation and tree dynamics "
        f"({_confusion_axis_label(confusion_metric)} and acceleration)."
    )

    config = get_llm_config()
    behavior_path = str(config.get("behavior_json_path", DEFAULT_BEHAVIOR_JSON_PATH))

    source_mode = st.radio(
        "Data source",
        options=["Live session model", "Load JSON file"],
        index=0,
        horizontal=True,
        key="behavior_source_mode",
    )

    top_k_features = st.slider(
        "Top K rows for ranked findings",
        min_value=5,
        max_value=40,
        value=15,
        step=1,
        key="behavior_top_k_features",
    )

    summary_payload: Optional[Dict[str, Any]] = None

    if source_mode == "Live session model":
        if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
            st.info("No trained model in session. Run Setup first or switch to JSON mode.")
            return

        with st.spinner("Computing SHAP-behavior findings from current session..."):
            summary_payload = _compute_live_behavior_summary_with_metric(
                st.session_state.summary_traj,
                top_k_features=top_k_features,
                confusion_metric=confusion_metric,
            )
        if summary_payload is None:
            return

        if st.button("Save latest findings JSON", key="save_behavior_findings_json"):
            try:
                saved_payload = st.session_state.summary_traj.generate_shap_behavior_correlation_json(
                    save=True,
                    top_k_features=top_k_features,
                    confusion_metric=confusion_metric,
                )
                st.success("Saved behavioral findings to trajectory_shap_behavior_correlation.json")
                summary_payload = saved_payload
            except Exception as exc:  # pragma: no cover - UI error path
                st.error(f"Failed to save findings JSON: {exc}")
    else:
        behavior_path = st.text_input(
            "Path to behavior-correlation JSON",
            value=behavior_path,
            key="behavior_json_path_input",
        )
        config["behavior_json_path"] = behavior_path
        try:
            summary_payload = _load_behavior_json(behavior_path)
        except FileNotFoundError:
            st.error(f"Behavior JSON not found: {behavior_path}")
            return
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON format: {exc}")
            return

    if summary_payload is None:
        st.warning("No findings available to display.")
        return

    metrics_df = _to_feature_df(summary_payload)
    if metrics_df.empty:
        st.warning("No feature-level findings found in the summary payload.")
        return

    feature_names_for_substitution: Optional[list[str]] = None
    if "summary_traj" in st.session_state and st.session_state.summary_traj is not None:
        feature_names_for_substitution = [
            str(name)
            for name in getattr(st.session_state.summary_traj, "feature_names", [])
        ]
    summary_traj = st.session_state.summary_traj if "summary_traj" in st.session_state else None

    selector_df = _build_selector_df(metrics_df, feature_names=feature_names_for_substitution)
    metrics_df = selector_df

    st.subheader("Single Feature Inspector")
    ordered_labels = (
        metrics_df.sort_values("conditional_importance_score", ascending=False)["selector_label"].tolist()
    )
    selected_feature_label = st.selectbox(
        "Select one feature",
        options=ordered_labels,
        index=0,
        key="behavior_selected_feature_name",
    )

    selected_row_df = metrics_df[metrics_df["selector_label"] == selected_feature_label]
    if selected_row_df.empty:
        st.warning("Selected feature was not found in findings.")
        return

    selected_row = selected_row_df.iloc[0]
    selected_feature_idx = int(selected_row["feature_idx"])

    _render_selected_feature_metrics(selected_row)
    _plot_global_scatter_with_selection(
        metrics_df,
        selected_feature_idx,
        confusion_metric=confusion_metric,
    )

    st.subheader("Selected Feature Dynamics")
    if source_mode == "Live session model":
        _plot_selected_feature_live_trajectories(
            st.session_state.summary_traj,
            selected_feature_idx,
            confusion_metric=confusion_metric,
        )
    else:
        st.info("Per-iteration dynamics require live session model artifacts. Switch to Live session mode for trajectory plots.")

    st.subheader("Ranked Findings")
    ranking_options = {
        "Top Conditional Importance": "top_conditional_importance",
        "Top SHAP Variation": "top_shap_variation",
        "Top Late Acceleration": "top_late_acceleration",
        "High SHAP and High Late Acceleration": "high_shap_high_late_acceleration",
    }
    selected_ranking_label = st.selectbox(
        "Ranking table",
        options=list(ranking_options.keys()),
        index=0,
        key="behavior_ranking_selection",
    )
    ranking_key = ranking_options[selected_ranking_label]
    ranking_rows = summary_payload.get("rankings", {}).get(ranking_key, [])
    ranking_df = pd.DataFrame(ranking_rows)
    if ranking_df.empty:
        st.info("No rows available for this ranking.")
    else:
        st.dataframe(ranking_df, width="content")

    with st.expander("Correlation focus pairs", expanded=False):
        focus_pairs = summary_payload.get("correlations", {}).get("focus_pairs", {})
        if not focus_pairs:
            st.info("No focus-pair correlation data in payload.")
        else:
            st.json(focus_pairs)

    with st.expander("All feature metrics", expanded=False):
        st.dataframe(metrics_df.sort_values("conditional_importance_score", ascending=False), width="content")

    quadrant_payload = _resolve_quadrant_payload(
        summary_payload,
        summary_traj,
        confusion_metric=confusion_metric,
    )
    _render_quadrant_categorization_section(
        quadrant_payload=quadrant_payload,
        feature_names=feature_names_for_substitution,
        confusion_metric=confusion_metric,
    )
