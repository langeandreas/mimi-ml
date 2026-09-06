"""Cohort attribution page for sub-population explainability."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from explainer.analysis.trajectory_utils import substitute_feature_names
from explainer.analysis.tree_analyzer import TreeAnalyzer


def _resolve_booster(model: Any):
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_.get_booster()
    return model.get_booster()


def _apply_readable_feature_labels(df: pd.DataFrame, col: str, feature_names: list[str]) -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out = substitute_feature_names(out, feature_index_col=col, feature_names=feature_names)
    out[col] = out[col].astype(str)
    return out


def render_cohort_attribution_page() -> None:
    st.header("Cohort Attribution")
    st.caption(
        "Identify sub-populations defined by repeated scope conditions where a "
        "feature acts as a final decider."
    )

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info("No trained model available. Run the Setup page first.")
        return

    summary_traj = st.session_state.summary_traj
    if not getattr(summary_traj, "model", None):
        st.info("Model not available in session state. Run setup first.")
        return

    tree_entries = getattr(getattr(summary_traj, "explain", None), "trees", None)
    if not tree_entries:
        st.warning("No in-memory tree objects available for cohort analysis.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        min_support = st.slider("Minimum cohort support", 10, 500, 60, key="cohort_min_support")
    with c2:
        min_paths = st.slider("Minimum path occurrences", 1, 30, 5, key="cohort_min_paths")
    with c3:
        top_k = st.slider("Top cohorts", 5, 50, 20, key="cohort_top_k")

    include_local_shap = st.checkbox(
        "Include local SHAP summaries",
        value=False,
        key="cohort_include_local_shap",
        help="Computes final SHAP values on training data; can take longer.",
    )

    try:
        with st.spinner("Building cohort attribution payload..."):
            tree_analyzer = TreeAnalyzer(summary_traj.classification)
            _ = tree_analyzer.compute_trajectory_metrics(_resolve_booster(summary_traj.model))

            final_shap_values = None
            if include_local_shap:
                estimator = (
                    summary_traj.model.best_estimator_
                    if hasattr(summary_traj.model, "best_estimator_")
                    else summary_traj.model
                )
                final_shap_values = summary_traj.classification.shap_values(estimator)

            payload = tree_analyzer.analyze_subpopulation_cohorts(
                tree_entries=tree_entries,
                model=summary_traj.model,
                final_shap_values=final_shap_values,
                min_support=min_support,
                min_paths=min_paths,
                top_k=top_k,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error generating cohort attribution: {exc}")
        return

    cohorts = payload.get("cohorts", [])
    if not cohorts:
        st.info("No cohorts met the thresholds. Lower minimum support or path count.")
        return

    st.metric("Cohorts", payload.get("metadata", {}).get("cohort_count", 0))

    cohort_df = pd.DataFrame(cohorts)
    if "decider_feature_idx" in cohort_df.columns:
        cohort_df = _apply_readable_feature_labels(
            cohort_df.rename(columns={"decider_feature_idx": "decider_feature"}),
            "decider_feature",
            tree_analyzer.feature_names,
        )
    else:
        cohort_df["decider_feature"] = cohort_df.get("decider_feature_name", "")

    roster_cols = [
        "cohort_id",
        "decider_feature",
        "support_count",
        "support_rate",
        "path_occurrences",
        "risk_rate",
        "mean_confidence",
        "decision_impact_local",
        "impact_magnitude_ratio_local",
        "decision_impact_local_raw",
        "late_decider_rate_local",
    ]
    roster_cols = [c for c in roster_cols if c in cohort_df.columns]
    st.subheader("Cohort roster")
    st.dataframe(
        cohort_df[roster_cols].sort_values(
            ["support_count", "decision_impact_local"],
            ascending=[False, False],
        ),
        width="content",
    )

    selector = st.selectbox(
        "Inspect cohort",
        options=cohort_df["cohort_id"].tolist(),
        key="cohort_selector",
    )
    selected = cohort_df[cohort_df["cohort_id"] == selector].iloc[0].to_dict()

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Support", int(selected.get("support_count", 0)))
    d2.metric("Risk rate", f"{float(selected.get('risk_rate', 0.0)):.3f}")
    d3.metric("Local impact (scaled)", f"{float(selected.get('decision_impact_local', 0.0)):.3f}")
    d4.metric("Impact ratio (vs global leaf)", f"{float(selected.get('impact_magnitude_ratio_local', 0.0)):.3f}")
    d5.metric("Late-decider local rate", f"{float(selected.get('late_decider_rate_local', 0.0)):.3f}")
    st.caption(
        f"Raw local impact (legacy scale): {float(selected.get('decision_impact_local_raw', 0.0)):.6f}"
    )

    st.markdown("**Scope signature**")
    st.code(str(selected.get("scope_signature", "")))

    top_paths = selected.get("top_paths", [])
    if top_paths:
        with st.expander("Top supporting paths", expanded=False):
            path_df = pd.DataFrame(top_paths)
            if "support_count" in path_df.columns:
                path_df = path_df.sort_values("support_count", ascending=False)
            st.dataframe(path_df, width="content")

    local_shap = selected.get("local_shap_summary", [])
    if local_shap:
        shap_df = pd.DataFrame(local_shap)
        if "feature_idx" in shap_df.columns:
            shap_df = _apply_readable_feature_labels(
                shap_df.rename(columns={"feature_idx": "feature"}),
                "feature",
                tree_analyzer.feature_names,
            )
        st.markdown("**Local SHAP summary (within cohort)**")
        st.dataframe(shap_df, width="content")

    with st.expander("Raw cohort JSON", expanded=False):
        st.json(
            {
                "metadata": payload.get("metadata", {}),
                "selected_cohort": selected,
            }
        )
