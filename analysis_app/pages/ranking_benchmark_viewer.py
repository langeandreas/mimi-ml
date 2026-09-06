"""Benchmark page: Conditional Importance vs plain SHAP ranking (suggestion 7.7)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from explainer.analysis.cohort_attribution_analyzer import CohortAttributionAnalyzer
from explainer.analysis.trajectory_utils import substitute_feature_names
from explainer.analysis.tree_analyzer import TreeAnalyzer


def _to_feature_rank_df(summary_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = list(summary_payload.get("feature_metrics", []))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("feature_idx", "mean_abs_shap", "conditional_importance_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["feature_idx"]).copy()
    df["feature_idx"] = df["feature_idx"].astype(int)
    return df


def _rank_map(df: pd.DataFrame, score_col: str) -> Dict[int, int]:
    ranked = df.sort_values(score_col, ascending=False)["feature_idx"].tolist()
    return {int(fid): int(i + 1) for i, fid in enumerate(ranked)}


def _build_contextual_conditional_scores(
    rank_df: pd.DataFrame,
    cohorts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Augment conditional score with cohort-decider evidence.

    This avoids degenerate cases where conditional_importance is nearly
    proportional to mean-|SHAP| and therefore yields identical ranking curves.
    """
    out = rank_df.copy()
    if out.empty or cohorts_df.empty:
        out["contextual_conditional_score"] = out.get("conditional_importance_score", 0.0)
        return out

    grp = cohorts_df.groupby("decider_feature_idx", dropna=True).agg(
        cohort_support_sum=("support_count", "sum"),
        cohort_impact_sum=("decision_impact_local", "sum"),
        cohort_weighted_impact=("decision_impact_local", lambda s: float(np.sum(np.asarray(s, dtype=float)))),
        cohort_count=("decider_feature_idx", "count"),
    )
    grp.index = grp.index.astype(int)

    out["cohort_support_sum"] = out["feature_idx"].map(grp["cohort_support_sum"]).fillna(0.0)
    out["cohort_impact_sum"] = out["feature_idx"].map(grp["cohort_impact_sum"]).fillna(0.0)
    out["cohort_count"] = out["feature_idx"].map(grp["cohort_count"]).fillna(0.0)

    support_boost = np.log1p(out["cohort_support_sum"].astype(float))
    impact_boost = np.log1p(np.maximum(out["cohort_impact_sum"].astype(float), 0.0))
    count_boost = np.log1p(out["cohort_count"].astype(float))

    # Normalize boosts for stable combination.
    support_boost = support_boost / max(float(support_boost.max()), 1.0)
    impact_boost = impact_boost / max(float(impact_boost.max()), 1.0)
    count_boost = count_boost / max(float(count_boost.max()), 1.0)

    base = out["conditional_importance_score"].astype(float)
    out["contextual_conditional_score"] = base * (
        1.0
        + 0.55 * impact_boost
        + 0.30 * support_boost
        + 0.15 * count_boost
    )
    return out


def _capture_metrics_for_k(
    cohorts_df: pd.DataFrame,
    feature_set: set[int],
) -> Dict[str, float]:
    if cohorts_df.empty:
        return {
            "cohort_recall": 0.0,
            "support_capture": 0.0,
            "weighted_impact_capture": 0.0,
            "captured_cohorts": 0.0,
            "total_cohorts": 0.0,
        }

    deciders = cohorts_df["decider_feature_idx"].astype(int)
    hit_mask = deciders.isin(feature_set)
    total = float(len(cohorts_df))
    captured = float(hit_mask.sum())

    support_total = float(cohorts_df["support_count"].sum())
    support_hit = float(cohorts_df.loc[hit_mask, "support_count"].sum())

    impact_weight = cohorts_df["support_count"].astype(float) * cohorts_df["decision_impact_local"].astype(float)
    impact_total = float(impact_weight.sum())
    impact_hit = float(impact_weight[hit_mask].sum())

    return {
        "cohort_recall": captured / max(total, 1.0),
        "support_capture": support_hit / max(support_total, 1.0),
        "weighted_impact_capture": impact_hit / max(impact_total, 1.0),
        "captured_cohorts": captured,
        "total_cohorts": total,
    }


def _feature_label(feature_idx: int, feature_names: Sequence[str]) -> str:
    tmp = pd.DataFrame({"feature_index": [int(feature_idx)]})
    out = substitute_feature_names(
        tmp,
        feature_index_col="feature_index",
        feature_names=list(feature_names),
    )
    return str(out.iloc[0]["feature_index"]) if not out.empty else str(feature_idx)


def _normalize_cohort_df(cohorts: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(cohorts)
    if df.empty:
        return df
    if "decision_impact_local" not in df.columns:
        df["decision_impact_local"] = 0.0
    if "support_count" not in df.columns:
        df["support_count"] = 0.0
    if "decider_feature_idx" not in df.columns:
        df["decider_feature_idx"] = -1
    if "scope_signature" not in df.columns:
        df["scope_signature"] = ""
    df["decision_impact_local"] = pd.to_numeric(
        df.get("decision_impact_local"), errors="coerce"
    ).fillna(0.0)
    df["support_count"] = pd.to_numeric(df.get("support_count"), errors="coerce").fillna(0.0)
    df["decider_feature_idx"] = pd.to_numeric(
        df.get("decider_feature_idx"), errors="coerce"
    ).fillna(-1).astype(int)
    df["scope_signature"] = df.get("scope_signature", "").astype(str)
    df["cohort_key"] = (
        df["decider_feature_idx"].astype(str)
        + "||"
        + df["scope_signature"]
    )
    df = df.sort_values(
        ["support_count", "decision_impact_local"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df["cohort_id"] = [f"C{i:03d}" for i in range(1, len(df) + 1)]
    return df


def render_ranking_benchmark_page() -> None:
    """Render the 7.7 benchmark page."""
    st.header("7.7 Benchmark: Conditional Importance vs SHAP")
    st.caption(
        "Compare ranking quality for capturing context-specific decider cohorts. "
        "Discovery is done on training data; capture metrics are evaluated on held-out test data."
    )

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info("No trained model available. Run Setup first.")
        return

    summary_traj = st.session_state.summary_traj
    if not getattr(summary_traj, "model", None):
        st.info("Model not available in session state. Run setup first.")
        return

    tree_entries = getattr(getattr(summary_traj, "explain", None), "trees", None)
    if not tree_entries:
        st.warning("No in-memory tree objects available for cohort benchmark.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        confusion_metric = st.radio(
            "Confusion metric",
            options=["hessian", "fci"],
            index=0,
            horizontal=True,
            key="benchmark_confusion_metric",
        )
    with c2:
        min_support = st.slider("Min cohort support", 10, 500, 60, key="benchmark_min_support")
    with c3:
        min_paths = st.slider("Min path occurrences", 1, 20, 5, key="benchmark_min_paths")

    c4, c5 = st.columns(2)
    with c4:
        max_k = st.slider("Max top-k to benchmark", 5, 40, 25, key="benchmark_max_k")
    with c5:
        min_impact = st.slider(
            "Min decision_impact_local",
            0.0,
            2.0,
            0.01,
            0.005,
            key="benchmark_min_impact",
        )

    X_train = summary_traj.classification.train_test.get("X_train")
    X_test = summary_traj.classification.train_test.get("X_test")
    if X_train is None or X_test is None or len(X_train) == 0 or len(X_test) == 0:
        st.warning("Train/test split data not available for out-of-sample benchmark.")
        return

    with st.spinner("Computing train-discovered rankings and test-set benchmark..."):
        behavior_summary = summary_traj.generate_shap_behavior_correlation_json(
            save=False,
            top_k_features=max(40, max_k),
            confusion_metric=confusion_metric,
        )

        tree_analyzer: TreeAnalyzer = summary_traj.tree_analyzer
        cohort_analyzer = CohortAttributionAnalyzer(tree_analyzer.feature_names)
        rules = cohort_analyzer.extract_path_rules(list(tree_entries))

        train_assigned = cohort_analyzer.assign_samples_to_rules(X_train, rules)
        discovered_train = cohort_analyzer.build_cohorts(
            train_assigned,
            min_support=min_support,
            min_paths=min_paths,
        )
        test_assigned = cohort_analyzer.assign_samples_to_rules(X_test, rules)
        evaluated_test = cohort_analyzer.build_cohorts(
            test_assigned,
            min_support=1,
            min_paths=1,
        )

    rank_df = _to_feature_rank_df(behavior_summary)
    discovered_train_df = _normalize_cohort_df(discovered_train)
    evaluated_test_df = _normalize_cohort_df(evaluated_test)
    if rank_df.empty:
        st.warning("No feature ranking rows found.")
        return
    if discovered_train_df.empty:
        st.warning("No train cohorts produced under current discovery thresholds.")
        return
    if evaluated_test_df.empty:
        st.warning("No test cohorts produced from frozen path rules.")
        return

    discovered_train_df = discovered_train_df[
        discovered_train_df["decision_impact_local"] >= float(min_impact)
    ].copy()
    discovered_keys = set(discovered_train_df["cohort_key"].astype(str).tolist())
    evaluated_test_df = evaluated_test_df[
        evaluated_test_df["cohort_key"].astype(str).isin(discovered_keys)
    ].copy()
    evaluated_test_df = evaluated_test_df[
        evaluated_test_df["decision_impact_local"] >= float(min_impact)
    ].copy()
    if evaluated_test_df.empty:
        st.warning("No discovered train cohorts were recovered on test after filters.")
        return

    rank_df = _build_contextual_conditional_scores(rank_df, discovered_train_df)
    shap_rank = _rank_map(rank_df, "mean_abs_shap")
    cond_rank = _rank_map(rank_df, "contextual_conditional_score")
    feature_names = [str(x) for x in getattr(summary_traj, "feature_names", [])]

    same_order = (
        rank_df.sort_values("mean_abs_shap", ascending=False)["feature_idx"].tolist()
        == rank_df.sort_values("contextual_conditional_score", ascending=False)["feature_idx"].tolist()
    )
    if same_order:
        st.warning(
            "SHAP and contextual conditional rankings are still identical under current train cohorts."
        )

    s1, s2, s3 = st.columns(3)
    s1.metric("Train cohorts discovered", len(discovered_train_df))
    s2.metric("Recovered on test", len(evaluated_test_df))
    s3.metric("Rules extracted", len(rules))

    curve_rows: List[Dict[str, Any]] = []
    for k in range(1, int(max_k) + 1):
        shap_set = {fid for fid, r in shap_rank.items() if r <= k}
        cond_set = {fid for fid, r in cond_rank.items() if r <= k}

        shap_m = _capture_metrics_for_k(evaluated_test_df, shap_set)
        cond_m = _capture_metrics_for_k(evaluated_test_df, cond_set)
        curve_rows.append(
            {
                "k": k,
                "shap_weighted_impact_capture": shap_m["weighted_impact_capture"],
                "cond_weighted_impact_capture": cond_m["weighted_impact_capture"],
                "shap_cohort_recall": shap_m["cohort_recall"],
                "cond_cohort_recall": cond_m["cohort_recall"],
                "shap_support_capture": shap_m["support_capture"],
                "cond_support_capture": cond_m["support_capture"],
            }
        )
    curve_df = pd.DataFrame(curve_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)
    axes[0].plot(curve_df["k"], curve_df["shap_weighted_impact_capture"], label="SHAP rank", lw=2)
    axes[0].plot(curve_df["k"], curve_df["cond_weighted_impact_capture"], label="Conditional rank", lw=2)
    axes[0].set_xlabel("Top-k features")
    axes[0].set_ylabel("Weighted impact capture")
    axes[0].set_title("Impact capture vs k")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(curve_df["k"], curve_df["shap_cohort_recall"], label="SHAP rank", lw=2)
    axes[1].plot(curve_df["k"], curve_df["cond_cohort_recall"], label="Conditional rank", lw=2)
    axes[1].set_xlabel("Top-k features")
    axes[1].set_ylabel("Cohort decider recall")
    axes[1].set_title("Decider cohort recall vs k")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig, width="content")

    selected_k = st.slider("Inspect detailed comparison at k", 1, int(max_k), min(10, int(max_k)), key="benchmark_k")
    shap_set = {fid for fid, r in shap_rank.items() if r <= selected_k}
    cond_set = {fid for fid, r in cond_rank.items() if r <= selected_k}

    shap_m = _capture_metrics_for_k(evaluated_test_df, shap_set)
    cond_m = _capture_metrics_for_k(evaluated_test_df, cond_set)
    uplift = cond_m["weighted_impact_capture"] - shap_m["weighted_impact_capture"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Weighted impact capture (SHAP)", f"{shap_m['weighted_impact_capture']:.3f}")
    m2.metric("Weighted impact capture (Contextual conditional)", f"{cond_m['weighted_impact_capture']:.3f}")
    m3.metric("Conditional uplift", f"{uplift:+.3f}")

    rescued = []
    for _, row in evaluated_test_df.iterrows():
        fid = int(row["decider_feature_idx"])
        in_cond = fid in cond_set
        in_shap = fid in shap_set
        if in_cond and not in_shap:
            rescued.append(
                {
                    "cohort_id": row.get("cohort_id"),
                    "scope_signature": str(row.get("scope_signature", "")),
                    "decider_feature": _feature_label(fid, feature_names),
                    "support_count": int(row.get("support_count", 0)),
                    "decision_impact_local": float(row.get("decision_impact_local", 0.0)),
                    "shap_rank": int(shap_rank.get(fid, 9999)),
                    "conditional_rank": int(cond_rank.get(fid, 9999)),
                }
            )
    rescued_df = pd.DataFrame(rescued).sort_values(
        ["decision_impact_local", "support_count"],
        ascending=[False, False],
    ) if rescued else pd.DataFrame()

    st.subheader("Cohorts rescued by conditional ranking (not in SHAP top-k)")
    if rescued_df.empty:
        st.info("No rescued cohorts at this k.")
    else:
        st.dataframe(rescued_df, width="content")

    with st.expander("Benchmark data (raw)", expanded=False):
        st.dataframe(curve_df, width="content")
