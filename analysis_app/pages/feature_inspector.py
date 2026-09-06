"""Feature Inspector page for SHAP-first deep dive analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from analysis_app.utils import get_llm_config
from explainer.analysis.cohort_attribution_analyzer import CohortAttributionAnalyzer
from explainer.analysis.trajectory_utils import substitute_feature_names


# Keep inspector-side signature filtering aligned with analyzer calibration.
_SIGNATURE_LEVEL_EPS_MULT = 0.10
_SIGNATURE_OUTLIER_MIN_FRAC = 0.25
_SIGNATURE_LATE_EARLY_SOFT_LIMIT = 0.50
_SIGNATURE_DELTA_EPS_MULT = 0.35
_SIGNATURE_OUTLIER_LATE_MIN_FRAC = 0.35
_SIGNATURE_UNSTABLE_REL_MULT = 0.85


FIELD_DESCRIPTIONS = {
    "mean_abs_shap": "Average absolute SHAP value at final iteration; higher means stronger average feature influence.",
    "std_abs_shap": "Variation of absolute SHAP values across samples; higher means more heterogeneous influence.",
    "p90_abs_shap": "90th percentile of absolute SHAP values; reflects high-impact tail behavior.",
    "p10_abs_shap": "10th percentile of absolute SHAP values; reflects low-impact tail behavior.",
    "conditional_importance_score": "Composite importance score combining final SHAP magnitude with behavioral dynamics.",
    "path_count": "Number of root-to-leaf paths where the selected feature appears.",
    "late_decider_rate": "Fraction of those paths where the feature appears in the last third of split decisions.",
    "decision_impact_score": "Composite path-context impact score combining late-decider frequency and leaf-value impact magnitude.",
    "late_in_late_training_rate": "Among late-decider occurrences, the share happening in later boosting iterations.",
    "scope_entropy": "Diversity of scope-defining predecessor features; lower means more consistent preconditions.",
    "mean_abs_leaf_when_late_decider": "Average absolute terminal leaf value on paths where the feature acts as late decider.",
    "impact_magnitude_ratio": "Leaf-impact magnitude relative to global mean absolute leaf magnitude baseline.",
    "mean_leaf_value_when_late_decider": "Signed average leaf value on late-decider paths; indicates direction of impact.",
    "early_acceleration_mean": "Mean acceleration in early training iterations.",
    "late_acceleration_mean": "Mean acceleration in late training iterations.",
    "early_high_fci_rate": "Fraction of early iterations where selected confusion metric exceeds high threshold.",
    "late_high_fci_rate": "Fraction of late iterations where selected confusion metric exceeds high threshold.",
    "activation_score": "Composite activation signal used for quadrant assignment (late acceleration, acceleration shift, gain signal, minus instability penalty).",
    "confusion_composite_score": "Composite confusion signal used for quadrant assignment (level and phase shift of selected confusion metric).",
    "instability_penalty": "Penalty that reduces activation score when acceleration/Hessian volatility is high.",
    "quadrant_category": "Assigned role from the activation/confusion split (median dataset split on each axis).",
    "quadrant_reason": "Trace line with score values, split thresholds, and phase-dominance labels used in assignment.",
    "signature": "Narrative signature summarizing early-vs-late acceleration behavior and volatility pattern.",
}


def _confusion_label(confusion_metric: str) -> str:
    return "Raw Hessian Mean (HVI signal)" if confusion_metric == "hessian" else "FCI"


def _confusion_series(metrics: Dict[str, Any], confusion_metric: str) -> np.ndarray:
    if confusion_metric == "hessian" and metrics.get("hessian_mean_raw"):
        return np.asarray(metrics.get("hessian_mean_raw", []), dtype=float)
    return np.asarray(metrics.get("fci", []), dtype=float)


def _compute_signature_label(
    early_accel: float,
    late_accel: float,
    volatility: float,
    acceleration_score: float,
    level_eps: float,
    delta_eps: float,
    unstable_vol_floor: float,
) -> str:
    """Apply signature rules with profile-calibrated deadbands."""
    eps = max(float(level_eps), 1e-6)
    shift_eps = max(float(delta_eps), eps)
    unstable_floor = max(float(unstable_vol_floor), eps)

    rel_scale = max(abs(acceleration_score), abs(early_accel), abs(late_accel), 1e-9)
    has_core_pattern = early_accel > eps and late_accel < -eps and (early_accel - late_accel) > shift_eps
    has_late_pattern = (
        late_accel > eps
        and (late_accel - early_accel) > shift_eps
        and early_accel < (_SIGNATURE_LATE_EARLY_SOFT_LIMIT * eps)
    )
    is_unstable = volatility > unstable_floor and volatility > (_SIGNATURE_UNSTABLE_REL_MULT * rel_scale)
    has_outlier_pattern = (
        early_accel > eps
        and late_accel > (_SIGNATURE_OUTLIER_LATE_MIN_FRAC * eps)
        and min(early_accel, late_accel) > (_SIGNATURE_OUTLIER_MIN_FRAC * eps)
    )

    if has_core_pattern:
        return "CORE DRIVER: Early optimization, then stabilization"
    if has_late_pattern:
        return "LATE LEARNER: Becomes important in later iterations"
    if has_outlier_pattern:
        return "OUTLIER SPECIALIST: Continuous late-stage optimization"
    if is_unstable:
        return "UNSTABLE: Oscillating importance (possible collinearity)"
    return "STABLE BASELINE: Consistent contribution throughout"


def _build_signature_lookup(
    quadrant_payload: Dict[str, Any],
    trajectory_metrics: Dict[int, Dict[str, Any]],
) -> Dict[int, str]:
    """Build feature_idx -> signature label map from profile metrics."""
    lookup: Dict[int, str] = {}
    thresholds = quadrant_payload.get("thresholds", {}) if isinstance(quadrant_payload, dict) else {}
    level_eps = float(thresholds.get("signature_level_eps", 0.0))
    delta_eps = float(thresholds.get("signature_delta_eps", 0.0))
    unstable_vol_floor = float(thresholds.get("signature_unstable_vol_floor", 0.0))

    # Fallback for older payloads without signature calibration keys.
    if level_eps <= 0.0 or delta_eps <= 0.0:
        high_accel_threshold = float(thresholds.get("high_accel_threshold", 0.0))
        phase_margin = float(thresholds.get("phase_dominance_margin", 0.15))
        base_scale = max(abs(high_accel_threshold), 1e-9)
        level_eps = max(_SIGNATURE_LEVEL_EPS_MULT * base_scale, 1e-6)
        delta_eps = max(_SIGNATURE_DELTA_EPS_MULT * base_scale, phase_margin * base_scale, level_eps)
        unstable_vol_floor = max(level_eps, unstable_vol_floor)

    for feature in list(quadrant_payload.get("features", [])):
        idx = int(feature.get("feature_idx", -1))
        if idx < 0:
            continue
        early_accel = float(feature.get("early_acceleration_mean", 0.0))
        late_accel = float(feature.get("late_acceleration_mean", 0.0))
        acceleration_score = float(feature.get("acceleration_score", 0.0))
        accelerations = np.asarray(trajectory_metrics.get(idx, {}).get("acceleration", []), dtype=float)
        volatility = float(np.std(accelerations)) if accelerations.size else 0.0
        lookup[idx] = _compute_signature_label(
            early_accel=early_accel,
            late_accel=late_accel,
            volatility=volatility,
            acceleration_score=acceleration_score,
            level_eps=level_eps,
            delta_eps=delta_eps,
            unstable_vol_floor=unstable_vol_floor,
        )
    return lookup


def _render_field_glossary() -> None:
    """Render a compact glossary describing metrics used in this page."""
    with st.expander("Field glossary", expanded=False):
        glossary_df = pd.DataFrame(
            [{"field": key, "meaning": value} for key, value in FIELD_DESCRIPTIONS.items()]
        )
        st.dataframe(glossary_df, width="content")


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


def _readable_name_for_idx(feature_idx: int, feature_names: list[str]) -> str:
    """Map a feature index to a human-readable feature name."""
    tmp_df = pd.DataFrame({"feature_index": [int(feature_idx)]})
    tmp_df = substitute_feature_names(
        tmp_df,
        feature_index_col="feature_index",
        feature_names=feature_names,
    )
    if tmp_df.empty:
        return str(feature_idx)
    return str(tmp_df.iloc[0]["feature_index"])


def _build_feature_selector_df(metrics_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Build selector labels with readable names while preserving stable IDs."""
    out = metrics_df.copy()
    out["feature_idx"] = pd.to_numeric(out["feature_idx"], errors="coerce").fillna(-1).astype(int)
    out["readable_feature_name"] = out["feature_idx"].apply(lambda idx: _readable_name_for_idx(int(idx), feature_names))
    out["selector_label"] = out.apply(
        lambda row: f"{row['readable_feature_name']} ({int(row['feature_idx'])})",
        axis=1,
    )
    return out


def _render_top_shap_summary(summary_traj: Any) -> Dict[str, Any]:
    """Render top SHAP view and return raw values for downstream drill-down."""
    shap_history = list(getattr(summary_traj.explain, "shap_values_history", []))
    if not shap_history:
        st.warning("No SHAP history found in session. Run Setup first.")
        return {}

    shap_history = sorted(shap_history, key=lambda item: int(item["iteration"]))
    final_entry = shap_history[-1]
    final_iter = int(final_entry["iteration"])

    st.subheader("1) Final SHAP Snapshot")
    st.caption("Start from final SHAP importance, then drill down into why a feature is more or less important.")

    shap_matrix = _normalize_shap_matrix(final_entry["shap_values"])
    callback_matrix = np.asarray(summary_traj.explain.X_train)
    if callback_matrix.ndim == 1:
        callback_matrix = callback_matrix.reshape(1, -1)

    feature_names = [str(name) for name in summary_traj.feature_names]
    n_rows = min(shap_matrix.shape[0], callback_matrix.shape[0])
    n_features = min(shap_matrix.shape[1], callback_matrix.shape[1], len(feature_names))
    if n_rows == 0 or n_features == 0:
        st.warning("SHAP matrix could not be aligned with training data.")
        return {}

    shap_matrix = shap_matrix[:n_rows, :n_features]
    X_sample = pd.DataFrame(callback_matrix[:n_rows, :n_features], columns=feature_names[:n_features])

    # Use readable feature names in the summary chart.
    rename_df = pd.DataFrame({"feature_index": feature_names[:n_features]})
    rename_df = substitute_feature_names(rename_df, feature_index_col="feature_index", feature_names=feature_names)
    X_sample.columns = rename_df["feature_index"].astype(str).tolist()

    fig = plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_matrix, X_sample, show=False, max_display=15, plot_size=(15, 8))
    plt.title(f"Final SHAP Summary (Iteration {final_iter})", fontsize=18, fontweight="bold", pad=16)
    plt.tight_layout()
    st.pyplot(fig, width="content")

    abs_means = np.mean(np.abs(shap_matrix), axis=0)
    abs_stds = np.std(np.abs(shap_matrix), axis=0)
    p90 = np.percentile(np.abs(shap_matrix), 90, axis=0)
    p10 = np.percentile(np.abs(shap_matrix), 10, axis=0)

    top_df = pd.DataFrame(
        {
            "feature_idx": list(range(n_features)),
            "feature_name": [str(name) for name in X_sample.columns],
            "mean_abs_shap": abs_means,
            "std_abs_shap": abs_stds,
            "p90_abs_shap": p90,
            "p10_abs_shap": p10,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Final iteration", final_iter, help="Last boosting iteration available in SHAP history.")
    c2.metric("Mean |SHAP| (all)", f"{float(np.mean(np.abs(shap_matrix))):.4f}", help=FIELD_DESCRIPTIONS["mean_abs_shap"])
    c3.metric("Features", int(n_features), help="Number of aligned features included in this SHAP snapshot.")

    with st.expander("Top features in final SHAP snapshot", expanded=False):
        st.dataframe(
            top_df.head(20),
            width="content",
            column_config={
                "mean_abs_shap": st.column_config.NumberColumn(help=FIELD_DESCRIPTIONS["mean_abs_shap"]),
                "std_abs_shap": st.column_config.NumberColumn(help=FIELD_DESCRIPTIONS["std_abs_shap"]),
                "p90_abs_shap": st.column_config.NumberColumn(help=FIELD_DESCRIPTIONS["p90_abs_shap"]),
                "p10_abs_shap": st.column_config.NumberColumn(help=FIELD_DESCRIPTIONS["p10_abs_shap"]),
            },
        )

    return {
        "final_iteration": final_iter,
        "shap_matrix": shap_matrix,
        "x_matrix": callback_matrix[:n_rows, :n_features],
        "aligned_feature_names": feature_names[:n_features],
        "top_shap_df": top_df,
        "feature_names": feature_names,
    }


def _get_booster(summary_traj: Any):
    """Get booster from session model object."""
    model = getattr(summary_traj, "model", None)
    if model is None:
        return None
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if estimator is None or not hasattr(estimator, "get_booster"):
        return None
    return estimator.get_booster()


def _render_training_dynamics_panel(
    feature_idx: int,
    readable_name: str,
    trajectory_metrics: Dict[int, Dict[str, Any]],
    confusion_metric: str,
) -> None:
    """Render per-feature dynamics over training iterations."""
    st.subheader("2) Training Dynamics")
    metrics = trajectory_metrics.get(feature_idx)
    if not metrics:
        st.info("No trajectory metrics for selected feature.")
        return

    iterations = np.asarray(metrics.get("iterations", []), dtype=float)
    accelerations = np.asarray(metrics.get("acceleration", []), dtype=float)
    fcis = _confusion_series(metrics, confusion_metric)
    cumulative = np.asarray(metrics.get("cumulative_gain", []), dtype=float)

    if iterations.size == 0:
        st.info("No iteration series available for this feature.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)

    axes[0].plot(iterations, accelerations, marker="o", color="#1f77b4", lw=2)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Acceleration")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, fcis, marker="s", color="#2ca02c", lw=2)
    axes[1].set_title(_confusion_label(confusion_metric))
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, cumulative, marker="^", color="#9467bd", lw=2)
    axes[2].set_title("Cumulative Gain")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{readable_name}: training dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, width="content")


def _render_category_reasoning_panel(signature: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
    """Render quadrant category and threshold context."""
    st.subheader("3) Behavioral Category Reasoning")

    category = str(signature.get("quadrant_category", "Unknown"))
    reason = str(signature.get("quadrant_reason", "No reason available"))
    signature_label = str(signature.get("signature", "No signature available"))
    accel_phase = str(signature.get("accel_phase_dominance", "n/a"))
    fci_phase = str(signature.get("fci_phase_dominance", "n/a"))
    activation_score = float(signature.get("activation_score", 0.0))
    confusion_score = float(signature.get("confusion_composite_score", 0.0))
    instability_penalty = float(signature.get("instability_penalty", 0.0))
    activation_split = float(thresholds.get("activation_split_threshold", thresholds.get("accel_split_threshold", 0.0)))
    confusion_split = float(thresholds.get("confusion_split_threshold", thresholds.get("fci_split_threshold", 0.0)))

    st.markdown(f"**Category:** {category}")
    st.caption(FIELD_DESCRIPTIONS["quadrant_category"])
    st.markdown(f"**Signature:** {signature_label}")
    st.caption(FIELD_DESCRIPTIONS["signature"])
    st.caption(reason)

    q1, q2, q3 = st.columns(3)
    q1.metric("Activation score", f"{activation_score:.4f}", help=FIELD_DESCRIPTIONS["activation_score"])
    q2.metric("Confusion composite", f"{confusion_score:.4f}", help=FIELD_DESCRIPTIONS["confusion_composite_score"])
    q3.metric("Instability penalty", f"{instability_penalty:.4f}", help=FIELD_DESCRIPTIONS["instability_penalty"])

    s1, s2 = st.columns(2)
    s1.metric("Activation split", f"{activation_split:.4f}")
    s2.metric("Confusion split", f"{confusion_split:.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Early accel mean", f"{float(signature.get('early_acceleration_mean', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["early_acceleration_mean"])
    c2.metric("Late accel mean", f"{float(signature.get('late_acceleration_mean', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["late_acceleration_mean"])
    c3.metric("Early high-confusion rate", f"{float(signature.get('early_high_fci_rate', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["early_high_fci_rate"])
    c4.metric("Late high-confusion rate", f"{float(signature.get('late_high_fci_rate', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["late_high_fci_rate"])

    st.write(
        f"Phase dominance: acceleration is **{accel_phase}**, confusion metric is **{fci_phase}**."
    )

    st.markdown("**Signature map (current definitions)**")
    signature_map_df = pd.DataFrame(
        [
            {
                "signature": "CORE DRIVER",
                "condition": "early_accel > 0 and late_accel < 0",
                "interpretation": "Early optimization, then stabilization",
            },
            {
                "signature": "LATE LEARNER",
                "condition": "early_accel < 0 and late_accel > 0",
                "interpretation": "Becomes important in later iterations",
            },
            {
                "signature": "UNSTABLE",
                "condition": "volatility > 0.5 * abs(acceleration_score)",
                "interpretation": "Oscillating importance (possible collinearity)",
            },
            {
                "signature": "OUTLIER SPECIALIST",
                "condition": "early_accel > 0 and late_accel > 0",
                "interpretation": "Continuous late-stage optimization",
            },
            {
                "signature": "STABLE BASELINE",
                "condition": "fallback case",
                "interpretation": "Consistent contribution throughout",
            },
        ]
    )
    st.dataframe(signature_map_df, width="content", hide_index=True)

    with st.expander("Threshold context", expanded=False):
        if thresholds:
            st.json(thresholds)
        else:
            st.info("No threshold payload found.")


def _render_tree_context_panel(
    feature_idx: int,
    readable_name: str,
    path_context: Dict[str, Any],
    feature_names: list[str],
) -> None:
    """Render path-level scope and late-decider context for selected feature."""
    st.subheader("4) Decision-Path Context")

    profiles_df = pd.DataFrame(path_context.get("feature_profiles", []))
    if profiles_df.empty:
        st.info("No decision-path profiles available.")
        return

    profiles_df["feature_idx"] = pd.to_numeric(profiles_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
    row_df = profiles_df[profiles_df["feature_idx"] == int(feature_idx)]
    if row_df.empty:
        st.info("Selected feature was not found in tree-path profiles.")
        return

    row = row_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Path count", int(row.get("path_count", 0)), help=FIELD_DESCRIPTIONS["path_count"])
    c2.metric("Late decider rate", f"{float(row.get('late_decider_rate', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["late_decider_rate"])
    c3.metric("Decision impact score", f"{float(row.get('decision_impact_score', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["decision_impact_score"])
    c4.metric("Scope entropy", f"{float(row.get('scope_entropy', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["scope_entropy"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Late-decider in late training", f"{float(row.get('late_in_late_training_rate', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["late_in_late_training_rate"])
    d2.metric("Mean |leaf| when late decider", f"{float(row.get('mean_abs_leaf_when_late_decider', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["mean_abs_leaf_when_late_decider"])
    d3.metric("Impact magnitude ratio", f"{float(row.get('impact_magnitude_ratio', 0.0)):.3f}", help=FIELD_DESCRIPTIONS["impact_magnitude_ratio"])
    d4.metric("Mean leaf when late decider", f"{float(row.get('mean_leaf_value_when_late_decider', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["mean_leaf_value_when_late_decider"])

    st.caption(
        "Decision impact score combines how often the feature is a late decider with how strong "
        "the resulting leaf outputs are relative to the global leaf-magnitude baseline."
    )

    scope_df = pd.DataFrame(row.get("top_scope_features", []))
    if not scope_df.empty:
        scope_df["feature_idx"] = pd.to_numeric(scope_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
        scope_df["scope_feature"] = scope_df["feature_idx"].apply(
            lambda idx: _readable_name_for_idx(int(idx), feature_names)
        )
        st.markdown("**Scope-defining features before this feature decides**")
        st.dataframe(scope_df[["scope_feature", "cooccurrence_count", "cooccurrence_rate"]], width="content")

    partner_df = pd.DataFrame(row.get("top_near_leaf_partners", []))
    if not partner_df.empty:
        partner_df["feature_idx"] = pd.to_numeric(partner_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
        partner_df["near_leaf_partner"] = partner_df["feature_idx"].apply(
            lambda idx: _readable_name_for_idx(int(idx), feature_names)
        )
        with st.expander("Near-leaf co-decider partners", expanded=False):
            st.dataframe(partner_df[["near_leaf_partner", "cooccurrence_count", "cooccurrence_rate"]], width="content")

    candidates_df = pd.DataFrame(path_context.get("outlier_specialist_candidates", []))
    is_candidate = False
    if not candidates_df.empty:
        candidates_df["feature_idx"] = pd.to_numeric(candidates_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
        is_candidate = bool((candidates_df["feature_idx"] == int(feature_idx)).any())

    if is_candidate:
        st.success(f"{readable_name} is flagged as an outlier-specialist candidate in path-context analysis.")

    # Action-oriented explanation for users coming from SHAP.
    scope_names = []
    if not scope_df.empty:
        scope_names = scope_df["scope_feature"].head(3).astype(str).tolist()
    if scope_names:
        st.info(
            "Why this feature is important: it tends to decide inside a scope first set by "
            + ", ".join(scope_names)
            + "."
        )
    else:
        st.info(
            "Why this feature is important: its late-decider behavior appears without a single dominant scope pattern."
        )


def _build_binned_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_bins: int = 12,
    min_points_per_bin: int = 12,
) -> Optional[Dict[str, List[float]]]:
    """Build a plot-ready piecewise curve by quantile binning x and averaging y."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n < 30:
        return None

    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 30:
        return None

    quantiles = np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if edges.size < 3:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if np.isclose(x_min, x_max):
            return None
        edges = np.linspace(x_min, x_max, 3)

    bin_ids = np.digitize(x, bins=edges[1:-1], right=False)
    centers: List[float] = []
    y_means: List[float] = []
    y_stds: List[float] = []
    counts: List[int] = []

    for bin_idx in range(max(edges.size - 1, 1)):
        members = bin_ids == bin_idx
        count = int(np.sum(members))
        if count < max(int(min_points_per_bin), 3):
            continue
        x_bin = x[members]
        y_bin = y[members]
        centers.append(float(np.mean(x_bin)))
        y_means.append(float(np.mean(y_bin)))
        y_stds.append(float(np.std(y_bin)))
        counts.append(count)

    if len(centers) < 2:
        return None

    order = np.argsort(np.asarray(centers))
    centers_arr = np.asarray(centers)[order]
    means_arr = np.asarray(y_means)[order]
    stds_arr = np.asarray(y_stds)[order]
    counts_arr = np.asarray(counts)[order]

    return {
        "x": centers_arr.astype(float).tolist(),
        "y": means_arr.astype(float).tolist(),
        "y_std": stds_arr.astype(float).tolist(),
        "count": counts_arr.astype(int).tolist(),
    }


def _serialize_curve_function(
    *,
    feature_idx: int,
    feature_name: str,
    context_label: str,
    curve: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Serialize a 1D feature-response function into a portable plotting payload."""
    return {
        "schema": "plot_function_v1",
        "function_type": "feature_response",
        "feature_idx": int(feature_idx),
        "feature_name": str(feature_name),
        "context": str(context_label),
        "representation": {
            "kind": "piecewise_linear",
            "x_knots": list(curve.get("x", [])),
            "y_knots": list(curve.get("y", [])),
            "y_std": list(curve.get("y_std", [])),
            "counts": list(curve.get("count", [])),
        },
    }


def _collect_feature_cohort_contexts(
    summary_traj: Any,
    *,
    aligned_feature_names: List[str],
    feature_idx: int,
    sample_count: int,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Extract top cohort contexts where the selected feature is a decider."""
    if feature_idx < 0 or feature_idx >= len(aligned_feature_names):
        return []
    if sample_count <= 0:
        return []

    X_train = summary_traj.classification.train_test.get("X_train")
    if X_train is None or len(X_train) == 0:
        return []
    X_df = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    if X_df.empty:
        return []

    n_rows = min(int(sample_count), int(X_df.shape[0]))
    n_cols = min(int(X_df.shape[1]), len(aligned_feature_names))
    if n_rows <= 0 or n_cols <= 0:
        return []
    X_df = X_df.iloc[:n_rows, :n_cols].copy()
    X_df.columns = aligned_feature_names[:n_cols]

    tree_entries = list(getattr(summary_traj.explain, "trees", []))
    if not tree_entries:
        return []

    analyzer = CohortAttributionAnalyzer(aligned_feature_names[:n_cols])
    rules = analyzer.extract_path_rules(tree_entries)
    assigned = analyzer.assign_samples_to_rules(X_df, rules)
    if not assigned:
        return []

    min_support = max(8, int(0.01 * n_rows))
    cohorts = analyzer.build_cohorts(assigned, min_support=min_support, min_paths=2)
    if not cohorts:
        return []

    candidates = [
        c for c in cohorts
        if int(c.get("decider_feature_idx", -1)) == int(feature_idx)
    ]
    candidates.sort(
        key=lambda c: (-float(c.get("decision_impact_local", 0.0)), -int(c.get("support_count", 0)))
    )

    contexts: List[Dict[str, Any]] = []
    for rank, cohort in enumerate(candidates[: max(int(top_k), 1)], start=1):
        support_ids = sorted(int(i) for i in cohort.get("support_ids", set()) if int(i) < n_rows)
        if len(support_ids) < max(min_support, 20):
            continue
        scope_conditions = list(cohort.get("scope_conditions", []))
        short_scope = " AND ".join(
            f"{c.get('feature_name', c.get('feature_idx'))} {c.get('op', '')} {c.get('threshold', '')}"
            for c in scope_conditions[:3]
        )
        if len(scope_conditions) > 3:
            short_scope += " ..."
        label = f"Context {rank}: {short_scope}" if short_scope else f"Context {rank}: repeated scope cohort"
        contexts.append(
            {
                "label": label,
                "support_ids": support_ids,
                "support_count": int(cohort.get("support_count", len(support_ids))),
                "decision_impact_local": float(cohort.get("decision_impact_local", 0.0)),
                "scope_conditions": scope_conditions,
            }
        )
    return contexts


def _build_partner_quantile_contexts(
    *,
    x_feature: np.ndarray,
    y_contrib: np.ndarray,
    partner_values: np.ndarray,
    partner_name: str,
) -> List[Dict[str, Any]]:
    """Build fallback contexts from partner-feature quantile slices."""
    x = np.asarray(x_feature, dtype=float).reshape(-1)
    y = np.asarray(y_contrib, dtype=float).reshape(-1)
    p = np.asarray(partner_values, dtype=float).reshape(-1)
    n = min(x.size, y.size, p.size)
    if n < 30:
        return []

    x = x[:n]
    y = y[:n]
    p = p[:n]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(p)
    x = x[mask]
    y = y[mask]
    p = p[mask]
    if x.size < 30:
        return []

    q1, q2 = np.quantile(p, [1.0 / 3.0, 2.0 / 3.0])
    slices = [
        (p <= q1, f"Partner low ({partner_name})"),
        ((p > q1) & (p <= q2), f"Partner mid ({partner_name})"),
        (p > q2, f"Partner high ({partner_name})"),
    ]
    contexts: List[Dict[str, Any]] = []
    for mask_slice, label in slices:
        ids = np.flatnonzero(mask_slice).astype(int).tolist()
        if len(ids) < 20:
            continue
        contexts.append(
            {
                "label": label,
                "support_ids": ids,
                "support_count": len(ids),
                "decision_impact_local": 0.0,
                "scope_conditions": [],
            }
        )
    return contexts


def _build_interaction_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    *,
    bins: int = 8,
    min_points_per_cell: int = 6,
) -> Optional[Dict[str, Any]]:
    """Build a plot-ready 2D interaction surface z=f(x,y) using quantile bins."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    n = min(x.size, y.size, z.size)
    if n < 40:
        return None

    x = x[:n]
    y = y[:n]
    z = z[:n]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if x.size < 40:
        return None

    xb = max(int(bins), 3)
    yb = max(int(bins), 3)
    x_edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, xb + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, yb + 1)))
    if x_edges.size < 3 or y_edges.size < 3:
        return None

    x_bin = np.digitize(x, bins=x_edges[1:-1], right=False)
    y_bin = np.digitize(y, bins=y_edges[1:-1], right=False)
    x_cells = x_edges.size - 1
    y_cells = y_edges.size - 1

    grid_mean = np.full((y_cells, x_cells), np.nan, dtype=float)
    grid_count = np.zeros((y_cells, x_cells), dtype=int)
    points: List[Dict[str, Any]] = []

    for yi in range(y_cells):
        for xi in range(x_cells):
            idx = (x_bin == xi) & (y_bin == yi)
            cnt = int(np.sum(idx))
            if cnt < max(int(min_points_per_cell), 3):
                continue
            z_mean = float(np.mean(z[idx]))
            x_center = float(np.mean(x[idx]))
            y_center = float(np.mean(y[idx]))
            grid_mean[yi, xi] = z_mean
            grid_count[yi, xi] = cnt
            points.append(
                {
                    "x_center": x_center,
                    "y_center": y_center,
                    "z_mean": z_mean,
                    "count": cnt,
                }
            )

    if not points:
        return None

    return {
        "schema": "plot_function_v1",
        "function_type": "interaction_surface",
        "representation": {
            "kind": "binned_surface",
            "x_edges": x_edges.astype(float).tolist(),
            "y_edges": y_edges.astype(float).tolist(),
            "z_grid_mean": np.nan_to_num(grid_mean, nan=np.nan).tolist(),
            "count_grid": grid_count.astype(int).tolist(),
            "points": points,
        },
    }


def _resolve_interaction_partner_idx(
    *,
    path_context: Dict[str, Any],
    feature_idx: int,
    n_features: int,
    shap_matrix: np.ndarray,
) -> Optional[int]:
    """Pick a partner feature for interaction plotting (context partner first)."""
    profiles_df = pd.DataFrame(path_context.get("feature_profiles", []))
    if not profiles_df.empty and "feature_idx" in profiles_df.columns:
        profiles_df["feature_idx"] = pd.to_numeric(profiles_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
        row_df = profiles_df[profiles_df["feature_idx"] == int(feature_idx)]
        if not row_df.empty:
            partner_df = pd.DataFrame(row_df.iloc[0].get("top_near_leaf_partners", []))
            if not partner_df.empty and "feature_idx" in partner_df.columns:
                partner_df["feature_idx"] = pd.to_numeric(partner_df["feature_idx"], errors="coerce").fillna(-1).astype(int)
                for idx in partner_df["feature_idx"].tolist():
                    if 0 <= int(idx) < n_features and int(idx) != int(feature_idx):
                        return int(idx)

    abs_means = np.mean(np.abs(shap_matrix), axis=0)
    order = np.argsort(abs_means)[::-1]
    for idx in order:
        if int(idx) != int(feature_idx) and 0 <= int(idx) < n_features:
            return int(idx)
    return None


def _render_plot_ready_functions_panel(
    *,
    summary_traj: Any,
    feature_idx: int,
    readable_name: str,
    shap_matrix: np.ndarray,
    x_matrix: np.ndarray,
    aligned_feature_names: List[str],
    path_context: Dict[str, Any],
) -> None:
    """Render global/context/cohort plot-ready functions and their serialized payloads."""
    st.subheader("5) Plot-Ready Feature Functions")

    if (
        shap_matrix.ndim != 2
        or x_matrix.ndim != 2
        or feature_idx < 0
        or feature_idx >= shap_matrix.shape[1]
        or feature_idx >= x_matrix.shape[1]
    ):
        st.info("Could not build plot-ready functions for the selected feature.")
        return

    n_rows = min(shap_matrix.shape[0], x_matrix.shape[0])
    n_features = min(shap_matrix.shape[1], x_matrix.shape[1], len(aligned_feature_names))
    if n_rows <= 0 or n_features <= 0:
        st.info("No aligned rows available for function serialization.")
        return

    shap_use = shap_matrix[:n_rows, :n_features]
    x_use = x_matrix[:n_rows, :n_features]

    x_feature = x_use[:, feature_idx]
    y_contrib = shap_use[:, feature_idx]

    global_curve = _build_binned_curve(x_feature, y_contrib, n_bins=12, min_points_per_bin=12)
    if global_curve is None:
        st.info("Not enough variation to estimate a global response function.")
        return

    serialized_functions: List[Dict[str, Any]] = [
        _serialize_curve_function(
            feature_idx=feature_idx,
            feature_name=readable_name,
            context_label="global",
            curve=global_curve,
        )
    ]

    contexts = _collect_feature_cohort_contexts(
        summary_traj,
        aligned_feature_names=aligned_feature_names[:n_features],
        feature_idx=feature_idx,
        sample_count=n_rows,
        top_k=3,
    )

    st.markdown("**Global and context-conditioned response curves**")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=130)
    ax.plot(global_curve["x"], global_curve["y"], color="#111111", lw=2.4, label="Global response")
    ax.fill_between(
        global_curve["x"],
        np.asarray(global_curve["y"]) - np.asarray(global_curve["y_std"]),
        np.asarray(global_curve["y"]) + np.asarray(global_curve["y_std"]),
        color="#777777",
        alpha=0.15,
    )

    context_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for idx, context in enumerate(contexts):
        ids = np.asarray(context.get("support_ids", []), dtype=int)
        if ids.size == 0:
            continue
        curve = _build_binned_curve(
            x_feature[ids],
            y_contrib[ids],
            n_bins=10,
            min_points_per_bin=max(8, int(0.02 * ids.size)),
        )
        if curve is None:
            continue
        ax.plot(
            curve["x"],
            curve["y"],
            lw=2.0,
            alpha=0.9,
            color=context_colors[idx % len(context_colors)],
            label=context["label"],
        )
        serialized_functions.append(
            _serialize_curve_function(
                feature_idx=feature_idx,
                feature_name=readable_name,
                context_label=context["label"],
                curve=curve,
            )
        )

    ax.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_title(f"{readable_name}: SHAP response function")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Expected SHAP contribution")
    ax.grid(True, alpha=0.25)

    partner_idx = _resolve_interaction_partner_idx(
        path_context=path_context,
        feature_idx=feature_idx,
        n_features=n_features,
        shap_matrix=shap_use,
    )
    if not contexts and partner_idx is not None:
        partner_name = _readable_name_for_idx(int(partner_idx), aligned_feature_names[:n_features])
        contexts = _build_partner_quantile_contexts(
            x_feature=x_feature,
            y_contrib=y_contrib,
            partner_values=x_use[:, int(partner_idx)],
            partner_name=partner_name,
        )
        for idx, context in enumerate(contexts):
            ids = np.asarray(context.get("support_ids", []), dtype=int)
            if ids.size == 0:
                continue
            curve = _build_binned_curve(
                x_feature[ids],
                y_contrib[ids],
                n_bins=10,
                min_points_per_bin=max(6, int(0.01 * ids.size)),
            )
            if curve is None:
                continue
            ax.plot(
                curve["x"],
                curve["y"],
                lw=1.8,
                alpha=0.85,
                color=context_colors[idx % len(context_colors)],
                linestyle="--",
                label=context["label"],
            )
            serialized_functions.append(
                _serialize_curve_function(
                    feature_idx=feature_idx,
                    feature_name=readable_name,
                    context_label=context["label"],
                    curve=curve,
                )
            )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, width="content")

    st.markdown("**Local contribution scatter**")
    sample_points_cap = 300
    order = np.argsort(np.abs(y_contrib))[::-1][:sample_points_cap]
    fig_local, ax_local = plt.subplots(1, 1, figsize=(10, 4), dpi=130)
    ax_local.scatter(
        x_feature[order],
        y_contrib[order],
        s=18,
        alpha=0.55,
        color="#1f77b4",
        edgecolors="none",
    )
    ax_local.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax_local.set_title(f"{readable_name}: local points (top |SHAP| samples)")
    ax_local.set_xlabel("Feature value")
    ax_local.set_ylabel("SHAP contribution")
    ax_local.grid(True, alpha=0.25)
    fig_local.tight_layout()
    st.pyplot(fig_local, width="content")

    interaction_payload: Optional[Dict[str, Any]] = None
    st.markdown("**Feature interaction surface**")
    if partner_idx is not None:
        interaction_payload = _build_interaction_surface(
            x_use[:, feature_idx],
            x_use[:, int(partner_idx)],
            y_contrib,
            bins=8,
            min_points_per_cell=3,
        )
        if interaction_payload is not None:
            z_grid = np.asarray(interaction_payload["representation"]["z_grid_mean"], dtype=float)
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), dpi=130)
            im = ax2.imshow(z_grid, cmap="coolwarm", aspect="auto", origin="lower")
            partner_name = _readable_name_for_idx(int(partner_idx), aligned_feature_names[:n_features])
            ax2.set_title(f"Interaction surface: {readable_name} × {partner_name}")
            ax2.set_xlabel(readable_name)
            ax2.set_ylabel(partner_name)
            fig2.colorbar(im, ax=ax2, label="Mean SHAP contribution")
            fig2.tight_layout()
            st.pyplot(fig2, width="content")

            interaction_payload["feature_idx"] = int(feature_idx)
            interaction_payload["feature_name"] = readable_name
            interaction_payload["partner_feature_idx"] = int(partner_idx)
            interaction_payload["partner_feature_name"] = partner_name
            serialized_functions.append(interaction_payload)
        else:
            partner_name = _readable_name_for_idx(int(partner_idx), aligned_feature_names[:n_features])
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), dpi=130)
            hb = ax2.hexbin(
                x_use[:, feature_idx],
                x_use[:, int(partner_idx)],
                C=y_contrib,
                reduce_C_function=np.mean,
                gridsize=24,
                cmap="coolwarm",
                mincnt=1,
            )
            ax2.set_title(f"Interaction map: {readable_name} × {partner_name}")
            ax2.set_xlabel(readable_name)
            ax2.set_ylabel(partner_name)
            fig2.colorbar(hb, ax=ax2, label="Mean SHAP contribution")
            fig2.tight_layout()
            st.pyplot(fig2, width="content")
    else:
        st.info("No interaction partner found for this feature.")

    local_payload = {
        "schema": "plot_function_v1",
        "function_type": "local_contribution_points",
        "feature_idx": int(feature_idx),
        "feature_name": str(readable_name),
        "representation": {
            "kind": "scatter",
            "x": x_feature[order].astype(float).tolist(),
            "phi": y_contrib[order].astype(float).tolist(),
        },
    }
    serialized_functions.append(local_payload)

    with st.expander("Serialized plot-ready functions", expanded=False):
        st.json(
            {
                "schema": "feature_plot_bundle_v1",
                "feature_idx": int(feature_idx),
                "feature_name": str(readable_name),
                "function_count": len(serialized_functions),
                "functions": serialized_functions,
            }
        )


def render_feature_inspector_page() -> None:
    """Render end-to-end SHAP-first feature inspector."""
    st.header("Feature Inspector")
    st.caption("Start from final SHAP importance, then inspect why a specific feature becomes more or less important.")
    _render_field_glossary()

    _ = get_llm_config()

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info("No trained model in session. Run Setup first.")
        return

    summary_traj = st.session_state.summary_traj
    confusion_metric = st.radio(
        "Confusion metric",
        options=["hessian", "fci"],
        index=0,
        horizontal=True,
        key="feature_inspector_confusion_metric",
    )
    st.caption(f"Using {_confusion_label(confusion_metric)} for behavioral/confusion-based views.")
    feature_names = [str(name) for name in getattr(summary_traj, "feature_names", [])]

    top_block = _render_top_shap_summary(summary_traj)
    if not top_block:
        return

    booster = _get_booster(summary_traj)
    if booster is None:
        st.warning("Could not access model booster for trajectory analysis.")
        return

    tree_analyzer = summary_traj.tree_analyzer
    trajectory_metrics = tree_analyzer.compute_trajectory_metrics(booster)
    behavior_summary = summary_traj.generate_shap_behavior_correlation_json(
        save=False,
        top_k_features=40,
        confusion_metric=confusion_metric,
    )
    quadrant_payload = behavior_summary.get("quadrant_categorization", {})
    thresholds = quadrant_payload.get("thresholds", {}) if isinstance(quadrant_payload, dict) else {}

    metrics_df = pd.DataFrame(behavior_summary.get("feature_metrics", []))
    if metrics_df.empty:
        st.warning("No feature metrics found in behavior summary.")
        return

    selector_df = _build_feature_selector_df(metrics_df, feature_names)
    signature_lookup = _build_signature_lookup(
        quadrant_payload if isinstance(quadrant_payload, dict) else {},
        trajectory_metrics,
    )
    selector_df["signature"] = selector_df["feature_idx"].map(signature_lookup).fillna("STABLE BASELINE: Consistent contribution throughout")

    st.divider()
    st.subheader("Select Feature For Deep Dive")
    signature_options = [
        "All",
        "CORE DRIVER: Early optimization, then stabilization",
        "LATE LEARNER: Becomes important in later iterations",
        "UNSTABLE: Oscillating importance (possible collinearity)",
        "OUTLIER SPECIALIST: Continuous late-stage optimization",
        "STABLE BASELINE: Consistent contribution throughout",
    ]
    selected_signature = st.selectbox(
        "Filter by signature",
        options=signature_options,
        index=0,
        key="feature_inspector_signature_filter",
        help="Limit feature candidates to a behavioral signature before selecting a feature.",
    )

    filtered_selector_df = selector_df
    if selected_signature != "All":
        filtered_selector_df = selector_df[selector_df["signature"] == selected_signature]
    if filtered_selector_df.empty:
        st.info("No features match the selected signature filter.")
        return

    selected_label = st.selectbox(
        "Choose a feature",
        options=filtered_selector_df.sort_values("conditional_importance_score", ascending=False)["selector_label"].tolist(),
        key="feature_inspector_selector",
        help="Select a feature to open its full drill-down across SHAP, behavioral dynamics, and tree-path context.",
    )

    selected_row_df = filtered_selector_df[filtered_selector_df["selector_label"] == selected_label]
    if selected_row_df.empty:
        st.warning("Selected feature not found.")
        return

    selected_row = selected_row_df.iloc[0]
    feature_idx = int(selected_row["feature_idx"])
    readable_name = str(selected_row["readable_feature_name"])

    st.markdown(f"### Deep Dive: {readable_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final mean |SHAP|", f"{float(selected_row.get('mean_abs_shap', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["mean_abs_shap"])
    c2.metric("Final SHAP variation (std)", f"{float(selected_row.get('std_abs_shap', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["std_abs_shap"])
    c3.metric("Conditional importance", f"{float(selected_row.get('conditional_importance_score', 0.0)):.4f}", help=FIELD_DESCRIPTIONS["conditional_importance_score"])

    _render_training_dynamics_panel(
        feature_idx,
        readable_name,
        trajectory_metrics,
        confusion_metric=confusion_metric,
    )

    signature = tree_analyzer.get_behavioral_signature(
        trajectory_metrics,
        feature_idx,
        confusion_metric=confusion_metric,
    )
    _render_category_reasoning_panel(signature, thresholds)

    path_context = tree_analyzer.analyze_feature_decision_contexts(
        tree_entries=list(getattr(summary_traj.explain, "trees", [])),
        trajectory_metrics=trajectory_metrics,
        top_k=10,
    )
    _render_tree_context_panel(feature_idx, readable_name, path_context, feature_names)
    _render_plot_ready_functions_panel(
        summary_traj=summary_traj,
        feature_idx=feature_idx,
        readable_name=readable_name,
        shap_matrix=np.asarray(top_block.get("shap_matrix")),
        x_matrix=np.asarray(top_block.get("x_matrix")),
        aligned_feature_names=[str(x) for x in top_block.get("aligned_feature_names", feature_names)],
        path_context=path_context,
    )

    with st.expander("Raw selected-feature records", expanded=False):
        st.json(
            {
                "feature_metrics_row": {k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in selected_row.items()},
                "behavioral_signature": signature,
            }
        )
