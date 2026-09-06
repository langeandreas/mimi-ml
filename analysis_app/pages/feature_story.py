"""Feature Story page for unified per-feature narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
import shap
import streamlit as st

from explainer.analysis.feature_story import FeatureStoryBuilder
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


def _build_final_snapshot(summary_traj: Any) -> Optional[Dict[str, Any]]:
    """Extract aligned final SHAP matrix and feature values."""
    shap_history = list(getattr(summary_traj.explain, "shap_values_history", []))
    if not shap_history:
        return None

    shap_history = sorted(shap_history, key=lambda item: int(item["iteration"]))
    final_entry = shap_history[-1]
    final_iter = int(final_entry["iteration"])

    shap_matrix = _normalize_shap_matrix(final_entry["shap_values"])
    callback_matrix = np.asarray(summary_traj.explain.X_train)
    if callback_matrix.ndim == 1:
        callback_matrix = callback_matrix.reshape(1, -1)

    feature_names = [str(name) for name in summary_traj.feature_names]
    n_rows = min(shap_matrix.shape[0], callback_matrix.shape[0])
    n_features = min(shap_matrix.shape[1], callback_matrix.shape[1], len(feature_names))
    if n_rows == 0 or n_features == 0:
        return None

    shap_matrix = shap_matrix[:n_rows, :n_features]
    x_matrix = callback_matrix[:n_rows, :n_features]
    aligned_feature_names = feature_names[:n_features]

    abs_means = np.mean(np.abs(shap_matrix), axis=0)
    abs_stds = np.std(np.abs(shap_matrix), axis=0)
    top_df = pd.DataFrame(
        {
            "feature_idx": list(range(n_features)),
            "feature_name": aligned_feature_names,
            "mean_abs_shap": abs_means,
            "std_abs_shap": abs_stds,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return {
        "final_iteration": final_iter,
        "shap_matrix": shap_matrix,
        "x_matrix": x_matrix,
        "feature_names": aligned_feature_names,
        "top_df": top_df,
    }


def _render_overall_shap_plot(snapshot: Dict[str, Any]) -> None:
    """Render overall SHAP summary plot with all aligned features."""
    shap_matrix = np.asarray(snapshot["shap_matrix"])
    x_matrix = np.asarray(snapshot["x_matrix"])
    feature_names = list(snapshot["feature_names"])
    final_iter = int(snapshot["final_iteration"])

    rename_df = pd.DataFrame({"feature_index": feature_names})
    rename_df = substitute_feature_names(
        rename_df,
        feature_index_col="feature_index",
        feature_names=feature_names,
    )
    readable_names = rename_df["feature_index"].astype(str).tolist()
    X_df = pd.DataFrame(x_matrix, columns=readable_names)

    fig = plt.figure(figsize=(15, 9))
    shap.summary_plot(
        shap_matrix,
        X_df,
        show=False,
        max_display=len(readable_names),
        plot_size=(15, 9),
    )
    plt.title(f"Overall SHAP Summary (Iteration {final_iter})", fontsize=17, fontweight="bold", pad=16)
    plt.tight_layout()
    st.pyplot(fig, width="content")

    c1, c2, c3 = st.columns(3)
    c1.metric("Final iteration", final_iter)
    c2.metric("Features shown", int(len(readable_names)))
    c3.metric("Mean |SHAP|", f"{float(np.mean(np.abs(shap_matrix))):.4f}")


def _compute_trajectory_importance_scores(
    *,
    top_df: pd.DataFrame,
    trajectory_metrics: Dict[int, Dict[str, Any]],
) -> Dict[int, float]:
    """Compute Conditional Importance Score (CIS) per feature from trajectory data.

    Formula (mirrors shap_behavior_correlator):
        CIS = mean_abs_shap × (1 + tanh(max(late_accel, 0))) × (1 + log1p(max(avg_fci, 0)))

    Distinct from the Scope Decider Score (SDS) which is path/context-evidence based.
    """
    out: Dict[int, float] = {}
    shap_map = {
        int(row["feature_idx"]): float(row["mean_abs_shap"])
        for _, row in top_df[["feature_idx", "mean_abs_shap"]].iterrows()
    }
    for feature_idx, metrics in trajectory_metrics.items():
        idx = int(feature_idx)
        mean_abs_shap = float(shap_map.get(idx, 0.0))
        fci = np.asarray(metrics.get("fci", []), dtype=float)
        accel = np.asarray(metrics.get("acceleration", []), dtype=float)
        avg_fci = float(np.mean(fci)) if fci.size else 0.0
        mid = len(accel) // 2
        late_accel = float(np.mean(accel[mid:])) if accel.size else 0.0
        accel_boost = 1.0 + float(np.tanh(max(late_accel, 0.0)))
        fci_boost = 1.0 + float(np.log1p(max(avg_fci, 0.0)))
        out[idx] = float(mean_abs_shap * accel_boost * fci_boost)
    # Normalize to [0, 1] relative to the maximum observed value.
    if out:
        max_val = max(out.values())
        if max_val > 0.0:
            out = {k: v / max_val for k, v in out.items()}
    return out


def _render_importance_comparison_plot(
    top_df: pd.DataFrame,
    scope_decider_scores: Dict[int, float],
    trajectory_importance_scores: Dict[int, float],
) -> None:
    """Render side-by-side SHAP-style bar charts comparing SDS and CIS.

    Left panel — Scope Decider Score (SDS):
        SDS = context_dependency × decision_likelihood × evidence_confidence
        Captures features that are *late deciders gated by scope/context evidence*.

    Right panel — Conditional Importance Score (CIS):
        CIS = mean_abs_shap × (1 + tanh(late_accel)) × (1 + log1p(avg_fci))
        Captures features whose *SHAP magnitude is amplified by training dynamics*.

    Both panels share the same feature order (sorted by SDS descending) so rank
    differences between the two metrics are immediately visible.
    """
    plot_df = top_df[["feature_idx", "readable_name", "mean_abs_shap"]].copy()
    plot_df["sds"] = plot_df["feature_idx"].apply(
        lambda i: float(scope_decider_scores.get(int(i), 0.0))
    )
    plot_df["cis"] = plot_df["feature_idx"].apply(
        lambda i: float(trajectory_importance_scores.get(int(i), 0.0))
    )
    # Sort ascending so the highest-CIS feature appears at the top after barh.
    plot_df = plot_df.sort_values("cis", ascending=True)

    n_features = len(plot_df)
    fig_height = max(4.5, 0.38 * n_features)
    fig, (ax_cis, ax_sds) = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)

    y_pos = np.arange(n_features)

    def _draw_bars(
        ax: Any,
        values: List[float],
        cmap_name: str,
        xlabel: str,
        title: str,
        label_suffix: str,
    ) -> None:
        cmap = plt.get_cmap(cmap_name)
        max_v = max(values) if values else 1.0
        bar_colors = [cmap(0.35 + 0.60 * (v / max(max_v, 1e-9))) for v in values]
        bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="none", height=0.7)
        for bar, val in zip(bars, values):
            if val > 0.005:
                ax.text(
                    val + max_v * 0.01,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{val:.3f}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#333333",
                )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0.0, max_v * 1.22 + 0.02)
        ax.axvline(x=0.0, color="#aaaaaa", linewidth=0.8)
        # Row separators make feature-to-bar alignment easier to scan.
        separators = np.arange(-0.5, len(values) + 0.5, 1.0)
        ax.hlines(
            separators,
            xmin=0.0,
            xmax=max_v * 1.22 + 0.02,
            colors="#d6d6d6",
            linewidth=0.6,
            alpha=0.8,
            zorder=0,
        )
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=max_v))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03)
        cbar.set_label(label_suffix, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    _draw_bars(
        ax_cis,
        plot_df["cis"].tolist(),
        "Blues",
        "Conditional Importance Score (CIS, normalized)",
        "Conditional Importance Score (CIS)\nmean|SHAP| × accel-boost × FCI-boost",
        "CIS magnitude",
    )
    _draw_bars(
        ax_sds,
        plot_df["sds"].tolist(),
        "Reds",
        "Scope Decider Score (SDS)",
        "Scope Decider Score (SDS)\ncontext-dep. × decision-likelihood × evidence",
        "SDS magnitude",
    )

    ax_cis.set_yticks(y_pos)
    ax_cis.set_yticklabels(plot_df["readable_name"].tolist(), fontsize=9)

    fig.suptitle(
        "Conditional Importance Score vs. Scope Decider Score\n"
        "(features ordered by CIS)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    st.pyplot(fig, width="content")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max SDS", f"{float(plot_df['sds'].max()):.4f}")
    c2.metric("Mean SDS", f"{float(plot_df['sds'].mean()):.4f}")
    c3.metric("Max CIS", f"{float(plot_df['cis'].max()):.4f}")
    c4.metric("Mean CIS", f"{float(plot_df['cis'].mean()):.4f}")


def _render_plot_functions(functions: List[Dict[str, Any]]) -> None:
    """Render plots from serialized `plot_function_v1` records."""
    response_rows = [
        f for f in functions
        if f.get("function_type") == "feature_response"
    ]
    if response_rows:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=130)
        for row in response_rows:
            rep = row.get("representation", {})
            x = np.asarray(rep.get("x_knots", []), dtype=float)
            y = np.asarray(rep.get("y_knots", []), dtype=float)
            if x.size < 2 or y.size < 2:
                continue
            label = str(row.get("context", "response"))
            ax.plot(x, y, lw=2, label=label)
        ax.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
        ax.set_title("Feature response functions")
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Expected SHAP contribution")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="content")

    local_row = next(
        (f for f in functions if f.get("function_type") == "local_contribution_points"),
        None,
    )
    if local_row is not None:
        rep = local_row.get("representation", {})
        x = np.asarray(rep.get("x", []), dtype=float)
        y = np.asarray(rep.get("phi", []), dtype=float)
        if x.size and y.size:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=130)
            ax.scatter(x, y, s=10, alpha=0.45, color="#1f77b4", edgecolors="none")
            ax.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
            ax.set_title("Local feature contributions")
            ax.set_xlabel("Feature value")
            ax.set_ylabel("SHAP contribution")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, width="content")

    surface_row = next(
        (f for f in functions if f.get("function_type") == "interaction_surface"),
        None,
    )
    if surface_row is not None:
        z_grid = np.asarray(
            surface_row.get("representation", {}).get("z_grid_mean", []),
            dtype=float,
        )
        if z_grid.size:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=130)
            im = ax.imshow(z_grid, cmap="coolwarm", aspect="auto", origin="lower")
            ax.set_title("Interaction surface")
            ax.set_xlabel("Feature value bins")
            ax.set_ylabel("Partner bins")
            fig.colorbar(im, ax=ax, label="Mean SHAP contribution")
            fig.tight_layout()
            st.pyplot(fig, width="content")


def _build_curve_from_slice(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_bins: int = 8,
    min_points: int = 6,
) -> Optional[Dict[str, np.ndarray]]:
    """Build simple binned response curve for a slice."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n < 8:
        return None
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 8:
        return None

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)))
    if edges.size < 3:
        return None
    bin_ids = np.digitize(x, bins=edges[1:-1], right=False)

    centers = []
    means = []
    for i in range(edges.size - 1):
        idx = bin_ids == i
        if int(np.sum(idx)) < max(int(min_points), 3):
            continue
        centers.append(float(np.mean(x[idx])))
        means.append(float(np.mean(y[idx])))

    if len(centers) < 2:
        return None
    order = np.argsort(np.asarray(centers))
    return {
        "x": np.asarray(centers, dtype=float)[order],
        "y": np.asarray(means, dtype=float)[order],
    }


def _get_scope_provider_candidates(
    *,
    decision_context: Dict[str, Any],
    feature_idx: int,
) -> List[int]:
    """Get top scope features that act before selected feature."""
    profiles_df = pd.DataFrame(decision_context.get("feature_profiles", []))
    if profiles_df.empty or "feature_idx" not in profiles_df.columns:
        return []
    profiles_df["feature_idx"] = pd.to_numeric(
        profiles_df["feature_idx"], errors="coerce"
    ).fillna(-1).astype(int)
    row_df = profiles_df[profiles_df["feature_idx"] == int(feature_idx)]
    if row_df.empty:
        return []
    row = row_df.iloc[0]
    scope_df = pd.DataFrame(row.get("top_scope_features", []))
    if scope_df.empty or "feature_idx" not in scope_df.columns:
        return []
    scope_df["feature_idx"] = pd.to_numeric(
        scope_df["feature_idx"], errors="coerce"
    ).fillna(-1).astype(int)
    return [int(v) for v in scope_df["feature_idx"].tolist()]


def _compute_context_dependency_scores(
    *,
    decision_context: Dict[str, Any],
    n_features: int,
) -> Dict[int, Dict[str, float]]:
    """Compute bounded context-dependency score per feature in [0, 1]."""
    out: Dict[int, Dict[str, float]] = {
        int(i): {"context_dependency_score": 0.0, "late_decider_rate": 0.0, "scope_specificity": 0.0}
        for i in range(max(int(n_features), 0))
    }
    profiles_df = pd.DataFrame(decision_context.get("feature_profiles", []))
    if profiles_df.empty or "feature_idx" not in profiles_df.columns:
        return out

    profiles_df["feature_idx"] = pd.to_numeric(
        profiles_df["feature_idx"], errors="coerce"
    ).fillna(-1).astype(int)

    for _, row in profiles_df.iterrows():
        idx = int(row.get("feature_idx", -1))
        if idx < 0 or idx >= n_features:
            continue

        late_decider_rate = float(row.get("late_decider_rate", 0.0))
        scope_entropy = float(row.get("scope_entropy", 0.0))
        scope_features = list(row.get("top_scope_features", []))
        k = len(scope_features)
        if k <= 0:
            scope_specificity = 0.0
            dominant_scope_rate = 0.0
        elif k == 1:
            scope_specificity = 1.0
            dominant_scope_rate = float(scope_features[0].get("cooccurrence_rate", 0.0))
        else:
            max_entropy = float(np.log2(k))
            norm_entropy = min(max(scope_entropy / max(max_entropy, 1e-9), 0.0), 1.0)
            scope_specificity = 1.0 - norm_entropy
            dominant_scope_rate = float(
                max(float(item.get("cooccurrence_rate", 0.0)) for item in scope_features)
            )

        dominant_scope_rate = min(max(dominant_scope_rate, 0.0), 1.0)
        score = late_decider_rate * (0.6 * scope_specificity + 0.4 * dominant_scope_rate)
        score = min(max(float(score), 0.0), 1.0)

        out[idx] = {
            "context_dependency_score": score,
            "late_decider_rate": min(max(late_decider_rate, 0.0), 1.0),
            "scope_specificity": min(max(scope_specificity, 0.0), 1.0),
        }
    return out


def _compute_scope_decider_scores(
    *,
    top_df: pd.DataFrame,
    decision_context: Dict[str, Any],
    context_scores: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    """Compute Scope Decider Score (SDS) per feature, normalized to [0, 1].

    SDS answers: *how much does this feature act as a late, context-specific
    decider backed by decision-path evidence?*

    SDS = context_dependency_score × decision_likelihood × abs_impact_gate × support_confidence

    Distinct from the Conditional Importance Score (CIS) which is trajectory-based:
    CIS = mean_abs_shap × (1 + tanh(late_accel)) × (1 + log1p(avg_fci)).
    """
    n_rows = int(len(top_df))
    if n_rows <= 0:
        return {}

    shap_rank = (
        top_df[["feature_idx", "mean_abs_shap"]]
        .assign(shap_rank_pct=lambda d: d["mean_abs_shap"].rank(method="average", pct=True))
    )
    shap_rank_map = {
        int(row["feature_idx"]): float(row["shap_rank_pct"])
        for _, row in shap_rank.iterrows()
    }
    shap_abs_map = {
        int(row["feature_idx"]): float(row["mean_abs_shap"])
        for _, row in top_df[["feature_idx", "mean_abs_shap"]].iterrows()
    }

    # Absolute-impact gate to suppress tiny-SHAP artifacts.
    shap_vals = np.asarray(top_df["mean_abs_shap"], dtype=float)
    p25 = float(np.percentile(shap_vals, 25)) if shap_vals.size else 0.0
    p90 = float(np.percentile(shap_vals, 90)) if shap_vals.size else 1.0
    # Keep a meaningful absolute floor; values below this should not be promoted.
    shap_floor = max(5e-4, p25)
    shap_span = max(p90 - shap_floor, 1e-9)

    impact_map: Dict[int, float] = {}
    path_count_map: Dict[int, float] = {}
    scope_partner_count_map: Dict[int, float] = {}
    scope_occurrence_total_map: Dict[int, float] = {}
    profiles_df = pd.DataFrame(decision_context.get("feature_profiles", []))
    if not profiles_df.empty and "feature_idx" in profiles_df.columns:
        profiles_df["feature_idx"] = pd.to_numeric(
            profiles_df["feature_idx"], errors="coerce"
        ).fillna(-1).astype(int)
        if "decision_impact_score" in profiles_df.columns:
            tmp = profiles_df[["feature_idx", "decision_impact_score"]].copy()
            tmp["impact_rank_pct"] = tmp["decision_impact_score"].rank(method="average", pct=True)
            impact_map = {
                int(row["feature_idx"]): float(row["impact_rank_pct"])
                for _, row in tmp.iterrows()
            }
        if "path_count" in profiles_df.columns:
            path_count_map = {
                int(row["feature_idx"]): float(row["path_count"])
                for _, row in profiles_df[["feature_idx", "path_count"]].iterrows()
            }
        if "top_scope_features" in profiles_df.columns:
            for _, row in profiles_df[["feature_idx", "top_scope_features"]].iterrows():
                idx = int(row["feature_idx"])
                scope_features = list(row.get("top_scope_features", []))
                scope_partner_count_map[idx] = float(len(scope_features))
                scope_occurrence_total_map[idx] = float(
                    sum(float(item.get("cooccurrence_count", 0.0)) for item in scope_features)
                )

    tree_count = float(decision_context.get("tree_count", 0) or 0)
    min_path_ref = max(20.0, 0.2 * tree_count)
    min_occ_ref = max(20.0, 0.2 * tree_count)

    cis_map: Dict[int, float] = {}
    for feature_idx in top_df["feature_idx"].astype(int).tolist():
        cds = float(context_scores.get(feature_idx, {}).get("context_dependency_score", 0.0))
        shap_norm = float(shap_rank_map.get(feature_idx, 0.0))
        impact_norm = float(impact_map.get(feature_idx, 0.0))
        decision_likelihood = 0.7 * shap_norm + 0.3 * impact_norm

        # Absolute SHAP gate: 0 below floor, smoothly rising above it.
        shap_abs = float(shap_abs_map.get(feature_idx, 0.0))
        abs_impact_gate = min(max((shap_abs - shap_floor) / shap_span, 0.0), 1.0)

        # Support confidence gate from path coverage and scope evidence.
        path_count = float(path_count_map.get(feature_idx, 0.0))
        scope_occ_total = float(scope_occurrence_total_map.get(feature_idx, 0.0))
        scope_partners = float(scope_partner_count_map.get(feature_idx, 0.0))

        path_support = min(max(path_count / max(min_path_ref, 1e-9), 0.0), 1.0)
        occurrence_support = min(max(scope_occ_total / max(min_occ_ref, 1e-9), 0.0), 1.0)
        partner_support = min(max(scope_partners / 5.0, 0.0), 1.0)
        support_confidence = 0.5 * path_support + 0.3 * occurrence_support + 0.2 * partner_support

        # Hard suppression for very sparse evidence.
        if path_count < 10 or scope_occ_total < 10:
            support_confidence *= 0.2

        cis = cds * decision_likelihood * abs_impact_gate * support_confidence
        cis_map[feature_idx] = min(max(float(cis), 0.0), 1.0)
    return cis_map


def _render_context_provider_section(
    *,
    selected_feature_idx: int,
    selected_feature_name: str,
    shap_matrix: np.ndarray,
    x_matrix: np.ndarray,
    feature_names: List[str],
    readable_names: List[str],
    decision_context: Dict[str, Any],
) -> None:
    """Render context-provider deep dive section."""
    st.divider()
    st.subheader("5) Context Provider Deep Dive")
    st.caption(
        "Choose a scope/context feature and inspect how that context shifts the selected feature's role."
    )

    n_rows = min(shap_matrix.shape[0], x_matrix.shape[0])
    n_features = min(shap_matrix.shape[1], x_matrix.shape[1], len(feature_names))
    if n_rows <= 0 or n_features <= 0:
        st.info("No aligned data available for context deep dive.")
        return

    shap_use = shap_matrix[:n_rows, :n_features]
    x_use = x_matrix[:n_rows, :n_features]
    y_target = shap_use[:, int(selected_feature_idx)]
    x_target = x_use[:, int(selected_feature_idx)]

    scope_candidates = [
        idx
        for idx in _get_scope_provider_candidates(
            decision_context=decision_context,
            feature_idx=selected_feature_idx,
        )
        if 0 <= int(idx) < n_features and int(idx) != int(selected_feature_idx)
    ]
    fallback_candidates = [
        int(i) for i in np.argsort(np.mean(np.abs(shap_use), axis=0))[::-1].tolist()
        if int(i) != int(selected_feature_idx)
    ]

    ordered_candidates = []
    seen = set()
    for idx in scope_candidates + fallback_candidates:
        if idx in seen:
            continue
        seen.add(idx)
        ordered_candidates.append(idx)

    if not ordered_candidates:
        st.info("No valid context provider feature available.")
        return

    provider_options = [
        f"{readable_names[idx] if idx < len(readable_names) else feature_names[idx]} ({idx})"
        for idx in ordered_candidates[:30]
    ]
    provider_label = st.selectbox(
        "Choose context provider feature",
        options=provider_options,
        key="feature_story_context_provider_selector",
    )
    provider_idx = int(provider_label.rsplit("(", 1)[1].rstrip(")"))
    provider_name = (
        readable_names[provider_idx]
        if provider_idx < len(readable_names)
        else feature_names[provider_idx]
    )

    provider_values = x_use[:, provider_idx]
    abs_bound = float(np.nanmax(np.abs(y_target))) if y_target.size else 1.0
    abs_bound = max(abs_bound, 1e-9)
    centered_norm = TwoSlopeNorm(vmin=-abs_bound, vcenter=0.0, vmax=abs_bound)

    # 2D view: SHAP(decider) as a function of decider value × scope-provider value.
    decider_edges = np.unique(np.quantile(x_target, np.linspace(0.0, 1.0, 10)))
    scope_edges = np.unique(np.quantile(provider_values, np.linspace(0.0, 1.0, 10)))
    if decider_edges.size >= 3 and scope_edges.size >= 3:
        decider_bin = np.digitize(x_target, bins=decider_edges[1:-1], right=False)
        scope_bin = np.digitize(provider_values, bins=scope_edges[1:-1], right=False)
        x_cells = decider_edges.size - 1
        y_cells = scope_edges.size - 1

        mean_grid = np.full((y_cells, x_cells), np.nan, dtype=float)
        count_grid = np.zeros((y_cells, x_cells), dtype=int)
        for yi in range(y_cells):
            for xi in range(x_cells):
                idx = (decider_bin == xi) & (scope_bin == yi)
                cnt = int(np.sum(idx))
                if cnt < 4:
                    continue
                mean_grid[yi, xi] = float(np.mean(y_target[idx]))
                count_grid[yi, xi] = cnt

        fig2d, ax2d = plt.subplots(1, 1, figsize=(10, 5), dpi=130)
        im = ax2d.imshow(
            mean_grid,
            cmap="coolwarm",
            aspect="auto",
            origin="lower",
            norm=centered_norm,
        )
        ax2d.set_title(
            f"SHAP({selected_feature_name}) across decider × scope context"
        )
        ax2d.set_xlabel(f"{selected_feature_name} (decider value bins)")
        ax2d.set_ylabel(f"{provider_name} (scope value bins)")
        cbar = fig2d.colorbar(im, ax=ax2d)
        cbar.set_label("Mean SHAP contribution")
        fig2d.tight_layout()
        st.pyplot(fig2d, width="content")

        with st.expander("2D bin sample counts", expanded=False):
            st.dataframe(pd.DataFrame(count_grid), width="content")
    else:
        fig2d, ax2d = plt.subplots(1, 1, figsize=(10, 5), dpi=130)
        hb = ax2d.hexbin(
            x_target,
            provider_values,
            C=y_target,
            reduce_C_function=np.mean,
            gridsize=28,
            cmap="coolwarm",
            norm=centered_norm,
            mincnt=1,
        )
        ax2d.set_title(
            f"SHAP({selected_feature_name}) across decider × scope context"
        )
        ax2d.set_xlabel(f"{selected_feature_name} (decider values)")
        ax2d.set_ylabel(f"{provider_name} (scope values)")
        cbar = fig2d.colorbar(hb, ax=ax2d)
        cbar.set_label("Mean SHAP contribution")
        fig2d.tight_layout()
        st.pyplot(fig2d, width="content")

    unique_provider_vals = np.unique(provider_values[np.isfinite(provider_values)])
    binary_provider = (
        unique_provider_vals.size <= 2
        and np.all(np.isin(np.round(unique_provider_vals, 8), [0.0, 1.0]))
    )
    masks: Dict[str, np.ndarray] = {}
    if binary_provider:
        masks["0"] = np.isclose(provider_values, 0.0)
        masks["1"] = np.isclose(provider_values, 1.0)
    else:
        # Rank-based terciles ensure low/mid/high slices are populated even with tied values.
        order = np.argsort(provider_values, kind="mergesort")
        terciles = np.array_split(order, 3)
        labels = ["Low", "Mid", "High"]
        for label, idxs in zip(labels, terciles):
            mask = np.zeros(provider_values.shape[0], dtype=bool)
            if idxs.size > 0:
                mask[idxs] = True
            masks[label] = mask

    rows = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=130)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, (label, mask) in zip(colors, masks.items()):
        idx = np.flatnonzero(mask).astype(int)
        if idx.size == 0:
            continue
        y_slice = y_target[idx]
        x_slice = x_target[idx]
        rows.append(
            {
                "context": label,
                "samples": int(idx.size),
                "mean_shap": float(np.mean(y_slice)),
                "mean_abs_shap": float(np.mean(np.abs(y_slice))),
                "std_abs_shap": float(np.std(np.abs(y_slice))),
            }
        )
        curve = _build_curve_from_slice(
            x_slice,
            y_slice,
            n_bins=8,
            min_points=max(4, int(0.01 * idx.size)),
        )
        if curve is not None:
            ax.plot(curve["x"], curve["y"], color=color, lw=2, label=label)
        elif idx.size >= 3:
            x_min = float(np.min(x_slice))
            x_max = float(np.max(x_slice))
            y_mean = float(np.mean(y_slice))
            if np.isclose(x_min, x_max):
                ax.scatter([x_min], [y_mean], color=color, s=45, label=label)
            else:
                ax.plot([x_min, x_max], [y_mean, y_mean], color=color, lw=2, linestyle="--", label=label)

    ax.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_title(f"1D slices of SHAP({selected_feature_name}) by {provider_name} levels")
    ax.set_xlabel(selected_feature_name)
    ax.set_ylabel("Expected SHAP contribution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, width="content")

    # Non-aggregated companion view: sample-level SHAP scatter by the same low/mid/high slices.
    fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(12, 4), dpi=130)
    unique_decider = np.unique(x_target[np.isfinite(x_target)])
    low_cardinality_decider = unique_decider.size > 0 and unique_decider.size <= 6
    mask_items = list(masks.items())
    n_slices = max(len(mask_items), 1)
    rng = np.random.default_rng(42)
    x_span_global = float(np.max(x_target) - np.min(x_target)) if x_target.size else 1.0
    x_span_global = max(x_span_global, 1.0)

    combo_positions: Dict[float, int] = {}
    combo_ticks: List[int] = []
    combo_tick_labels: List[str] = []
    if low_cardinality_decider:
        dec_levels = sorted(float(v) for v in unique_decider.tolist())
        pos = 0
        tick_step = 3
        for level in dec_levels:
            combo_positions[float(level)] = pos
            combo_ticks.append(pos)
            combo_tick_labels.append(f"(decider={level:g})")
            pos += tick_step

    for slice_idx, (color, (label, mask)) in enumerate(zip(colors, mask_items)):
        idx = np.flatnonzero(mask).astype(int)
        if idx.size == 0:
            continue
        x_slice = x_target[idx]
        y_slice = y_target[idx]
        if low_cardinality_decider and combo_positions:
            x_plot_vals = []
            for x_val in x_slice:
                nearest_level = min(
                    unique_decider.tolist(),
                    key=lambda v: abs(float(v) - float(x_val)),
                )
                x_plot_vals.append(float(combo_positions[float(nearest_level)]))
            x_plot = np.asarray(x_plot_vals, dtype=float) + rng.normal(
                loc=0.0, scale=0.12, size=len(x_plot_vals)
            )
        else:
            # Slice-dodge + jitter prevents one slice from hiding another.
            offset = (slice_idx - (n_slices - 1) / 2.0) * 0.06 * x_span_global
            jitter = rng.normal(loc=0.0, scale=0.012 * x_span_global, size=x_slice.shape[0])
            x_plot = x_slice + offset + jitter
        ax_scatter.scatter(
            x_plot,
            y_slice,
            s=14,
            alpha=0.33,
            color=color,
            edgecolors="none",
            label=f"{label} samples",
        )
    ax_scatter.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax_scatter.set_title(f"Sample-level SHAP({selected_feature_name}) by {provider_name} slices")
    if low_cardinality_decider and combo_ticks:
        ax_scatter.set_xticks(combo_ticks)
        ax_scatter.set_xticklabels(combo_tick_labels, rotation=0, ha="center", fontsize=8)
        ax_scatter.set_xlabel(f"{selected_feature_name} value")
        ax_scatter.text(
            0.0,
            -0.22,
            f"Tick key: decider = {selected_feature_name}; scope slices are color-coded",
            transform=ax_scatter.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    else:
        ax_scatter.set_xlabel(f"{selected_feature_name} (slice-dodged)")
    ax_scatter.set_ylabel("SHAP contribution")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(loc="best", fontsize=8)
    fig_scatter.tight_layout()
    fig_scatter.subplots_adjust(bottom=0.28)
    st.pyplot(fig_scatter, width="content")

    if rows:
        context_df = pd.DataFrame(rows)
        st.dataframe(
            context_df.style.format(
                {
                    "mean_shap": "{:.4f}",
                    "mean_abs_shap": "{:.4f}",
                    "std_abs_shap": "{:.4f}",
                }
            ),
            width="content",
        )
        best_row = context_df.sort_values("mean_abs_shap", ascending=False).iloc[0]
        st.info(
            f"Strongest context is **{best_row['context']}** (mean |SHAP|={best_row['mean_abs_shap']:.4f}), "
            f"showing how **{provider_name}** shifts the role of **{selected_feature_name}**."
        )
    else:
        st.info("Not enough samples in context slices to summarize role changes.")


def _render_top_context_slices_section(
    *,
    selected_feature_idx: int,
    selected_feature_name: str,
    shap_matrix: np.ndarray,
    x_matrix: np.ndarray,
    feature_names: List[str],
    readable_names: List[str],
    decision_context: Dict[str, Any],
) -> None:
    """Render 1D low/mid/high slice curves for top 3 context features."""
    st.divider()
    st.subheader("4) Top Context Feature Slices")
    st.caption(
        "Top 3 context features are selected dynamically. Binary context features are sliced as 0/1; "
        f"other context features use low/mid/high bands for SHAP({selected_feature_name}) against decider values."
    )

    n_rows = min(shap_matrix.shape[0], x_matrix.shape[0])
    n_features = min(shap_matrix.shape[1], x_matrix.shape[1], len(feature_names))
    if n_rows <= 0 or n_features <= 0:
        st.info("No aligned data available for context slices.")
        return

    shap_use = shap_matrix[:n_rows, :n_features]
    x_use = x_matrix[:n_rows, :n_features]
    y_target = shap_use[:, int(selected_feature_idx)]
    x_target = x_use[:, int(selected_feature_idx)]

    scope_candidates = [
        idx
        for idx in _get_scope_provider_candidates(
            decision_context=decision_context,
            feature_idx=selected_feature_idx,
        )
        if 0 <= int(idx) < n_features and int(idx) != int(selected_feature_idx)
    ]

    fallback_candidates = [
        int(i)
        for i in np.argsort(np.mean(np.abs(shap_use), axis=0))[::-1].tolist()
        if int(i) != int(selected_feature_idx)
    ]

    ordered = []
    seen = set()
    for idx in scope_candidates + fallback_candidates:
        if idx in seen:
            continue
        seen.add(int(idx))
        ordered.append(int(idx))
        if len(ordered) >= 3:
            break

    if not ordered:
        st.info("No context features available.")
        return

    slice_labels = ["Low", "Mid", "High"]
    rows: List[Dict[str, Any]] = []
    context_names: List[str] = []
    cmap_cycle = ["Reds", "Blues", "Greens", "Purples", "Oranges"]
    marker_cycle = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=130)
    max_points_per_context = 1800
    rng = np.random.default_rng(42)
    finite_decider = x_target[np.isfinite(x_target)]
    decider_levels = np.unique(finite_decider) if finite_decider.size else np.array([])
    low_cardinality_mode = decider_levels.size > 0 and decider_levels.size <= 4
    combo_positions: Dict[float, int] = {}
    combo_ticks: List[int] = []
    combo_labels: List[str] = []
    if low_cardinality_mode:
        sorted_levels = sorted(float(v) for v in decider_levels.tolist())
        pos = 0
        tick_step = 3
        for level in sorted_levels:
            combo_positions[float(level)] = pos
            combo_ticks.append(pos)
            combo_labels.append(f"(decider={level:g})")
            pos += tick_step

    for order_pos, context_idx in enumerate(ordered):
        context_name = (
            readable_names[context_idx]
            if context_idx < len(readable_names)
            else feature_names[context_idx]
        )
        context_names.append(context_name)
        context_values = x_use[:, context_idx]
        finite_mask = np.isfinite(context_values) & np.isfinite(x_target) & np.isfinite(y_target)
        if not np.any(finite_mask):
            continue

        x_plot = x_target[finite_mask]
        y_plot = y_target[finite_mask]
        context_plot = context_values[finite_mask]

        if x_plot.size > max_points_per_context:
            sample_idx = np.linspace(0, x_plot.size - 1, max_points_per_context).astype(int)
            x_plot = x_plot[sample_idx]
            y_plot = y_plot[sample_idx]
            context_plot = context_plot[sample_idx]

        cmin = float(np.min(context_plot))
        cmax = float(np.max(context_plot))
        if np.isclose(cmin, cmax):
            norm_scope = np.full(context_plot.shape, 0.5, dtype=float)
        else:
            norm_scope = (context_plot - cmin) / max(cmax - cmin, 1e-9)

        unique_scope_vals = np.unique(context_plot[np.isfinite(context_plot)])
        binary_scope = (
            unique_scope_vals.size <= 2
            and np.all(np.isin(np.round(unique_scope_vals, 8), [0.0, 1.0]))
        )
        if binary_scope:
            scope_labels_plot = np.where(np.isclose(context_plot, 0.0), "0", "1")
        else:
            q1_plot, q2_plot = np.quantile(context_plot, [1 / 3, 2 / 3])
            scope_labels_plot = np.where(
                context_plot <= q1_plot,
                "Low",
                np.where(context_plot <= q2_plot, "Mid", "High"),
            )

        x_for_scatter = x_plot.copy()
        if low_cardinality_mode and combo_positions:
            x_combo = []
            for x_val, scope_label in zip(x_plot, scope_labels_plot):
                closest_level = min(combo_positions.keys(), key=lambda k: abs(float(k) - float(x_val)))
                x_combo.append(float(combo_positions[float(closest_level)]))
            x_for_scatter = np.asarray(x_combo, dtype=float) + rng.normal(
                loc=0.0, scale=0.12, size=len(x_combo)
            )
        else:
            # Continuous mode: light jitter for readability.
            unique_x = np.unique(x_plot)
            if unique_x.size <= 4:
                x_span = float(np.max(unique_x) - np.min(unique_x)) if unique_x.size > 1 else 1.0
                jitter_scale = 0.04 * max(x_span, 1.0)
                x_for_scatter = x_plot + rng.normal(loc=0.0, scale=jitter_scale, size=x_plot.shape[0])

        cmap = plt.get_cmap(cmap_cycle[order_pos % len(cmap_cycle)])
        point_colors = cmap(0.25 + 0.7 * norm_scope)
        marker = marker_cycle[order_pos % len(marker_cycle)]
        ax.scatter(
            x_for_scatter,
            y_plot,
            c=point_colors,
            s=13,
            alpha=0.38,
            marker=marker,
            edgecolors="none",
            label=f"{context_name} ({cmap_cycle[order_pos % len(cmap_cycle)]}: low→high)",
        )

        unique_context_vals = np.unique(context_values[np.isfinite(context_values)])
        binary_context = (
            unique_context_vals.size <= 2
            and np.all(np.isin(np.round(unique_context_vals, 8), [0.0, 1.0]))
        )
        if binary_context:
            slice_defs = [
                ("0", np.isclose(context_values, 0.0)),
                ("1", np.isclose(context_values, 1.0)),
            ]
        else:
            rank_order = np.argsort(context_values, kind="mergesort")
            terciles = np.array_split(rank_order, 3)
            slice_defs = []
            for label, idxs in zip(slice_labels, terciles):
                mask = np.zeros(context_values.shape[0], dtype=bool)
                if idxs.size > 0:
                    mask[idxs] = True
                slice_defs.append((label, mask))

        for label, mask in slice_defs:
            ids = np.flatnonzero(mask).astype(int)
            if ids.size == 0:
                continue

            x_slice = x_target[ids]
            y_slice = y_target[ids]
            scope_vals = context_values[ids]
            rows.append(
                {
                    "context_feature": context_name,
                    "slice": label,
                    "samples": int(ids.size),
                    "scope_value_min": float(np.min(scope_vals)),
                    "scope_value_max": float(np.max(scope_vals)),
                    "mean_shap": float(np.mean(y_slice)),
                    "mean_abs_shap": float(np.mean(np.abs(y_slice))),
                }
            )

    ax.axhline(y=0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_title(f"{selected_feature_name}: SHAP scatter colored by scope-context value")
    if low_cardinality_mode and combo_ticks:
        ax.set_xticks(combo_ticks)
        ax.set_xticklabels(combo_labels, rotation=0, ha="center", fontsize=8)
        ax.set_xlabel(f"{selected_feature_name} value")
        ax.text(
            0.0,
            -0.18,
            f"Tick key: decider = {selected_feature_name}; scope values are color-intensity encoded",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    else:
        ax.set_xlabel(selected_feature_name)
    ax.set_ylabel(f"SHAP contribution of {selected_feature_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23)
    st.pyplot(fig, width="content")

    st.caption(
        "Top context features used: "
        + ", ".join(context_names)
        + ". Within each feature, lighter dots indicate lower scope values and darker dots indicate higher scope values."
    )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="content")


def _build_feature_overview_df(
    *,
    top_df: pd.DataFrame,
    decision_context: Dict[str, Any],
    context_scores: Dict[int, Dict[str, float]],
    trajectory_metrics: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """Build all-feature overview with context, impact, and trajectory stats."""
    out = top_df[["feature_idx", "readable_name", "mean_abs_shap", "std_abs_shap"]].copy()

    profiles_df = pd.DataFrame(decision_context.get("feature_profiles", []))
    if not profiles_df.empty and "feature_idx" in profiles_df.columns:
        profiles_df["feature_idx"] = pd.to_numeric(
            profiles_df["feature_idx"], errors="coerce"
        ).fillna(-1).astype(int)
        keep_cols = [
            "feature_idx",
            "scope_entropy",
            "decision_impact_score",
            "late_decider_rate",
            "path_count",
            "impact_magnitude_ratio",
            "mean_abs_leaf_when_late_decider",
        ]
        keep_cols = [c for c in keep_cols if c in profiles_df.columns]
        out = out.merge(
            profiles_df[keep_cols],
            on="feature_idx",
            how="left",
        )
    else:
        out["scope_entropy"] = np.nan
        out["decision_impact_score"] = np.nan
        out["late_decider_rate"] = np.nan
        out["path_count"] = np.nan
        out["impact_magnitude_ratio"] = np.nan
        out["mean_abs_leaf_when_late_decider"] = np.nan

    out["context_dependency_score"] = out["feature_idx"].apply(
        lambda i: float(context_scores.get(int(i), {}).get("context_dependency_score", 0.0))
    )
    out["scope_specificity"] = out["feature_idx"].apply(
        lambda i: float(context_scores.get(int(i), {}).get("scope_specificity", 0.0))
    )

    avg_fci_map: Dict[int, float] = {}
    late_accel_map: Dict[int, float] = {}
    total_gain_map: Dict[int, float] = {}
    for idx, metrics in trajectory_metrics.items():
        fci = np.asarray(metrics.get("fci", []), dtype=float)
        accel = np.asarray(metrics.get("acceleration", []), dtype=float)
        gains = np.asarray(metrics.get("gains", []), dtype=float)
        mid = len(accel) // 2
        avg_fci_map[int(idx)] = float(np.mean(fci)) if fci.size else 0.0
        late_accel_map[int(idx)] = float(np.mean(accel[mid:])) if accel.size else 0.0
        total_gain_map[int(idx)] = float(np.sum(gains)) if gains.size else 0.0

    out["avg_fci"] = out["feature_idx"].map(avg_fci_map).fillna(0.0)
    out["late_acceleration_mean"] = out["feature_idx"].map(late_accel_map).fillna(0.0)
    out["total_gain"] = out["feature_idx"].map(total_gain_map).fillna(0.0)

    # Useful compact composite for ranking: influence × context dependence × decision impact.
    out["context_impact_index"] = (
        out["mean_abs_shap"].fillna(0.0)
        * out["context_dependency_score"].fillna(0.0)
        * out["decision_impact_score"].fillna(0.0)
    )

    return out.sort_values("context_impact_index", ascending=False)


def render_feature_story_page() -> None:
    """Render feature story page with global SHAP and per-feature drill-down."""
    st.header("Feature Story")
    st.caption("See all-feature SHAP behavior first, then deep dive into one feature narrative.")

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info("No trained model in session. Run Setup first.")
        return

    summary_traj = st.session_state.summary_traj
    snapshot = _build_final_snapshot(summary_traj)
    if snapshot is None:
        st.warning("No aligned SHAP snapshot available.")
        return

    feature_names = list(snapshot["feature_names"])
    top_df = snapshot["top_df"].copy()
    label_df = pd.DataFrame({"feature_index": feature_names})
    label_df = substitute_feature_names(label_df, "feature_index", feature_names)
    readable_names = label_df["feature_index"].astype(str).tolist()
    top_df["readable_name"] = top_df["feature_idx"].apply(
        lambda idx: readable_names[int(idx)] if int(idx) < len(readable_names) else str(idx)
    )
    top_df["selector_label"] = top_df.apply(
        lambda row: f"{row['readable_name']} ({int(row['feature_idx'])})",
        axis=1,
    )

    booster = None
    model = getattr(summary_traj, "model", None)
    if model is not None:
        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if estimator is not None and hasattr(estimator, "get_booster"):
            booster = estimator.get_booster()

    trajectory_metrics = {}
    decision_context = {}
    cohort_attribution = {}
    if booster is not None:
        tree_analyzer = summary_traj.tree_analyzer
        trajectory_metrics = tree_analyzer.compute_trajectory_metrics(booster)
        decision_context = tree_analyzer.analyze_feature_decision_contexts(
            tree_entries=list(getattr(summary_traj.explain, "trees", [])),
            trajectory_metrics=trajectory_metrics,
            top_k=20,
        )
        cohort_attribution = summary_traj.generate_cohort_attribution_json(
            save=False,
            min_support=8,
            min_paths=2,
            top_k=25,
            top_local_shap=8,
        )

    context_scores = _compute_context_dependency_scores(
        decision_context=decision_context if decision_context else {},
        n_features=len(feature_names),
    )
    scope_decider_scores = _compute_scope_decider_scores(
        top_df=top_df,
        decision_context=decision_context if decision_context else {},
        context_scores=context_scores,
    )
    trajectory_importance_scores = _compute_trajectory_importance_scores(
        top_df=top_df,
        trajectory_metrics=trajectory_metrics if trajectory_metrics else {},
    )
    top_df["context_dependency_score"] = top_df["feature_idx"].apply(
        lambda i: float(context_scores.get(int(i), {}).get("context_dependency_score", 0.0))
    )
    top_df["scope_decider_score"] = top_df["feature_idx"].apply(
        lambda i: float(scope_decider_scores.get(int(i), 0.0))
    )
    top_df["conditional_importance_score"] = top_df["feature_idx"].apply(
        lambda i: float(trajectory_importance_scores.get(int(i), 0.0))
    )
    top_df["selector_label"] = top_df.apply(
        lambda row: (
            f"{row['readable_name']} ({int(row['feature_idx'])}) "
            f"· SDS {float(row['scope_decider_score']):.2f}"
        ),
        axis=1,
    )
    with st.expander("Top Scope Decider Score features", expanded=False):
        st.dataframe(
            top_df[["readable_name", "feature_idx", "context_dependency_score", "scope_decider_score", "conditional_importance_score"]]
            .sort_values("scope_decider_score", ascending=False)
            .head(10),
            width="content",
        )

    st.subheader("1) Overall SHAP View")
    st.caption("Beeswarm summary of SHAP values at the final training iteration.")
    _render_overall_shap_plot(snapshot)

    st.subheader("1b) Scope Decider Score (SDS) vs. Conditional Importance Score (CIS)")
    st.caption(
        "**CIS** (Conditional Importance Score, blue) = mean|SHAP| × accel-boost × FCI-boost — "
        "highlights features whose *SHAP magnitude is amplified by training dynamics*. "
        "**SDS** (Scope Decider Score, red) = context-dependency × decision-likelihood × evidence-confidence — "
        "highlights features that are *late deciders gated by scope/context*. "
        "Features are ordered by CIS; rank differences between the two panels reveal what each metric sees "
        "that the other misses."
    )
    _render_importance_comparison_plot(
        top_df=top_df,
        scope_decider_scores=scope_decider_scores,
        trajectory_importance_scores=trajectory_importance_scores,
    )

    st.divider()
    st.subheader("2) All-Feature Overview")
    overview_df = _build_feature_overview_df(
        top_df=top_df,
        decision_context=decision_context if decision_context else {},
        context_scores=context_scores,
        trajectory_metrics=trajectory_metrics if trajectory_metrics else {},
    )
    overview_df["scope_decider_score"] = overview_df["feature_idx"].apply(
        lambda i: float(scope_decider_scores.get(int(i), 0.0))
    )
    overview_df["conditional_importance_score"] = overview_df["feature_idx"].apply(
        lambda i: float(trajectory_importance_scores.get(int(i), 0.0))
    )
    overview_df = overview_df.sort_values("scope_decider_score", ascending=False)
    st.dataframe(
        overview_df,
        width="content",
        column_config={
            "scope_decider_score": st.column_config.NumberColumn(
                "scope_decider_score",
                help="SDS [0,1]: context_dependency × decision_likelihood × evidence_confidence. Highlights late deciders whose role is gated by scope/context.",
            ),
            "conditional_importance_score": st.column_config.NumberColumn(
                "conditional_importance_score",
                help="CIS (normalized) [0,1]: mean|SHAP| × (1+tanh(late_accel)) × (1+log1p(avg_fci)). Highlights features amplified by training dynamics.",
            ),
            "context_dependency_score": st.column_config.NumberColumn(
                "context_dependency_score",
                help="Bounded [0,1] measure of how strongly a feature's role depends on scope/context.",
            ),
            "scope_entropy": st.column_config.NumberColumn(
                "scope_entropy",
                help="Higher means broader/more diverse contexts; lower means more concentrated scope dependence.",
            ),
            "decision_impact_score": st.column_config.NumberColumn(
                "decision_impact_score",
                help="Path-based impact metric for late-decider influence strength.",
            ),
            "context_impact_index": st.column_config.NumberColumn(
                "context_impact_index",
                help="Composite ranking: mean_abs_shap × context_dependency_score × decision_impact_score.",
            ),
        },
    )

    st.divider()
    st.subheader("3) Feature Deep Dive")
    selected_label = st.selectbox(
        "Select feature",
        options=top_df.sort_values("scope_decider_score", ascending=False)["selector_label"].tolist(),
        key="feature_story_selector",
    )
    selected_row = top_df[top_df["selector_label"] == selected_label].iloc[0]
    feature_idx = int(selected_row["feature_idx"])
    readable_name = str(selected_row["readable_name"])

    st.caption(
        "**Scope Decider Score (SDS)** range: **0.00 → 1.00** — "
        "context_dependency × decision_likelihood × evidence_confidence. "
        "**Conditional Importance Score (CIS)** — mean|SHAP| × accel-boost × FCI-boost (normalized)."
    )
    selected_sds = float(scope_decider_scores.get(feature_idx, 0.0))
    selected_cis = float(trajectory_importance_scores.get(feature_idx, 0.0))
    cds_col1, cds_col2, cds_col3 = st.columns([1, 1, 2])
    cds_col1.metric("SDS", f"{selected_sds:.3f}")
    cds_col2.metric("CIS", f"{selected_cis:.3f}")
    cds_col3.progress(min(max(selected_sds, 0.0), 1.0), text="SDS")

    builder = FeatureStoryBuilder(feature_names)
    story = builder.build_feature_story(
        feature_idx=feature_idx,
        shap_values=np.asarray(snapshot["shap_matrix"]),
        feature_matrix=np.asarray(snapshot["x_matrix"]),
        trajectory_metrics=trajectory_metrics if trajectory_metrics else None,
        decision_context=decision_context if decision_context else None,
        cohort_attribution=cohort_attribution if cohort_attribution else None,
        top_k_contexts=3,
        top_k_cohorts=3,
    )

    st.markdown(f"### {readable_name}")
    narrative = story.get("narrative", {})
    st.info(str(narrative.get("headline", "")))

    gp = story.get("global_profile", {})
    orole = story.get("outcome_role", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean |SHAP|", f"{float(gp.get('mean_abs_shap', 0.0)):.4f}")
    c2.metric("Monotonicity", str(gp.get("monotonicity", "mixed")))
    c3.metric("Conditional importance", f"{float(orole.get('conditional_importance_score', 0.0)):.4f}")
    c4.metric("Stability (CV)", f"{float(orole.get('stability_cv', 0.0)):.4f}")

    _render_plot_functions(list(story.get("plot_functions", [])))

    dctx = pd.DataFrame(story.get("decision_contexts", []))
    if not dctx.empty:
        st.markdown("**Decision contexts**")
        st.dataframe(dctx, width="content")

    cohorts = pd.DataFrame(story.get("cohort_attributions", []))
    if not cohorts.empty:
        st.markdown("**Cohort attributions**")
        st.dataframe(cohorts, width="content")

    st.markdown("**Narrative details**")
    st.write(str(narrative.get("context", "")))
    st.write(str(narrative.get("cohort", "")))
    st.write(str(narrative.get("interaction", "")))

    with st.expander("Feature story JSON", expanded=False):
        st.json(story)

    _render_top_context_slices_section(
        selected_feature_idx=feature_idx,
        selected_feature_name=readable_name,
        shap_matrix=np.asarray(snapshot["shap_matrix"]),
        x_matrix=np.asarray(snapshot["x_matrix"]),
        feature_names=feature_names,
        readable_names=readable_names,
        decision_context=decision_context if decision_context else {},
    )

    _render_context_provider_section(
        selected_feature_idx=feature_idx,
        selected_feature_name=readable_name,
        shap_matrix=np.asarray(snapshot["shap_matrix"]),
        x_matrix=np.asarray(snapshot["x_matrix"]),
        feature_names=feature_names,
        readable_names=readable_names,
        decision_context=decision_context if decision_context else {},
    )
