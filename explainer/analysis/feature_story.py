"""Feature story object builder.

Creates a single, plot-ready and narrative-friendly object per feature by
combining:
1. Global profile from feature value -> SHAP contribution response
2. Decision contexts from scope->decider path analysis
3. Cohort attributions from cohort_attribution_v1 payloads
4. Interaction effects with a partner feature
5. Outcome-role metrics from SHAP + trajectory dynamics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .trajectory_utils import round_float


class FeatureStoryBuilder:
    """Builds `feature_story_v1` payloads for one or many features."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = [str(name) for name in feature_names]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_story(
        self,
        *,
        feature_idx: int,
        shap_values: Any,
        feature_matrix: Any,
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        decision_context: Optional[Dict[str, Any]] = None,
        cohort_attribution: Optional[Dict[str, Any]] = None,
        partner_feature_idx: Optional[int] = None,
        top_k_contexts: int = 3,
        top_k_cohorts: int = 3,
    ) -> Dict[str, Any]:
        """Build one feature story object."""
        shap_2d = _coerce_2d(shap_values)
        x_2d = _coerce_2d(feature_matrix)
        n_rows = min(shap_2d.shape[0], x_2d.shape[0])
        n_features = min(shap_2d.shape[1], x_2d.shape[1], len(self.feature_names))

        if n_rows <= 0 or n_features <= 0:
            raise ValueError("No aligned rows/features available for feature story generation.")
        if feature_idx < 0 or feature_idx >= n_features:
            raise ValueError(f"feature_idx out of range: {feature_idx}")

        shap_use = shap_2d[:n_rows, :n_features]
        x_use = x_2d[:n_rows, :n_features]

        feature_name = self.feature_names[feature_idx]
        x_feature = x_use[:, feature_idx]
        phi_feature = shap_use[:, feature_idx]

        global_curve = _build_binned_curve(x_feature, phi_feature, n_bins=12, min_points_per_bin=10)
        if global_curve is None:
            global_curve = {"x": [], "y": [], "y_std": [], "count": []}

        decision_contexts = self._extract_decision_contexts(
            decision_context=decision_context,
            feature_idx=feature_idx,
            top_k=top_k_contexts,
        )
        cohort_contexts = self._extract_cohort_contexts(
            cohort_attribution=cohort_attribution,
            feature_idx=feature_idx,
            top_k=top_k_cohorts,
        )

        context_functions = self._build_context_functions(
            feature_idx=feature_idx,
            feature_name=feature_name,
            x_feature=x_feature,
            phi_feature=phi_feature,
            decision_contexts=decision_contexts,
            cohort_contexts=cohort_contexts,
        )

        partner_idx = self._resolve_partner_idx(
            partner_feature_idx=partner_feature_idx,
            feature_idx=feature_idx,
            n_features=n_features,
            decision_context=decision_context,
            x_use=x_use,
            phi_feature=phi_feature,
        )

        interaction_payload = None
        if partner_idx is not None:
            interaction_payload = _build_interaction_surface(
                x_values=x_use[:, feature_idx],
                y_values=x_use[:, partner_idx],
                z_values=phi_feature,
                bins=8,
                min_points_per_cell=4,
            )

        outcome_role = self._build_outcome_role(
            feature_idx=feature_idx,
            phi_feature=phi_feature,
            trajectory_metrics=trajectory_metrics,
        )
        global_profile = self._build_global_profile(
            x_feature=x_feature,
            phi_feature=phi_feature,
            global_curve=global_curve,
        )

        plot_functions: List[Dict[str, Any]] = [
            _serialize_curve_function(
                feature_idx=feature_idx,
                feature_name=feature_name,
                context_label="global",
                curve=global_curve,
            )
        ]
        plot_functions.extend(context_functions)
        if interaction_payload is not None:
            interaction_payload["feature_idx"] = int(feature_idx)
            interaction_payload["feature_name"] = str(feature_name)
            interaction_payload["partner_feature_idx"] = int(partner_idx)
            interaction_payload["partner_feature_name"] = str(self.feature_names[partner_idx])
            plot_functions.append(interaction_payload)
        plot_functions.append(
            {
                "schema": "plot_function_v1",
                "function_type": "local_contribution_points",
                "feature_idx": int(feature_idx),
                "feature_name": str(feature_name),
                "representation": {
                    "kind": "scatter",
                    "x": x_feature.astype(float).tolist(),
                    "phi": phi_feature.astype(float).tolist(),
                },
            }
        )

        interaction_summary = {
            "partner_feature_idx": int(partner_idx) if partner_idx is not None else None,
            "partner_feature_name": (
                str(self.feature_names[partner_idx]) if partner_idx is not None else None
            ),
            "interaction_surface_available": bool(interaction_payload is not None),
        }

        narrative = self._build_narrative(
            feature_name=feature_name,
            global_profile=global_profile,
            decision_contexts=decision_contexts,
            cohort_contexts=cohort_contexts,
            outcome_role=outcome_role,
            interaction_summary=interaction_summary,
        )

        return {
            "schema": "feature_story_v1",
            "feature_idx": int(feature_idx),
            "feature_name": str(feature_name),
            "identity": {
                "feature_idx": int(feature_idx),
                "feature_name": str(feature_name),
                "n_samples": int(n_rows),
            },
            "global_profile": global_profile,
            "decision_contexts": decision_contexts,
            "cohort_attributions": cohort_contexts,
            "interactions": interaction_summary,
            "outcome_role": outcome_role,
            "plot_functions": plot_functions,
            "narrative": narrative,
        }

    def build_feature_stories(
        self,
        *,
        feature_indices: Sequence[int],
        shap_values: Any,
        feature_matrix: Any,
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        decision_context: Optional[Dict[str, Any]] = None,
        cohort_attribution: Optional[Dict[str, Any]] = None,
        top_k_contexts: int = 3,
        top_k_cohorts: int = 3,
    ) -> Dict[str, Any]:
        """Build stories for multiple features and return one bundle."""
        stories = []
        for feature_idx in feature_indices:
            stories.append(
                self.build_feature_story(
                    feature_idx=int(feature_idx),
                    shap_values=shap_values,
                    feature_matrix=feature_matrix,
                    trajectory_metrics=trajectory_metrics,
                    decision_context=decision_context,
                    cohort_attribution=cohort_attribution,
                    top_k_contexts=top_k_contexts,
                    top_k_cohorts=top_k_cohorts,
                )
            )

        return {
            "schema": "feature_story_bundle_v1",
            "metadata": {
                "feature_count": int(len(stories)),
                "top_k_contexts": int(top_k_contexts),
                "top_k_cohorts": int(top_k_cohorts),
            },
            "stories": stories,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_decision_contexts(
        self,
        *,
        decision_context: Optional[Dict[str, Any]],
        feature_idx: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not decision_context:
            return []
        profiles = pd.DataFrame(decision_context.get("feature_profiles", []))
        if profiles.empty:
            return []
        profiles["feature_idx"] = pd.to_numeric(profiles["feature_idx"], errors="coerce").fillna(-1).astype(int)
        row_df = profiles[profiles["feature_idx"] == int(feature_idx)]
        if row_df.empty:
            return []
        row = row_df.iloc[0]
        top_scope = list(row.get("top_scope_features", []))[: max(int(top_k), 1)]

        contexts: List[Dict[str, Any]] = []
        for item in top_scope:
            scope_idx = int(item.get("feature_idx", -1))
            scope_name = self.feature_names[scope_idx] if 0 <= scope_idx < len(self.feature_names) else str(scope_idx)
            contexts.append(
                {
                    "context_type": "scope_to_decider",
                    "scope_feature_idx": scope_idx,
                    "scope_feature_name": scope_name,
                    "cooccurrence_count": int(item.get("cooccurrence_count", 0)),
                    "cooccurrence_rate": round_float(float(item.get("cooccurrence_rate", 0.0))),
                }
            )
        return contexts

    def _extract_cohort_contexts(
        self,
        *,
        cohort_attribution: Optional[Dict[str, Any]],
        feature_idx: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not cohort_attribution:
            return []
        cohorts = list(cohort_attribution.get("cohorts", []))
        selected = [
            c for c in cohorts if int(c.get("decider_feature_idx", -1)) == int(feature_idx)
        ]
        selected.sort(
            key=lambda c: (
                -float(c.get("decision_impact_local", 0.0)),
                -int(c.get("support_count", 0)),
            )
        )

        out: List[Dict[str, Any]] = []
        for cohort in selected[: max(int(top_k), 1)]:
            out.append(
                {
                    "cohort_id": str(cohort.get("cohort_id", "")),
                    "scope_signature": str(cohort.get("scope_signature", "")),
                    "support_count": int(cohort.get("support_count", 0)),
                    "support_rate": round_float(float(cohort.get("support_rate", 0.0))),
                    "risk_rate": round_float(float(cohort.get("risk_rate", 0.0))),
                    "decision_impact_local": round_float(float(cohort.get("decision_impact_local", 0.0))),
                    "sample_id_preview": list(cohort.get("sample_id_preview", [])),
                }
            )
        return out

    def _build_context_functions(
        self,
        *,
        feature_idx: int,
        feature_name: str,
        x_feature: np.ndarray,
        phi_feature: np.ndarray,
        decision_contexts: List[Dict[str, Any]],
        cohort_contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []

        # Build cohort-conditioned response curves when sample previews are available.
        for cohort in cohort_contexts:
            ids = np.asarray(cohort.get("sample_id_preview", []), dtype=int)
            if ids.size < 10:
                continue
            ids = ids[(ids >= 0) & (ids < len(x_feature))]
            if ids.size < 10:
                continue
            curve = _build_binned_curve(
                x_feature[ids],
                phi_feature[ids],
                n_bins=8,
                min_points_per_bin=4,
            )
            if curve is None:
                continue
            functions.append(
                _serialize_curve_function(
                    feature_idx=feature_idx,
                    feature_name=feature_name,
                    context_label=f"cohort::{cohort.get('cohort_id', '')}",
                    curve=curve,
                )
            )

        # Include scope contexts as metadata if no direct sample assignment is available.
        if not functions and decision_contexts:
            for context in decision_contexts:
                functions.append(
                    {
                        "schema": "plot_function_v1",
                        "function_type": "context_descriptor",
                        "context": f"scope::{context.get('scope_feature_name', '')}",
                        "representation": {
                            "kind": "metadata_only",
                            "cooccurrence_count": int(context.get("cooccurrence_count", 0)),
                            "cooccurrence_rate": float(context.get("cooccurrence_rate", 0.0)),
                        },
                    }
                )
        return functions

    def _resolve_partner_idx(
        self,
        *,
        partner_feature_idx: Optional[int],
        feature_idx: int,
        n_features: int,
        decision_context: Optional[Dict[str, Any]],
        x_use: np.ndarray,
        phi_feature: np.ndarray,
    ) -> Optional[int]:
        if partner_feature_idx is not None:
            idx = int(partner_feature_idx)
            if 0 <= idx < n_features and idx != feature_idx:
                return idx

        # Prefer near-leaf partners from decision context.
        if decision_context:
            profiles = pd.DataFrame(decision_context.get("feature_profiles", []))
            if not profiles.empty and "feature_idx" in profiles.columns:
                profiles["feature_idx"] = pd.to_numeric(profiles["feature_idx"], errors="coerce").fillna(-1).astype(int)
                row_df = profiles[profiles["feature_idx"] == int(feature_idx)]
                if not row_df.empty:
                    partners = pd.DataFrame(row_df.iloc[0].get("top_near_leaf_partners", []))
                    if not partners.empty and "feature_idx" in partners.columns:
                        partners["feature_idx"] = pd.to_numeric(partners["feature_idx"], errors="coerce").fillna(-1).astype(int)
                        for idx in partners["feature_idx"].tolist():
                            if 0 <= int(idx) < n_features and int(idx) != feature_idx:
                                return int(idx)

        # Fallback: strongest absolute correlation with this feature's SHAP contributions.
        best_idx: Optional[int] = None
        best_abs_corr = -1.0
        for idx in range(n_features):
            if idx == feature_idx:
                continue
            x_col = np.asarray(x_use[:, idx], dtype=float)
            corr = _safe_corr(x_col, phi_feature)
            if abs(corr) > best_abs_corr:
                best_abs_corr = abs(corr)
                best_idx = idx
        return best_idx

    def _build_global_profile(
        self,
        *,
        x_feature: np.ndarray,
        phi_feature: np.ndarray,
        global_curve: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        mean_abs = float(np.mean(np.abs(phi_feature))) if phi_feature.size else 0.0
        std_abs = float(np.std(np.abs(phi_feature))) if phi_feature.size else 0.0
        p10 = float(np.percentile(phi_feature, 10)) if phi_feature.size else 0.0
        p90 = float(np.percentile(phi_feature, 90)) if phi_feature.size else 0.0
        x_q10 = float(np.percentile(x_feature, 10)) if x_feature.size else 0.0
        x_q90 = float(np.percentile(x_feature, 90)) if x_feature.size else 0.0

        monotonicity = "mixed"
        curve_x = np.asarray(global_curve.get("x", []), dtype=float)
        curve_y = np.asarray(global_curve.get("y", []), dtype=float)
        if curve_x.size >= 3 and curve_y.size >= 3:
            diffs = np.diff(curve_y)
            frac_pos = float(np.mean(diffs > 0))
            frac_neg = float(np.mean(diffs < 0))
            if frac_pos >= 0.75:
                monotonicity = "increasing"
            elif frac_neg >= 0.75:
                monotonicity = "decreasing"

        return {
            "mean_abs_shap": round_float(mean_abs),
            "std_abs_shap": round_float(std_abs),
            "contribution_p10": round_float(p10),
            "contribution_p90": round_float(p90),
            "feature_value_p10": round_float(x_q10),
            "feature_value_p90": round_float(x_q90),
            "monotonicity": monotonicity,
            "response_curve": {
                "x": list(global_curve.get("x", [])),
                "y": list(global_curve.get("y", [])),
                "y_std": list(global_curve.get("y_std", [])),
                "count": list(global_curve.get("count", [])),
            },
        }

    def _build_outcome_role(
        self,
        *,
        feature_idx: int,
        phi_feature: np.ndarray,
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        abs_phi = np.abs(np.asarray(phi_feature, dtype=float))
        mean_abs = float(np.mean(abs_phi)) if abs_phi.size else 0.0
        std_abs = float(np.std(abs_phi)) if abs_phi.size else 0.0
        stability = float(std_abs / max(mean_abs, 1e-9))

        out = {
            "mean_abs_shap": round_float(mean_abs),
            "stability_cv": round_float(stability),
            "predictive_polarity": round_float(float(np.mean(np.asarray(phi_feature, dtype=float)))),
        }

        if trajectory_metrics and feature_idx in trajectory_metrics:
            metrics = trajectory_metrics[feature_idx]
            accel = np.asarray(metrics.get("acceleration", []), dtype=float)
            fci = np.asarray(metrics.get("fci", []), dtype=float)
            gains = np.asarray(metrics.get("gains", []), dtype=float)
            mid = len(accel) // 2
            late_accel = float(np.mean(accel[mid:])) if accel.size else 0.0
            avg_fci = float(np.mean(fci)) if fci.size else 0.0
            conditional_score = mean_abs * (1.0 + np.tanh(max(late_accel, 0.0))) * (1.0 + np.log1p(max(avg_fci, 0.0)))

            out.update(
                {
                    "avg_fci": round_float(avg_fci),
                    "late_acceleration_mean": round_float(late_accel),
                    "total_gain": round_float(float(np.sum(gains)) if gains.size else 0.0),
                    "conditional_importance_score": round_float(float(conditional_score)),
                }
            )
        return out

    def _build_narrative(
        self,
        *,
        feature_name: str,
        global_profile: Dict[str, Any],
        decision_contexts: List[Dict[str, Any]],
        cohort_contexts: List[Dict[str, Any]],
        outcome_role: Dict[str, Any],
        interaction_summary: Dict[str, Any],
    ) -> Dict[str, str]:
        monotonicity = str(global_profile.get("monotonicity", "mixed"))
        role_score = outcome_role.get("conditional_importance_score", outcome_role.get("mean_abs_shap", 0.0))
        top_scope = decision_contexts[0]["scope_feature_name"] if decision_contexts else "diverse scopes"
        top_cohort = cohort_contexts[0]["cohort_id"] if cohort_contexts else "no dominant cohort"
        partner = interaction_summary.get("partner_feature_name") or "no stable partner"

        return {
            "headline": (
                f"{feature_name} shows a {monotonicity} global response with role score {round_float(float(role_score))}."
            ),
            "context": f"Most repeated decision scope before this feature is: {top_scope}.",
            "cohort": f"Most relevant cohort attribution: {top_cohort}.",
            "interaction": f"Primary interaction partner: {partner}.",
        }


def _coerce_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return np.asarray(arr[:, :, -1])
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _build_binned_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_bins: int = 12,
    min_points_per_bin: int = 10,
) -> Optional[Dict[str, List[float]]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n < 20:
        return None
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 20:
        return None

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)))
    if edges.size < 3:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if np.isclose(x_min, x_max):
            return None
        edges = np.linspace(x_min, x_max, 3)

    bin_ids = np.digitize(x, bins=edges[1:-1], right=False)
    centers: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []

    for bin_idx in range(edges.size - 1):
        members = bin_ids == bin_idx
        cnt = int(np.sum(members))
        if cnt < max(int(min_points_per_bin), 3):
            continue
        x_bin = x[members]
        y_bin = y[members]
        centers.append(float(np.mean(x_bin)))
        means.append(float(np.mean(y_bin)))
        stds.append(float(np.std(y_bin)))
        counts.append(cnt)

    if len(centers) < 2:
        return None

    order = np.argsort(np.asarray(centers))
    return {
        "x": np.asarray(centers)[order].astype(float).tolist(),
        "y": np.asarray(means)[order].astype(float).tolist(),
        "y_std": np.asarray(stds)[order].astype(float).tolist(),
        "count": np.asarray(counts)[order].astype(int).tolist(),
    }


def _serialize_curve_function(
    *,
    feature_idx: int,
    feature_name: str,
    context_label: str,
    curve: Dict[str, List[float]],
) -> Dict[str, Any]:
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


def _build_interaction_surface(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    bins: int = 8,
    min_points_per_cell: int = 4,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    n = min(x.size, y.size, z.size)
    if n < 30:
        return None
    x = x[:n]
    y = y[:n]
    z = z[:n]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if x.size < 30:
        return None

    x_edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, max(int(bins), 3) + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, max(int(bins), 3) + 1)))
    if x_edges.size < 3 or y_edges.size < 3:
        return None

    x_bin = np.digitize(x, bins=x_edges[1:-1], right=False)
    y_bin = np.digitize(y, bins=y_edges[1:-1], right=False)
    x_cells = x_edges.size - 1
    y_cells = y_edges.size - 1
    z_grid = np.full((y_cells, x_cells), np.nan, dtype=float)
    cnt_grid = np.zeros((y_cells, x_cells), dtype=int)

    for yi in range(y_cells):
        for xi in range(x_cells):
            idx = (x_bin == xi) & (y_bin == yi)
            cnt = int(np.sum(idx))
            if cnt < max(int(min_points_per_cell), 3):
                continue
            z_grid[yi, xi] = float(np.mean(z[idx]))
            cnt_grid[yi, xi] = cnt

    if int(np.sum(cnt_grid > 0)) == 0:
        return None

    return {
        "schema": "plot_function_v1",
        "function_type": "interaction_surface",
        "representation": {
            "kind": "binned_surface",
            "x_edges": x_edges.astype(float).tolist(),
            "y_edges": y_edges.astype(float).tolist(),
            "z_grid_mean": np.nan_to_num(z_grid, nan=np.nan).tolist(),
            "count_grid": cnt_grid.astype(int).tolist(),
        },
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)
