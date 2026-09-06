"""Correlates final-step SHAP values with trajectory-derived behavioral metrics.

The central output is a *conditional importance score* that rewards features
that simultaneously have high mean absolute SHAP impact *and* late-phase
acceleration on high-coverage splits:

    conditional_importance = mean_abs_shap
                             × (1 + tanh(max(late_accel, 0)))
                             × (1 + log1p(avg_fci))
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .trajectory_utils import round_float


class ShapBehaviorCorrelator:
    """Links final-step SHAP variation with FCI and acceleration metrics."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def correlate(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        final_shap_values: Any,
        top_k: int = 15,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Compute per-feature SHAP statistics and correlate them with trajectory metrics.

        Args:
            trajectory_metrics: Output of ``TreeAnalyzer.compute_trajectory_metrics()``.
            final_shap_values: Final-iteration SHAP values shaped ``(n_samples, n_features)``.
                               Also accepts binary-class output ``(n_samples, n_features, 2)``,
                               in which case the last class slice is used.
            top_k: Number of rows kept in each ranked output view.

        Returns:
            Dictionary with:
            - ``n_features_analyzed``: int
            - ``metric_definitions``: human-readable description of derived metrics
            - ``correlations``: Pearson & Spearman matrices plus three focus pairs
            - ``feature_metrics``: full per-feature table sorted by conditional importance
            - ``rankings``: four ranked views (conditional importance, SHAP variation,
              late acceleration, and high-SHAP × high-late-acceleration)
        """
        confusion_metric = _normalize_confusion_metric(confusion_metric)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        shap_array = _coerce_shap(final_shap_values)
        n_features = min(int(shap_array.shape[1]), len(self.feature_names))

        rows = []
        for feature_idx in range(n_features):
            metrics = trajectory_metrics.get(feature_idx, {})
            rows.append(
                self._build_row(
                    feature_idx,
                    metrics,
                    shap_array[:, feature_idx],
                    confusion_metric=confusion_metric,
                )
            )

        feature_df = pd.DataFrame(rows)
        if feature_df.empty:
            return {
                "n_features_analyzed": 0,
                "correlations": {"pearson": {}, "spearman": {}},
                "feature_metrics": [],
                "rankings": {},
            }

        numeric_cols = [
            "mean_abs_shap", "std_abs_shap", "shap_tail_span",
            "avg_fci", "last_fci",
            "avg_hessian_raw", "last_hessian_raw",
            "early_acceleration_mean", "late_acceleration_mean",
            "acceleration_volatility", "total_gain",
            "conditional_importance_score",
        ]
        pearson_df = feature_df[numeric_cols].corr(method="pearson").fillna(0.0)  # type: ignore[call-arg]
        spearman_df = feature_df[numeric_cols].corr(method="spearman").fillna(0.0)  # type: ignore[call-arg]

        round_cols = numeric_cols + ["p90_abs_shap", "p10_abs_shap"]

        hi_shap = feature_df["mean_abs_shap"].quantile(0.75)
        hi_late_accel = feature_df["late_acceleration_mean"].quantile(0.75)
        high_both_mask = (
            (feature_df["mean_abs_shap"] >= hi_shap)
            & (feature_df["late_acceleration_mean"] >= hi_late_accel)
        )

        def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
            out = df.copy()
            for col in round_cols:
                if col in out.columns:
                    out[col] = out[col].map(_safe_round)
            return [{str(k): v for k, v in rec.items()} for rec in out.to_dict(orient="records")]

        rankings = {
            "top_conditional_importance": _records(
                feature_df.sort_values("conditional_importance_score", ascending=False).head(top_k)
            ),
            "top_shap_variation": _records(
                feature_df.sort_values("std_abs_shap", ascending=False).head(top_k)
            ),
            "top_late_acceleration": _records(
                feature_df.sort_values("late_acceleration_mean", ascending=False).head(top_k)
            ),
            "high_shap_high_late_acceleration": _records(
                pd.DataFrame(feature_df[high_both_mask])
                .sort_values("conditional_importance_score", ascending=False)
                .head(top_k)
            ),
        }

        export_df = feature_df.copy()
        for col in round_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].map(_safe_round)

        return {
            "n_features_analyzed": int(len(feature_df)),
            "metric_definitions": {
                "shap_tail_span": "p90(|SHAP|) - p10(|SHAP|) at final iteration",
                "avg_fci": (
                    "Mean confusion signal across iterations "
                    f"(selected metric: {confusion_metric})"
                ),
                "avg_hessian_raw": "Mean per-split raw Hessian (HVI signal) across iterations",
                "late_acceleration_mean": "Mean acceleration on second half of training",
                "conditional_importance_score": (
                    "mean_abs_shap * (1 + tanh(max(late_accel,0))) * (1 + log1p(avg_fci))"
                ),
            },
            "correlations": {
                "pearson": pearson_df.round(4).to_dict(),
                "spearman": spearman_df.round(4).to_dict(),
                "focus_pairs": {
                    "std_abs_shap_vs_avg_fci": {
                        "pearson": _safe_round(pearson_df.loc["std_abs_shap", "avg_fci"]),
                        "spearman": _safe_round(spearman_df.loc["std_abs_shap", "avg_fci"]),
                    },
                    "std_abs_shap_vs_late_acceleration_mean": {
                        "pearson": _safe_round(pearson_df.loc["std_abs_shap", "late_acceleration_mean"]),
                        "spearman": _safe_round(spearman_df.loc["std_abs_shap", "late_acceleration_mean"]),
                    },
                    "mean_abs_shap_vs_conditional_importance_score": {
                        "pearson": _safe_round(pearson_df.loc["mean_abs_shap", "conditional_importance_score"]),
                        "spearman": _safe_round(spearman_df.loc["mean_abs_shap", "conditional_importance_score"]),
                    },
                },
            },
            "feature_metrics": (
                export_df.sort_values("conditional_importance_score", ascending=False)
                .to_dict(orient="records")
            ),
            "rankings": rankings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        feature_idx: int,
        metrics: Dict[str, Any],
        shap_col: Any,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Compute per-feature statistics for one feature."""
        accelerations = np.asarray(metrics.get("acceleration", []), dtype=float)
        fcis = np.asarray(_confusion_series(metrics, confusion_metric), dtype=float)
        hessian_raw = np.asarray(metrics.get("hessian_mean_raw", []), dtype=float)
        gains = np.asarray(metrics.get("gains", []), dtype=float)

        mid = len(accelerations) // 2
        early_accel = float(np.mean(accelerations[:mid])) if mid else 0.0
        late_accel = float(np.mean(accelerations[mid:])) if accelerations.size else 0.0

        abs_shap = np.abs(np.asarray(shap_col, dtype=float))
        mean_abs_shap = float(np.mean(abs_shap)) if abs_shap.size else 0.0
        std_abs_shap = float(np.std(abs_shap)) if abs_shap.size else 0.0
        p90 = float(np.percentile(abs_shap, 90)) if abs_shap.size else 0.0
        p10 = float(np.percentile(abs_shap, 10)) if abs_shap.size else 0.0

        avg_fci = float(np.mean(fcis)) if fcis.size else 0.0
        last_fci = float(fcis[-1]) if fcis.size else 0.0
        avg_hessian_raw = float(np.mean(hessian_raw)) if hessian_raw.size else 0.0
        last_hessian_raw = float(hessian_raw[-1]) if hessian_raw.size else 0.0
        accel_volatility = float(np.std(accelerations)) if accelerations.size else 0.0
        total_gain = float(np.sum(gains)) if gains.size else 0.0

        # Conditional-importance: strong final SHAP × stable high-coverage late lift.
        accel_boost = 1.0 + float(np.tanh(max(late_accel, 0.0)))
        fci_boost = 1.0 + float(np.log1p(max(avg_fci, 0.0)))
        conditional_score = mean_abs_shap * accel_boost * fci_boost

        return {
            "feature_idx": int(feature_idx),
            "feature_name": metrics.get("feature_name", self.feature_names[feature_idx]),
            "mean_abs_shap": mean_abs_shap,
            "std_abs_shap": std_abs_shap,
            "p90_abs_shap": p90,
            "p10_abs_shap": p10,
            "shap_tail_span": p90 - p10,
            "avg_fci": avg_fci,
            "last_fci": last_fci,
            "avg_hessian_raw": avg_hessian_raw,
            "last_hessian_raw": last_hessian_raw,
            "confusion_metric": confusion_metric,
            "early_acceleration_mean": early_accel,
            "late_acceleration_mean": late_accel,
            "acceleration_volatility": accel_volatility,
            "total_gain": total_gain,
            "conditional_importance_score": conditional_score,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _coerce_shap(values: Any) -> "np.ndarray":
    """Normalise SHAP arrays to shape ``(n_samples, n_features)``."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 3:
        # Binary classifier: (samples, features, classes) — use the last class.
        return np.asarray(arr[:, :, -1])
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


def _safe_round(value: Any) -> float:
    """Null-safe float rounding for correlation matrix export."""
    if pd.isna(value):
        return 0.0
    try:
        numeric = float(np.real(np.asarray(value).reshape(-1)[0]))
    except (TypeError, ValueError, IndexError):
        return 0.0
    return round_float(numeric)


def _confusion_series(metrics: Dict[str, Any], confusion_metric: str) -> List[float]:
    """Return preferred confusion signal (raw Hessian mean or FCI fallback)."""
    if confusion_metric == "hessian" and "hessian_mean_raw" in metrics and metrics.get("hessian_mean_raw"):
        return list(metrics.get("hessian_mean_raw", []))
    return list(metrics.get("fci", []))


def _normalize_confusion_metric(value: str) -> str:
    metric = str(value or "hessian").strip().lower()
    if metric not in {"hessian", "fci"}:
        raise ValueError("confusion_metric must be 'hessian' or 'fci'")
    return metric
