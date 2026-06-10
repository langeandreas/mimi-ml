"""SHAP value analysis and trajectory generation."""

from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from .trajectory_utils import round_float, TOP_K_SHAP_FEATURES, expand_shap_step


class ShapAnalyzer:
    """Analyzes SHAP values and generates SHAP-based trajectories."""

    def __init__(self, explain_callback):
        """Initialize with ExplainerCallback containing SHAP history."""
        self.explain = explain_callback
        self.change_point_epsilon: Optional[float] = None

    def set_change_point_epsilon(self, epsilon: Optional[float]) -> None:
        """Set the relative drift threshold for change-point filtering."""
        if epsilon is not None and epsilon < 0:
            raise ValueError("change_point_epsilon must be >= 0 when provided")
        self.change_point_epsilon = epsilon

    def build_shap_step_payloads(self) -> List[Dict[str, Any]]:
        """Builds per-iteration SHAP payloads for filtering and exports.
        
        Returns list of step payloads with impact magnitude and top features.
        """
        step_payloads = []
        for entry in self.explain.shap_values_history:
            shap_values = np.asarray(entry["shap_values"])

            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            # Tail-based feature impact: lower 30% and upper 30% SHAP values per feature
            tail_count = max(1, int(np.ceil(shap_values.shape[0] * 0.3)))
            sorted_shap = np.sort(shap_values, axis=0)
            low_impact = sorted_shap[:tail_count, :].mean(axis=0)
            high_impact = sorted_shap[-tail_count:, :].mean(axis=0)
            impact_magnitude = (np.abs(low_impact) + np.abs(high_impact)) / 2.0
            top_idx = [
                int(i)
                for i in np.argsort(impact_magnitude)[::-1]
                if impact_magnitude[i] > 0
            ]

            step_payloads.append({
                "entry": entry,
                "iteration": int(entry["iteration"]),
                "impact_magnitude": impact_magnitude,
                "top_idx": top_idx,
                "high_impact": high_impact,
                "low_impact": low_impact,
            })

        return step_payloads

    def filter_change_point_payloads(self, step_payloads: List[Dict]) -> List[Dict]:
        """Keeps SHAP payloads at significant change points according to epsilon.
        
        If epsilon is None or <= 0, retains all steps.
        """
        if not step_payloads:
            return []

        epsilon = self.change_point_epsilon
        retain_all = epsilon is None or epsilon <= 0
        threshold = 0.0
        if not retain_all:
            if epsilon is None:
                raise ValueError("change_point_epsilon must be provided when filtering is enabled")
            threshold = float(epsilon)

        retained_steps = []
        previous_magnitude = None

        for step in step_payloads:
            if retain_all or previous_magnitude is None:
                retained_steps.append(step)
                previous_magnitude = step["impact_magnitude"]
                continue

            baseline = np.maximum(np.abs(previous_magnitude), 1e-12)
            relative_change = np.abs(step["impact_magnitude"] - previous_magnitude) / baseline

            if float(np.max(relative_change)) >= threshold:
                retained_steps.append(step)
                previous_magnitude = step["impact_magnitude"]

        # Always keep the final step so late-stage model state is represented
        if retained_steps[-1]["iteration"] != step_payloads[-1]["iteration"]:
            retained_steps.append(step_payloads[-1])

        return retained_steps

    def summarize_shap_step(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Summarizes a single SHAP step with feature impacts."""
        shap_values = np.asarray(entry["shap_values"])

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Tail-based feature impact
        tail_count = max(1, int(np.ceil(shap_values.shape[0] * 0.3)))
        sorted_shap = np.sort(shap_values, axis=0)
        low_impact = sorted_shap[:tail_count, :].mean(axis=0)
        high_impact = sorted_shap[-tail_count:, :].mean(axis=0)
        impact_magnitude = (np.abs(low_impact) + np.abs(high_impact)) / 2.0
        top_idx = [
            int(i)
            for i in np.argsort(impact_magnitude)[::-1][:TOP_K_SHAP_FEATURES]
            if impact_magnitude[i] > 0
        ]

        return {
            "iteration": int(entry["iteration"]),
            "base_value": round_float(np.asarray(entry["base_value"]).mean()),
            "shap_magnitude": round_float(impact_magnitude.sum()),
            "top_shap_features": [
                {
                    "feature_index": int(i),
                    "mean_abs_shap": round_float(impact_magnitude[i]),
                    "high_value_feature_impact": round_float(high_impact[i]),
                    "low_value_feature_impact": round_float(low_impact[i]),
                }
                for i in top_idx
            ],
        }

    def get_sample_size(self) -> int:
        """Get the sample size used in SHAP calculation."""
        if self.explain.shap_values_history:
            return int(np.asarray(self.explain.shap_values_history[0]["shap_values"]).shape[0])
        return 0
