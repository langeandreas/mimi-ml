"""SHAP value analysis and trajectory generation."""

import json
from numbers import Integral
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .trajectory_utils import (
    round_float,
    TOP_K_SHAP_FEATURES,
    expand_shap_step,
    generate_output_file,
    generate_output_dir,
    substitute_feature_names,
)


class ShapAnalyzer:
    """Analyzes SHAP values and generates SHAP-based trajectories."""

    def __init__(self, explain_callback):
        """Initialize with ExplainerCallback containing SHAP history."""
        self.explain = explain_callback
        self.change_point_epsilon: Optional[float] = None


    def get_sample_size(self) -> int:
        """Get the sample size used in SHAP calculation."""
        if self.explain.shap_values_history:
            return int(np.asarray(self.explain.shap_values_history[0]["shap_values"]).shape[0])
        return 0


    def set_change_point_epsilon(self, epsilon: Optional[float]) -> None:
        """Set the relative drift threshold for change-point filtering."""
        if epsilon is not None and epsilon < 0:
            raise ValueError("change_point_epsilon must be >= 0 when provided")
        self.change_point_epsilon = epsilon

    def build_shap_step_payloads(self) -> List[Dict[str, Any]]:
        """Builds per-iteration SHAP payloads for filtering and exports.
        
        Returns list of step payloads with impact magnitude and top features.
        """
        steps_shap = []
        for entry in self.explain.shap_values_history:
            shap_values = np.asarray(entry["shap_values"])

            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            # tail-based feature impact, lower 30% and upper 30% average SHAP values per feature
            tail_count = max(1, int(np.ceil(shap_values.shape[0] * 0.3)))
            sorted_shap = np.sort(shap_values, axis=0)
            low_impact = sorted_shap[:tail_count, :].mean(axis=0) 
            high_impact = sorted_shap[-tail_count:, :].mean(axis=0)
            impact_magnitude = (np.abs(low_impact) + np.abs(high_impact)) / 2.0 
            top_idx = [ # all features with positive impact magnitude
                int(i)
                for i in np.argsort(impact_magnitude)[::-1]
                if impact_magnitude[i] > 0
            ]

            steps_shap.append({
                "entry": entry,
                "iteration": int(entry["iteration"]),
                "impact_magnitude": impact_magnitude,
                "top_idx": top_idx,
                "high_impact": high_impact,
                "low_impact": low_impact,
            })

        return steps_shap


    def filter_change_point_payloads(self, steps_shap: List[Dict]) -> List[Dict]:
        """Keeps SHAP payloads at significant change points according to epsilon.
        
        If epsilon is None or <= 0, keeps all steps.
        """
        if not steps_shap:
            return []

        epsilon = self.change_point_epsilon
        keep_all = epsilon is None or epsilon <= 0
        threshold = 0.0
        if not keep_all:
            if epsilon is None:
                raise ValueError("change_point_epsilon must be provided when filtering is enabled")
            threshold = float(epsilon)

        filtered_steps = []
        previous_magnitude = None

        for step in steps_shap:
            if keep_all or previous_magnitude is None:
                filtered_steps.append(step)
                previous_magnitude = step["impact_magnitude"]
                continue

            baseline = np.maximum(np.abs(previous_magnitude), 1e-12) # avoid division by zero
            relative_change = np.abs(step["impact_magnitude"] - previous_magnitude) / baseline

            if float(np.max(relative_change)) >= threshold:
                filtered_steps.append(step)
                previous_magnitude = step["impact_magnitude"]

        # always keep the final step so late-stage model state is represented
        if filtered_steps[-1]["iteration"] != steps_shap[-1]["iteration"]:
            filtered_steps.append(steps_shap[-1])

        return filtered_steps


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


    def configure(
        self,
        feature_names: List[str],
        save: bool = False,
        path: str = '../data/results/',
        tree_analyzer=None,
        model=None,
    ) -> None:
        """Configure the analyzer with output settings and cross-analyzer dependencies.

        Args:
            feature_names: Ordered list of feature name strings
            save: Whether to persist generated artifacts by default
            path: Output directory or file path for saved artifacts
            tree_analyzer: TreeAnalyzer instance for behavior correlation methods
            model: XGBoost model used when no explicit booster is provided
        """
        self.feature_names = feature_names
        self.save = save
        self.path = path
        self.tree_analyzer = tree_analyzer
        self.model = model
        self.shap_history_df = self.generate_shap_df()


    def generate_shap_df(self) -> pd.DataFrame:
        """Generates a DataFrame containing tail-based SHAP values.

        If change_point_epsilon is provided (> 0), only change-point iterations are kept
        based on relative drift in per-feature impact magnitude.

        Returns:
            A DataFrame containing SHAP impacts for retained iterations
        """
        step_payloads = self.build_shap_step_payloads()
        empty = pd.DataFrame(columns=["iteration", "feature_index", "mean_abs_shap", "high_value_feature_impact", "low_value_feature_impact"])
        if not step_payloads:
            return empty

        retained_steps = self.filter_change_point_payloads(step_payloads)
        feature_names = getattr(self, 'feature_names', [])

        frames = []
        for step in retained_steps:
            top_idx = step["top_idx"]
            frames.append(pd.DataFrame({
                "iteration": step["iteration"],
                "feature_index": [feature_names[i] if i < len(feature_names) else str(i) for i in top_idx],
                "mean_abs_shap": [round_float(step["impact_magnitude"][i]) for i in top_idx],
                "high_value_feature_impact": [round_float(step["high_impact"][i]) for i in top_idx],
                "low_value_feature_impact": [round_float(step["low_impact"][i]) for i in top_idx],
            }))

        if not frames:
            return empty

        shap_impact_df = pd.concat(frames, ignore_index=True)
        shap_impact_df = substitute_feature_names(shap_impact_df, "feature_index", feature_names)
        return shap_impact_df


    @staticmethod
    def _normalize_binary_shap_values(shap_values_output: Any) -> np.ndarray:
        """Normalize SHAP outputs from binary classifiers into a 2D array."""
        if isinstance(shap_values_output, list):
            return np.asarray(shap_values_output[-1])

        shap_array = np.asarray(shap_values_output)
        if shap_array.ndim == 2:
            return shap_array
        if shap_array.ndim == 3:
            return np.asarray(shap_array[:, :, -1])
        raise ValueError(f"Unsupported SHAP output shape: {shap_array.shape}")


    def generate_shap_json(
        self,
        save=None,
        readable_feature_names: bool = True,
        compact: bool = True,
    ):
        """Generates the SHAP summary JSON.

        Args:
            save: Override for default save setting
            readable_feature_names: Replace feature IDs with human-readable names
            compact: Use compact format to reduce file size

        Returns:
            Dictionary containing SHAP summary
        """
        step_payloads = self.build_shap_step_payloads()
        retained_steps = self.filter_change_point_payloads(step_payloads)
        shap_steps = [self.summarize_shap_step(step["entry"]) for step in retained_steps]
        sample_size = self.get_sample_size()
        feature_names = getattr(self, 'feature_names', [])

        shap_json: Dict[str, Any] = {
            "shap_sample_size": sample_size,
            "iteration_progress": shap_steps,
        }

        if readable_feature_names:
            for step in shap_json["iteration_progress"]:
                top_features_df = pd.DataFrame(step.get("top_shap_features", []))
                top_features_df = substitute_feature_names(top_features_df, "feature_index", feature_names)
                step["top_shap_features"] = top_features_df.to_dict(orient="records")

        if compact:
            compact_steps = [
                {
                    "i": int(step["iteration"]),
                    "b": round_float(step["base_value"]),
                    "m": round_float(step["shap_magnitude"]),
                    "f": [
                        [
                            item["feature_index"],
                            round_float(item["mean_abs_shap"]),
                            round_float(item["high_value_feature_impact"]),
                            round_float(item["low_value_feature_impact"]),
                        ]
                        for item in step.get("top_shap_features", [])
                    ],
                }
                for step in shap_json["iteration_progress"]
            ]
            shap_json = {
                "schema_version": 2,
                "shap_sample_size": shap_json["shap_sample_size"],
                "iteration_row_format": ["i", "b", "m", "f"],
                "feature_row_format": [
                    "feature_index",
                    "mean_abs_shap",
                    "high_value_feature_impact",
                    "low_value_feature_impact",
                ],
                "iteration_progress": compact_steps,
            }

        if self.save or save:
            with open(generate_output_file(self.path, "trajectory_shap.json"), 'w', encoding='utf-8') as fp:
                json.dump(shap_json, fp, ensure_ascii=False, separators=(',', ':'))

        return shap_json


    def generate_shap_llm_summary_json(
        self,
        save=None,
        top_k_features: int = 10,
        milestone_count: int = 6,
        readable_feature_names: bool = True,
    ) -> Dict[str, Any]:
        """Generate a compact, LLM-oriented SHAP summary with key trajectory signals.

        Prioritizes trend and ranking information over per-iteration full detail.

        Args:
            save: Override for default save setting
            top_k_features: Number of top features to include
            milestone_count: Number of milestone iterations to sample
            readable_feature_names: Use human-readable feature names

        Returns:
            Dictionary with LLM-optimized SHAP summary
        """
        if top_k_features < 1:
            raise ValueError("top_k_features must be >= 1")
        if milestone_count < 2:
            raise ValueError("milestone_count must be >= 2")

        shap_json = self.generate_shap_json(save=False, readable_feature_names=readable_feature_names, compact=True)
        shap_steps = [expand_shap_step(step) for step in shap_json.get("iteration_progress", [])]
        if not shap_steps:
            summary: Dict[str, Any] = {
                "schema_version": 1,
                "summary_type": "shap_llm",
                "shap_sample_size": int(shap_json.get("shap_sample_size", 0)),
                "iteration_count": 0,
                "top_feature_trends": [],
                "milestones": [],
                "change_points": [],
            }
            if self.save or save:
                with open(generate_output_file(self.path, "trajectory_shap_llm_summary.json"), 'w', encoding='utf-8') as fp:
                    json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))
            return summary

        shap_steps = sorted(shap_steps, key=lambda s: int(s["iteration"]))

        feature_series: Dict[str, List[Dict[str, Any]]] = {}
        for step in shap_steps:
            iteration = int(step["iteration"])
            for item in step.get("top_shap_features", []):
                feature_index = str(item["feature_index"])
                feature_series.setdefault(feature_index, []).append({
                    "iteration": iteration,
                    "mean_abs_shap": float(item["mean_abs_shap"]),
                    "high": float(item["high_value_feature_impact"]),
                    "low": float(item["low_value_feature_impact"]),
                })

        ranked_features = sorted(
            feature_series.items(),
            key=lambda kv: max(p["mean_abs_shap"] for p in kv[1]),
            reverse=True,
        )[:top_k_features]

        top_feature_trends = []
        for feature_index, points in ranked_features:
            pts = sorted(points, key=lambda p: p["iteration"])
            first, last = pts[0], pts[-1]
            peak = max(p["mean_abs_shap"] for p in pts)
            top_feature_trends.append({
                "feature_index": feature_index,
                "first_iteration": int(first["iteration"]),
                "last_iteration": int(last["iteration"]),
                "first_mean_abs_shap": round_float(first["mean_abs_shap"]),
                "last_mean_abs_shap": round_float(last["mean_abs_shap"]),
                "peak_mean_abs_shap": round_float(peak),
                "delta_mean_abs_shap": round_float(last["mean_abs_shap"] - first["mean_abs_shap"]),
                "last_high_value_impact": round_float(last["high"]),
                "last_low_value_impact": round_float(last["low"]),
            })

        total_steps = len(shap_steps)
        milestone_indexes = sorted(set(
            np.linspace(0, total_steps - 1, num=min(milestone_count, total_steps), dtype=int).tolist()
        ))
        milestones = []
        for idx in milestone_indexes:
            step = shap_steps[idx]
            top_features = sorted(
                step.get("top_shap_features", []),
                key=lambda item: float(item["mean_abs_shap"]),
                reverse=True,
            )[:min(5, top_k_features)]
            milestones.append({
                "iteration": int(step["iteration"]),
                "base_value": round_float(step["base_value"]),
                "shap_magnitude": round_float(step["shap_magnitude"]),
                "top_features": [
                    {
                        "feature_index": str(item["feature_index"]),
                        "mean_abs_shap": round_float(item["mean_abs_shap"]),
                    }
                    for item in top_features
                ],
            })

        change_points = []
        previous_magnitude = None
        for step in shap_steps:
            magnitude = float(step["shap_magnitude"])
            if previous_magnitude is None:
                previous_magnitude = magnitude
                continue
            delta = magnitude - previous_magnitude
            if abs(delta) >= 0.05:
                top_feature = None
                if step.get("top_shap_features"):
                    top_feature = str(
                        sorted(
                            step["top_shap_features"],
                            key=lambda item: float(item["mean_abs_shap"]),
                            reverse=True,
                        )[0]["feature_index"]
                    )
                change_points.append({
                    "iteration": int(step["iteration"]),
                    "delta_shap_magnitude": round_float(delta),
                    "shap_magnitude": round_float(magnitude),
                    "top_feature": top_feature,
                })
            previous_magnitude = magnitude

        summary = {
            "schema_version": 1,
            "summary_type": "shap_llm",
            "source_schema_version": int(shap_json.get("schema_version", 1)),
            "shap_sample_size": int(shap_json.get("shap_sample_size", 0)),
            "iteration_count": int(total_steps),
            "iteration_range": [int(shap_steps[0]["iteration"]), int(shap_steps[-1]["iteration"])],
            "magnitude": {
                "start": round_float(shap_steps[0]["shap_magnitude"]),
                "end": round_float(shap_steps[-1]["shap_magnitude"]),
                "max": round_float(max(float(s["shap_magnitude"]) for s in shap_steps)),
                "delta": round_float(float(shap_steps[-1]["shap_magnitude"]) - float(shap_steps[0]["shap_magnitude"])),
            },
            "top_feature_trends": top_feature_trends,
            "milestones": milestones,
            "change_points": change_points,
        }

        if self.save or save:
            with open(generate_output_file(self.path, "trajectory_shap_llm_summary.json"), 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))

        return summary


    def generate_shap_behavior_correlation_json(
        self,
        booster=None,
        final_shap_values=None,
        save=None,
        top_k_features: int = 15,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Correlate final-step SHAP variation with FCI/acceleration signals.

        Produces a compact, high-information payload for identifying conditionally
        important features that combine strong SHAP effects with late training dynamics.

        Args:
            booster: Optional XGBoost booster. If None, inferred from self.model.
            final_shap_values: Optional final SHAP array. If None, uses the last
                entry in explain.shap_values_history.
            save: Override default save behavior.
            top_k_features: Number of rows retained in ranked sections.

        Returns:
            Dictionary with feature metrics, rankings, and correlation matrices.
        """
        if booster is None:
            model = self.model
            if model is None:
                raise ValueError("Unable to infer booster. Provide booster explicitly or set self.model.")
            estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
            if not hasattr(estimator, "get_booster"):
                raise ValueError("Unable to infer booster. Provide booster explicitly or set self.model.")
            booster = estimator.get_booster()

        if final_shap_values is None:
            if not self.explain.shap_values_history:
                raise ValueError("No SHAP history available. Provide final_shap_values explicitly.")
            final_shap_values = self.explain.shap_values_history[-1]["shap_values"]

        if self.tree_analyzer is None:
            raise ValueError("tree_analyzer is not configured. Call configure() before using this method.")

        trajectory_metrics = self.tree_analyzer.compute_trajectory_metrics(booster)
        summary = self.tree_analyzer.correlate_final_shap_with_behavior(
            trajectory_metrics=trajectory_metrics,
            final_shap_values=final_shap_values,
            top_k=top_k_features,
            confusion_metric=confusion_metric,
        )
        summary["quadrant_categorization"] = self.tree_analyzer.categorize_features_into_quadrants(
            trajectory_metrics=trajectory_metrics,
            confusion_metric=confusion_metric,
        )
        summary["summary_type"] = "shap_behavior_correlation"
        summary["top_k_features"] = int(top_k_features)
        summary["confusion_metric"] = str(confusion_metric)

        if self.save or save:
            with open(generate_output_file(self.path, "trajectory_shap_behavior_correlation.json"), 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))

        return summary


    def plot_shap_history(
        self,
        feature_names: Optional[Union[List[str], int]] = None,
        iso3: Optional[str] = None,
        y: str = 'abs',
        title: Optional[str] = None,
        titlefontsize: int = 14,
        save: bool = False,
    ):
        """Plot SHAP trajectories with one line per feature over iterations."""
        if self.shap_history_df.empty:
            print("No SHAP history data available to plot.")
            return

        plot_df = self.shap_history_df.copy()
        if isinstance(feature_names, list) and feature_names:
            selected = [str(name) for name in feature_names]
            plot_df = plot_df[plot_df["feature_index"].astype(str).isin(selected)]
        elif isinstance(feature_names, Integral) and not isinstance(feature_names, bool) and feature_names:
            top_series = pd.Series(plot_df.groupby("feature_index")["mean_abs_shap"].max())
            top_features = top_series.nlargest(int(feature_names)).index.tolist()
            plot_df = plot_df[plot_df["feature_index"].astype(str).isin(top_features)]

        if plot_df.empty:
            print("No SHAP history data available for the selected features.")
            return

        unique_features = sorted(set(plot_df["feature_index"].astype(str).tolist()))
        plt.figure(figsize=(12, len(unique_features) * 0.35 + 4))
        cmap = plt.get_cmap("tab20")

        def _adjust_color(color, blend_target=(1.0, 1.0, 1.0), strength=0.25):
            # Blend a color towards a target to create a related but distinguishable shade.
            return tuple((1 - strength) * c + strength * t for c, t in zip(color[:3], blend_target))

        feature_color_map = {feature: cmap(idx % cmap.N) for idx, feature in enumerate(unique_features)}

        if y == 'abs':
            y_col = "mean_abs_shap"
        elif y == 'high':
            y_col = "high_value_feature_impact"
        elif y == 'low':
            y_col = "low_value_feature_impact"
        elif y == 'high-low':
            y_col = ["high_value_feature_impact", "low_value_feature_impact"]
        else:
            raise ValueError(f"Invalid y value: {y}. Expected 'abs', 'high', 'low', or 'high-low'.")

        for feature in unique_features:
            feature_data = pd.DataFrame(plot_df[plot_df["feature_index"].astype(str) == feature]).sort_values("iteration")
            base_color = feature_color_map[feature]
            if y == 'high-low':
                plt.plot(
                    feature_data["iteration"], feature_data["high_value_feature_impact"],
                    alpha=0.6, color=_adjust_color(base_color, blend_target=(0.0, 0.0, 0.0), strength=0.15),
                    linestyle="-", label=f"{feature} (high)"
                )
                plt.plot(
                    feature_data["iteration"], feature_data["low_value_feature_impact"],
                    alpha=0.6, color=_adjust_color(base_color, blend_target=(1.0, 1.0, 1.0), strength=0.25),
                    linestyle="--", label=f"{feature} (low)"
                )
            else:
                plt.plot(feature_data["iteration"], feature_data[y_col], alpha=0.6, color=base_color, label=feature)

        plt.xlabel("Iteration")
        ylabel_map = {
            'abs': "Mean Absolute SHAP Value",
            'high': "High Value Feature Impact",
            'low': "Low Value Feature Impact",
            'high-low': "High-Low Value Feature Impact",
        }
        plt.ylabel(ylabel_map[y])
        plt.title(title or f'SHAP Value Trajectory - {iso3}' if iso3 else 'SHAP Value Trajectory', fontsize=titlefontsize)
        plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        if save:
            output_dir = generate_output_dir(self.path)
            output_file = output_dir / f"shap_history_{iso3 if iso3 else 'unknown'}.png"
            plt.savefig(output_file)
            print(f"SHAP history plot saved to {output_file}")

        plt.show()
