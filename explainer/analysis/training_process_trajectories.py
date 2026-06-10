"""Main orchestrator for trajectory summarization and analysis."""

import json
from numbers import Integral
from typing import Optional, List, Union, Dict, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .trajectory_utils import (
    round_float,
    generate_output_file,
    generate_output_dir,
    substitute_feature_names,
    expand_shap_step,
    build_phase_summaries,
)
from .tree_analyzer import TreeAnalyzer
from .shap_analyzer import ShapAnalyzer
from .feature_profiles_analyzer import FeatureProfilesAnalyzer


class TrajectorySummarizer:
    """Orchestrates trajectory analysis using specialized analyzers.
    
    Coordinates tree analysis, SHAP analysis, and feature profile generation.
    """

    def __init__(
        self,
        explain,
        classification,
        save: bool = False,
        path: str = '../data/results/',
        change_point_epsilon: Optional[float] = None,
    ):
        """Initialize the summarizer with callbacks and configuration.
        
        Args:
            explain: ExplainerCallback containing training history
            classification: Classification instance with train/test data
            save: Whether to save generated JSONs by default
            path: Output directory or file path for artifacts
            change_point_epsilon: Relative drift threshold for SHAP filtering
        """
        if change_point_epsilon is not None and change_point_epsilon < 0:
            raise ValueError("change_point_epsilon must be >= 0 when provided")

        self.explain = explain
        self.classification = classification
        self.save = save
        self.path = path
        self.change_point_epsilon = change_point_epsilon
        self.feature_names = [str(c) for c in self.classification.train_test["X_train"].columns]

        # Initialize specialized analyzers
        self.tree_analyzer = TreeAnalyzer(classification)
        
        self.shap_analyzer = ShapAnalyzer(explain)
        self.shap_analyzer.set_change_point_epsilon(change_point_epsilon)
        
        self.feature_profiles_analyzer = FeatureProfilesAnalyzer()

        # Pre-compute SHAP dataframe for visualization
        self.shap_history_df = self.generate_shap_df()


    # === Output File Helpers ===

    def _generate_output_file(self, default_filename: str):
        """Returns a writable output file path."""
        return generate_output_file(self.path, default_filename)

    def _generate_output_dir(self):
        """Returns an output directory path."""
        return generate_output_dir(self.path)


    # === Tree Analysis ===

    def generate_tree_json(self, save=None):
        """Generates the tree summary JSON using TreeAnalyzer.
        
        Args:
            save: Override for default save setting
            readable_feature_names: Replace feature IDs with human-readable names
            
        Returns:
            Dictionary containing tree summary
        """
        progress = []
        for entry in self.explain.trees:
            tree_step = self.tree_analyzer.summarize_tree_step(entry)
            if tree_step is not None:
                progress.append(tree_step)
                
        tree_json = {
            "feature_names": self.feature_names,
            "phase_summaries": build_phase_summaries(progress),
            "iteration_progress": progress,
        }

        if save or self.save:
            output_file = self._generate_output_file("trajectory_tree.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(tree_json, fp, ensure_ascii=False, separators=(',', ':'))
        return tree_json



    # === SHAP Analysis ===

    def generate_shap_json(self, save=None, readable_feature_names: bool = True, compact: bool = True):
        """Generates the SHAP summary JSON using ShapAnalyzer.
        
        Args:
            save: Override for default save setting
            readable_feature_names: Replace feature IDs with human-readable names
            compact: Use compact format to reduce file size
            
        Returns:
            Dictionary containing SHAP summary
        """
        step_payloads = self.shap_analyzer.build_shap_step_payloads()
        retained_steps = self.shap_analyzer.filter_change_point_payloads(step_payloads)
        shap_steps = [self.shap_analyzer.summarize_shap_step(step["entry"]) for step in retained_steps]
        sample_size = self.shap_analyzer.get_sample_size()

        shap_json = {
            "shap_sample_size": sample_size,
            "iteration_progress": shap_steps,
        }

        if readable_feature_names:
            for step in shap_json["iteration_progress"]:
                top_features_df = pd.DataFrame(step.get("top_shap_features", []))
                top_features_df = substitute_feature_names(top_features_df, "feature_index", self.feature_names)
                step["top_shap_features"] = top_features_df.to_dict(orient="records")

        if compact:
            compact_steps = []
            for step in shap_json["iteration_progress"]:
                compact_steps.append(
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
                )

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
            output_file = self._generate_output_file("trajectory_shap.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(shap_json, fp, ensure_ascii=False, separators=(',', ':'))

        return shap_json


    def generate_shap_llm_summary_json(
        self,
        save=None,
        top_k_features: int = 10,
        milestone_count: int = 6,
        readable_feature_names: bool = True,
    ):
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
            summary = {
                "schema_version": 1,
                "summary_type": "shap_llm",
                "shap_sample_size": int(shap_json.get("shap_sample_size", 0)),
                "iteration_count": 0,
                "top_feature_trends": [],
                "milestones": [],
                "change_points": [],
            }
            if self.save or save:
                output_file = self._generate_output_file("trajectory_shap_llm_summary.json")
                with open(output_file, 'w', encoding='utf-8') as fp:
                    json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))
            return summary

        shap_steps = sorted(shap_steps, key=lambda step: int(step["iteration"]))

        feature_series = {}
        for step in shap_steps:
            iteration = int(step["iteration"])
            for item in step.get("top_shap_features", []):
                feature_index = str(item["feature_index"])
                feature_series.setdefault(feature_index, []).append(
                    {
                        "iteration": iteration,
                        "mean_abs_shap": float(item["mean_abs_shap"]),
                        "high": float(item["high_value_feature_impact"]),
                        "low": float(item["low_value_feature_impact"]),
                    }
                )

        ranked_features = sorted(
            feature_series.items(),
            key=lambda kv: max(point["mean_abs_shap"] for point in kv[1]),
            reverse=True,
        )[:top_k_features]

        top_feature_trends = []
        for feature_index, points in ranked_features:
            points_sorted = sorted(points, key=lambda point: point["iteration"])
            first = points_sorted[0]
            last = points_sorted[-1]
            peak = max(point["mean_abs_shap"] for point in points_sorted)
            top_feature_trends.append(
                {
                    "feature_index": feature_index,
                    "first_iteration": int(first["iteration"]),
                    "last_iteration": int(last["iteration"]),
                    "first_mean_abs_shap": round_float(first["mean_abs_shap"]),
                    "last_mean_abs_shap": round_float(last["mean_abs_shap"]),
                    "peak_mean_abs_shap": round_float(peak),
                    "delta_mean_abs_shap": round_float(last["mean_abs_shap"] - first["mean_abs_shap"]),
                    "last_high_value_impact": round_float(last["high"]),
                    "last_low_value_impact": round_float(last["low"]),
                }
            )

        total_steps = len(shap_steps)
        milestone_indexes = sorted(set(np.linspace(0, total_steps - 1, num=min(milestone_count, total_steps), dtype=int).tolist()))
        milestones = []
        for idx in milestone_indexes:
            step = shap_steps[idx]
            top_features = sorted(
                step.get("top_shap_features", []),
                key=lambda item: float(item["mean_abs_shap"]),
                reverse=True,
            )[:min(5, top_k_features)]

            milestones.append(
                {
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
                }
            )

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
                change_points.append(
                    {
                        "iteration": int(step["iteration"]),
                        "delta_shap_magnitude": round_float(delta),
                        "shap_magnitude": round_float(magnitude),
                        "top_feature": top_feature,
                    }
                )
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
                "max": round_float(max(float(step["shap_magnitude"]) for step in shap_steps)),
                "delta": round_float(float(shap_steps[-1]["shap_magnitude"]) - float(shap_steps[0]["shap_magnitude"])),
            },
            "top_feature_trends": top_feature_trends,
            "milestones": milestones,
            "change_points": change_points,
        }

        if self.save or save:
            output_file = self._generate_output_file("trajectory_shap_llm_summary.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))

        return summary


    # === Feature Profiles ===

    def generate_llm_optimized_feature_profiles(
        self,
        save=None,
        top_k_features: int = 15,
        readable_feature_names: bool = True,
    ) -> Dict[str, Any]:
        """Generate feature-centric profiles optimized for LLM consumption.
        
        Organizes SHAP and tree information by feature rather than iteration,
        enabling LLMs to reason about "what did each feature do?" directly.
        
        Args:
            save: Whether to save the output JSON
            top_k_features: Number of top features to include in profiles
            readable_feature_names: Whether to use human-readable feature names
            
        Returns:
            Dictionary with feature profiles organized for LLM analysis
        """
        shap_json = self.generate_shap_json(save=False, readable_feature_names=False, compact=False)
        tree_json = self.generate_tree_json(save=False)
        
        feature_profiles = FeatureProfilesAnalyzer.build_feature_profiles(
            shap_json=shap_json,
            tree_json=tree_json,
            feature_names=self.feature_names,
            top_k_features=top_k_features,
            readable_feature_names=readable_feature_names,
        )
        
        summary = {
            "schema": "feature_profiles_v1",
            "schema_description": "Feature-centric format organized by feature for LLM analysis",
            "summary_type": "feature_profiles",
            "metadata": {
                "top_k_features": top_k_features,
                "total_features_tracked": len(feature_profiles),
                "iterations_analyzed": len(shap_json.get("iteration_progress", [])),
                "shap_sample_size": shap_json.get("shap_sample_size", 0),
            },
            "features": feature_profiles,
        }
        
        if self.save or save:
            output_file = self._generate_output_file("trajectory_feature_profiles_llm.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, separators=(',', ':'))
        
        return summary

    # === Utilities ===
    
    def generate_shap_df(self):
        """Generates a DataFrame containing tail-based SHAP values.

        If change_point_epsilon is provided (> 0), only change-point iterations are kept
        based on relative drift in per-feature impact magnitude.

        Returns:
            A DataFrame containing SHAP impacts for retained iterations
        """
        step_payloads = self.shap_analyzer.build_shap_step_payloads()
        if not step_payloads:
            return pd.DataFrame(columns=["iteration", "feature_index", "mean_abs_shap", "high_value_feature_impact", "low_value_feature_impact"])

        retained_steps = self.shap_analyzer.filter_change_point_payloads(step_payloads)

        frames = []
        for step in retained_steps:
            top_idx = step["top_idx"]
            frames.append(pd.DataFrame({
                "iteration": step["iteration"],
                "feature_index": [self.feature_names[i] for i in top_idx],
                "mean_abs_shap": [round_float(step["impact_magnitude"][i]) for i in top_idx],
                "high_value_feature_impact": [round_float(step["high_impact"][i]) for i in top_idx],
                "low_value_feature_impact": [round_float(step["low_impact"][i]) for i in top_idx],
            }))

        if not frames:
            return pd.DataFrame(columns=["iteration", "feature_index", "mean_abs_shap", "high_value_feature_impact", "low_value_feature_impact"])

        shap_impact_df = pd.concat(frames, ignore_index=True)
        shap_impact_df = substitute_feature_names(shap_impact_df, "feature_index", self.feature_names)
        return shap_impact_df

    def plot_shap_history(self, feature_names: Optional[Union[List[str], int]] = None, iso3: Optional[str] = None, y:str = 'abs', title: Optional[str] = None, titlefontsize: int = 14, save: bool = False):
        """Plot SHAP trajectories with one line per feature over iterations."""
        if self.shap_history_df.empty:
            print("No SHAP history data available to plot.")
            return

        plot_df = self.shap_history_df.copy()
        if isinstance(feature_names, list) and feature_names:
            selected = {str(name) for name in feature_names}
            plot_df = plot_df[plot_df["feature_index"].astype(str).isin(selected)]
        elif isinstance(feature_names, Integral) and not isinstance(feature_names, bool) and feature_names:
            top_features = plot_df.groupby("feature_index")["mean_abs_shap"].max().nlargest(int(feature_names)).index.astype(str)
            plot_df = plot_df[plot_df["feature_index"].astype(str).isin(top_features)]

        if plot_df.empty:
            print("No SHAP history data available for the selected features.")
            return

        unique_features = sorted(plot_df["feature_index"].astype(str).unique())
        plt.figure(figsize=(12, len(unique_features) * 0.35 + 4))
        cmap = plt.get_cmap("tab20")

        def _adjust_color(color, blend_target=(1.0, 1.0, 1.0), strength=0.25):
            # Blend a color towards a target to create a related but distinguishable shade.
            return tuple((1 - strength) * c + strength * t for c, t in zip(color[:3], blend_target))

        feature_color_map = {
            feature: cmap(idx % cmap.N)
            for idx, feature in enumerate(unique_features)
        }

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
            feature_data = plot_df[plot_df["feature_index"].astype(str) == feature].sort_values("iteration")
            base_color = feature_color_map[feature]
            if y == 'high-low':
                plt.plot(
                    feature_data["iteration"],
                    feature_data["high_value_feature_impact"],
                    alpha=0.6,
                    color=_adjust_color(base_color, blend_target=(0.0, 0.0, 0.0), strength=0.15),
                    linestyle="-",
                    label=f"{feature} (high)"
                )
                plt.plot(
                    feature_data["iteration"],
                    feature_data["low_value_feature_impact"],
                    alpha=0.6,
                    color=_adjust_color(base_color, blend_target=(1.0, 1.0, 1.0), strength=0.25),
                    linestyle="--",
                    label=f"{feature} (low)"
                )
            else:
                plt.plot(
                    feature_data["iteration"],
                    feature_data[y_col],
                    alpha=0.6,
                    color=base_color,
                    label=feature
                )

        plt.xlabel("Iteration")
        if y == 'abs':
            plt.ylabel("Mean Absolute SHAP Value")
        elif y == 'high':
            plt.ylabel("High Value Feature Impact")
        elif y == 'low':
            plt.ylabel("Low Value Feature Impact")
        elif y == 'high-low':
            plt.ylabel("High-Low Value Feature Impact")
        plt.title(title or f'SHAP Value Trajectory - {iso3}' if iso3 else 'SHAP Value Trajectory', fontsize=titlefontsize)
        plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        if save:
            output_dir = self._generate_output_dir()
            output_file = output_dir / f"shap_history_{iso3 if iso3 else 'unknown'}.png"
            plt.savefig(output_file)
            print(f"SHAP history plot saved to {output_file}")

        plt.show()

    