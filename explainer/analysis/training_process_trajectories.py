"""Main orchestrator for trajectory summarization and analysis."""

import json
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import shap
import pycountry
from sklearn.metrics import confusion_matrix

from .trajectory_utils import (
    round_float,
    generate_output_file,
    generate_output_dir,
    readable_target_name,
    substitute_feature_names,
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
        model: Optional[Any] = None,
        save: bool = False,
        path: str = '../data/results/',
        change_point_epsilon: Optional[float] = None,
        country_iso: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the summarizer with callbacks and configuration.

        Args:
            explain: ExplainerCallback containing training history
            classification: Classification instance with train/test data
            model: XGBoost model for trajectory metric computation
            save: Whether to save generated JSONs by default
            path: Output directory or file path for artifacts
            change_point_epsilon: Relative drift threshold for SHAP filtering
            country_iso: Country identifier (e.g. 'ETH', 'NGA', 'LKA')
            model_name: Name of the model variant
        """
        if change_point_epsilon is not None and change_point_epsilon < 0:
            raise ValueError("change_point_epsilon must be >= 0 when provided")

        self.explain = explain
        self.classification = classification
        self.model = model
        self.save = save
        self.path = path
        self.change_point_epsilon = change_point_epsilon
        self.country_iso = country_iso
        self.model_name = model_name
        self.feature_names = [str(c) for c in self.classification.train_test["X_train"].columns]
        # Initialize specialized analyzers
        self.tree_analyzer = TreeAnalyzer(classification)

        self.shap_analyzer = ShapAnalyzer(explain)
        self.shap_analyzer.set_change_point_epsilon(change_point_epsilon)
        self.shap_analyzer.configure(
            feature_names=self.feature_names,
            save=save,
            path=path,
            tree_analyzer=self.tree_analyzer,
            model=model,
        )

        self.feature_profiles_analyzer = FeatureProfilesAnalyzer()
        self.final_explainability_json: Optional[Dict[str, Any]] = None

        # Pre-computed SHAP DataFrame (also available on shap_analyzer.shap_history_df)
        self.shap_history_df = self.shap_analyzer.shap_history_df


    # === Output File Helpers ===

    def _generate_output_file(self, default_filename: str):
        """Returns a writable output file path."""
        return generate_output_file(self.path, default_filename)

    def _generate_output_dir(self):
        """Returns an output directory path."""
        return generate_output_dir(self.path)

    def _resolve_country_name(self) -> str:
        """Return a display country name when ISO metadata is available."""
        if not self.country_iso:
            return "the target country"
        country = pycountry.countries.get(alpha_3=self.country_iso)
        return country.name if country is not None else self.country_iso


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

    # === Context ===

    def generate_context_json(self, thresholds: Optional[dict] = None, save=None) -> Dict[str, Any]:
        """Generate context JSON with country and target variable information.

        Captures metadata about the analysis context including country identifier,
        target variable being predicted, and model information.

        Args:
            thresholds: Dictionary containing threshold values for target variables
            save: Override for default save setting

        Returns:
            Dictionary containing context metadata
        """

        readable_target = readable_target_name(self.classification.type_target)

        thresholds = thresholds or {
            'zn_ai': 10.2,
            'vita_rae_mcg': 490,
            'folate_mcg': 250,
            'vitb12_mcg': 2,
            'fe_mg': 22.4,
            'overall_mar': 0.75,
        }
        country = self.country_iso if self.country_iso else "the target country"

        context = {
            "schema_version": 1,
            "schema_type": "analysis_context",
            "country": self._resolve_country_name(),
            "model_name": self.model_name,
            "prediction_goal": (
                f"Predict whether household {readable_target} in {country} "
                f"is below the inadequacy threshold of {thresholds.get(self.classification.type_target, 'N/A')}."
            ),
            "features_count": len(self.feature_names),
        }

        if save or self.save:
            output_file = self._generate_output_file("trajectory_context.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(context, fp, ensure_ascii=False, separators=(',', ':'))

        return context



    # === SHAP Analysis ===

    def generate_final_explainability_json(
        self,
        model,
        predictions,
        probs=None,
        performance: Optional[Dict[str, Any]] = None,
        save=None,
        top_k_features: int = 15,
    ) -> Dict[str, Any]:
        """Generate a final-model explainability payload without trajectory history."""
        if top_k_features < 1:
            raise ValueError("top_k_features must be >= 1")

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        train_df = self.classification.train_test["X_train"]
        test_df = self.classification.train_test["Y_test"]
        train_target = self.classification.train_test["Y_train"]
        shap_array = self.shap_analyzer._normalize_binary_shap_values(self.classification.shap_values(estimator))

        feature_rows = []
        for feature_idx, feature_name in enumerate(train_df.columns):
            feature_values = train_df.iloc[:, feature_idx].to_numpy()
            shap_column = shap_array[:, feature_idx]
            median_value = float(np.median(feature_values))
            high_mask = feature_values >= median_value
            low_mask = feature_values < median_value

            high_impact = float(np.mean(shap_column[high_mask])) if np.any(high_mask) else float(np.mean(shap_column))
            low_impact = float(np.mean(shap_column[low_mask])) if np.any(low_mask) else float(np.mean(shap_column))

            feature_rows.append(
                {
                    "feature_name": str(feature_name),
                    "feature_index": int(feature_idx),
                    "mean_abs_shap": round_float(float(np.mean(np.abs(shap_column)))),
                    "mean_shap": round_float(float(np.mean(shap_column))),
                    "std_abs_shap": round_float(float(np.std(np.abs(shap_column)))),
                    "high_value_feature_impact": round_float(high_impact),
                    "low_value_feature_impact": round_float(low_impact),
                    "value_median": round_float(median_value),
                }
            )

        feature_rows_df = substitute_feature_names(
            pd.DataFrame(feature_rows),
            "feature_name",
            self.feature_names,
        )
        feature_rows = feature_rows_df.to_dict(orient="records")

        top_features = sorted(
            feature_rows,
            key=lambda item: float(item["mean_abs_shap"]),
            reverse=True,
        )[:top_k_features]

        raw_cm = confusion_matrix(test_df, predictions, normalize=None)
        normalized_cm = confusion_matrix(test_df, predictions, normalize="true")

        expected_value = getattr(shap.TreeExplainer(estimator), "expected_value", None)
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value_scalar = float(np.asarray(expected_value).reshape(-1)[-1])
        elif expected_value is None:
            expected_value_scalar = None
        else:
            expected_value_scalar = float(expected_value)

        payload = {
            "schema_version": 2,
            "summary_type": "final_model_explainability",
            "target": readable_target_name(self.classification.type_target),
            "country": self._resolve_country_name(),
            "country_iso": self.country_iso,
            "model": self.model_name,
            "counts": {
                "train": int(len(train_df)),
                "test": int(len(test_df)),
                "features": int(train_df.shape[1]),
            },
            "balance": {
                "train_pos_rate": round_float(float(train_target.mean().iloc[0])),
                "test_pos_rate": round_float(float(test_df.mean().iloc[0])),
            },
            "performance": performance if performance is not None else self.classification.perf_ind_classification(predictions, probs=probs),
            "confusion": {
                "counts": np.asarray(raw_cm).tolist(),
                "norm_true": np.round(np.asarray(normalized_cm), 4).tolist(),
            },
            "shap": {
                "base": round_float(expected_value_scalar) if expected_value_scalar is not None else None,
                "sample": int(shap_array.shape[0]),
                "mean_abs": round_float(float(np.mean(np.abs(shap_array)))),
                "feature_row_format": [
                    "feature_name",
                    "mean_abs_shap",
                    "mean_shap",
                    "std_abs_shap",
                    "high_value_feature_impact",
                    "low_value_feature_impact",
                    "value_median",
                ],
                "top_features": [
                    [
                        str(item["feature_name"]),
                        item["mean_abs_shap"],
                        item["mean_shap"],
                        item["std_abs_shap"],
                        item["high_value_feature_impact"],
                        item["low_value_feature_impact"],
                        item["value_median"],
                    ]
                    for item in top_features
                ],
            },
        }

        if self.save or save:
            output_file = self._generate_output_file("final_explainability.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(payload, fp, ensure_ascii=False, separators=(',', ':'))

        self.final_explainability_json = payload
        return payload

    def generate_shap_json(self, save=None, readable_feature_names: bool = True, compact: bool = True):
        """Delegates to ShapAnalyzer.generate_shap_json."""
        return self.shap_analyzer.generate_shap_json(save=save, readable_feature_names=readable_feature_names, compact=compact)

    def generate_shap_llm_summary_json(self, save=None, top_k_features: int = 10, milestone_count: int = 6, readable_feature_names: bool = True):
        """Delegates to ShapAnalyzer.generate_shap_llm_summary_json."""
        return self.shap_analyzer.generate_shap_llm_summary_json(save=save, top_k_features=top_k_features, milestone_count=milestone_count, readable_feature_names=readable_feature_names)

    def generate_shap_behavior_correlation_json(
        self,
        booster=None,
        final_shap_values=None,
        save=None,
        top_k_features: int = 15,
        confusion_metric: str = "hessian",
    ):
        """Delegates to ShapAnalyzer.generate_shap_behavior_correlation_json."""
        return self.shap_analyzer.generate_shap_behavior_correlation_json(
            booster=booster,
            final_shap_values=final_shap_values,
            save=save,
            top_k_features=top_k_features,
            confusion_metric=confusion_metric,
        )

    def generate_cohort_attribution_json(
        self,
        *,
        model=None,
        final_shap_values=None,
        save=None,
        min_support: int = 50,
        min_paths: int = 5,
        top_k: int = 20,
        top_local_shap: int = 8,
    ) -> Dict[str, Any]:
        """Generate cohort attribution artifact from repeated scope->decider contexts."""
        if not getattr(self.explain, "trees", None):
            return {"schema": "cohort_attribution_v1", "metadata": {"cohort_count": 0}, "cohorts": []}

        payload = self.tree_analyzer.analyze_subpopulation_cohorts(
            tree_entries=self.explain.trees,
            model=model if model is not None else self.model,
            final_shap_values=final_shap_values,
            min_support=min_support,
            min_paths=min_paths,
            top_k=top_k,
            top_local_shap=top_local_shap,
        )

        if self.save or save:
            output_file = self._generate_output_file("trajectory_cohort_attribution.json")
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(payload, fp, ensure_ascii=False, separators=(',', ':'))

        return payload

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

    def plot_shap_history(self, feature_names=None, iso3: Optional[str] = None, y: str = 'abs', title: Optional[str] = None, titlefontsize: int = 14, save: bool = False):
        """Delegates to ShapAnalyzer.plot_shap_history."""
        return self.shap_analyzer.plot_shap_history(feature_names=feature_names, iso3=iso3, y=y, title=title, titlefontsize=titlefontsize, save=save)
