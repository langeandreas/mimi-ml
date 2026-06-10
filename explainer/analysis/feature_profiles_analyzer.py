"""Feature profile generation optimized for LLM consumption."""

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from .trajectory_utils import round_float, substitute_feature_names


class FeatureProfilesAnalyzer:
    """Generates feature-centric profiles for LLM analysis."""

    @staticmethod
    def compute_feature_stability(trajectory_values: List[float]) -> str:
        """Classify stability of a feature's trajectory."""
        if len(trajectory_values) < 2:
            return "limited_data"
        
        std_dev = float(np.std(trajectory_values))
        mean_val = float(np.mean(trajectory_values))
        cv = std_dev / mean_val if mean_val != 0 else 0
        
        if cv < 0.1:
            return "very_stable"
        elif cv < 0.25:
            return "stable"
        elif cv < 0.5:
            return "moderate_variation"
        else:
            return "highly_variable"

    @staticmethod
    def interpret_decision_impact(high_impact: float, low_impact: float, feature_name: str) -> str:
        """Generate human-readable interpretation of feature's decision impact."""
        swing = abs(high_impact - low_impact)
        if swing < 0.05:
            return f"{feature_name} has minimal impact on predictions regardless of value"
        elif high_impact > 0 and low_impact < 0:
            return f"High {feature_name} → positive prediction; Low {feature_name} → negative prediction"
        elif high_impact > 0 and low_impact > 0:
            return f"{feature_name} generally increases positive prediction (high > low)"
        elif high_impact < 0 and low_impact < 0:
            return f"{feature_name} generally decreases positive prediction (high < low)"
        else:
            return f"{feature_name} has complex non-monotonic effect"

    @staticmethod
    def identify_stabilization_point(trajectory_sorted: List[Dict], window_size: int = 3) -> int:
        """Identify iteration after which a feature's importance stabilizes."""
        if len(trajectory_sorted) < window_size:
            return trajectory_sorted[-1]["iteration"] if trajectory_sorted else -1
        
        # Look for point where variance stabilizes
        shap_values = [x["mean_abs_shap"] for x in trajectory_sorted]
        min_std = float('inf')
        stabilization_idx = len(trajectory_sorted) - 1
        
        for i in range(len(trajectory_sorted) - window_size):
            window_std = float(np.std(shap_values[i:i+window_size]))
            if window_std < min_std:
                min_std = window_std
                stabilization_idx = i + window_size - 1
        
        return trajectory_sorted[stabilization_idx]["iteration"] if stabilization_idx < len(trajectory_sorted) else trajectory_sorted[-1]["iteration"]

    @staticmethod
    def extract_feature_interactions(tree_json: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract feature co-occurrence patterns from tree splits.
        
        Returns dict mapping each feature to a list of features it interacts with.
        """
        feature_interactions: Dict[str, set] = {}
        feature_names = tree_json.get("feature_names", [])
        
        for step in tree_json.get("iteration_progress", []):
            if step is None:
                continue
            split_features = [item["feature_index"] for item in step.get("top_split_features", [])]
            for i, feat1 in enumerate(split_features):
                if feat1 >= len(feature_names):
                    continue
                fname1 = str(feature_names[feat1])
                if fname1 not in feature_interactions:
                    feature_interactions[fname1] = set()
                for feat2 in split_features[i+1:]:
                    if feat2 >= len(feature_names):
                        continue
                    fname2 = str(feature_names[feat2])
                    feature_interactions[fname1].add(fname2)
        
        return {k: sorted(list(v)) for k, v in feature_interactions.items()}

    @staticmethod
    def build_feature_profiles(
        shap_json: Dict[str, Any],
        tree_json: Dict[str, Any],
        feature_names: List[str],
        top_k_features: int = 15,
        readable_feature_names: bool = True,
    ) -> Dict[str, Dict]:
        """Build feature-centric profiles from SHAP and tree data.
        
        Returns dictionary mapping feature names to their profiles.
        """
        # Build feature trajectories from SHAP
        feature_trajectories: Dict[str, List[Dict]] = {}
        for step in shap_json.get("iteration_progress", []):
            iteration = int(step["iteration"])
            for feature_info in step.get("top_shap_features", []):
                feat_idx = int(feature_info["feature_index"])
                if feat_idx >= len(feature_names):
                    continue
                    
                fname = str(feature_names[feat_idx])
                if readable_feature_names:
                    try:
                        names_df = pd.read_csv('data/features_explanations.csv')
                        if "codename" in names_df.columns and "explanation" in names_df.columns:
                            name_map = dict(zip(names_df["codename"].astype(str), names_df["explanation"].astype(str)))
                            fname = name_map.get(fname, fname)
                    except:
                        pass
                
                if fname not in feature_trajectories:
                    feature_trajectories[fname] = []
                
                feature_trajectories[fname].append({
                    "iteration": iteration,
                    "mean_abs_shap": float(feature_info.get("mean_abs_shap", 0)),
                    "high_value_impact": float(feature_info.get("high_value_feature_impact", 0)),
                    "low_value_impact": float(feature_info.get("low_value_feature_impact", 0)),
                })
        
        # Sort trajectories and compute statistics
        feature_profiles: Dict[str, Dict] = {}
        for fname, trajectory in feature_trajectories.items():
            trajectory_sorted = sorted(trajectory, key=lambda x: x["iteration"])
            shap_values = [x["mean_abs_shap"] for x in trajectory_sorted]
            
            first_iter = trajectory_sorted[0]
            last_iter = trajectory_sorted[-1]
            peak_iter = max(trajectory_sorted, key=lambda x: x["mean_abs_shap"])
            
            # Calculate statistics
            peak_value = peak_iter["mean_abs_shap"]
            start_value = first_iter["mean_abs_shap"]
            end_value = last_iter["mean_abs_shap"]
            delta = end_value - start_value
            stability = FeatureProfilesAnalyzer.compute_feature_stability(shap_values)
            
            feature_profiles[fname] = {
                "importance_trajectory": {
                    "start": round_float(start_value),
                    "end": round_float(end_value),
                    "peak": round_float(peak_value),
                    "peak_at_iteration": int(peak_iter["iteration"]),
                    "delta": round_float(delta),
                    "direction": "↑ increasing" if delta > 0.01 else ("↓ decreasing" if delta < -0.01 else "→ stable"),
                    "stability": stability,
                    "iteration_range": [int(first_iter["iteration"]), int(last_iter["iteration"])],
                },
                "decision_impact": {
                    "high_value_effect": round_float(last_iter["high_value_impact"]),
                    "low_value_effect": round_float(last_iter["low_value_impact"]),
                    "total_swing": round_float(
                        abs(last_iter["high_value_impact"] - last_iter["low_value_impact"])
                    ),
                    "interpretation": FeatureProfilesAnalyzer.interpret_decision_impact(
                        last_iter["high_value_impact"],
                        last_iter["low_value_impact"],
                        fname
                    ),
                },
                "emergence_pattern": {
                    "first_important_iteration": int(first_iter["iteration"]),
                    "reached_peak_at": int(peak_iter["iteration"]),
                    "iterations_to_peak": int(peak_iter["iteration"] - first_iter["iteration"]),
                    "stabilized_after": FeatureProfilesAnalyzer.identify_stabilization_point(trajectory_sorted),
                },
                "prevalence": {
                    "appearances_in_trajectory": len(trajectory_sorted),
                    "percentage_of_iterations": round_float(
                        100 * len(trajectory_sorted) / len(shap_json.get("iteration_progress", []))
                    ),
                },
            }
        
        # Extract tree-based relationships
        feature_interactions = FeatureProfilesAnalyzer.extract_feature_interactions(tree_json)
        for fname in feature_profiles:
            related = feature_interactions.get(fname, [])
            feature_profiles[fname]["related_features"] = related[:5]  # Top 5 interactions
        
        # Select top K features by peak importance
        top_features = sorted(
            [(fname, profile["importance_trajectory"]["peak"]) for fname, profile in feature_profiles.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k_features]
        
        return {fname: feature_profiles[fname] for fname, _ in top_features}
