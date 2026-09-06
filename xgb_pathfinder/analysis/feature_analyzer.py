"""
Feature analyzer for xgb-pathfinder.

Builds comprehensive per-feature profiles combining:
- Trajectory metrics (velocity, acceleration, etc.)
- SHAP decision impact (high/low tail effects)
- Feature interactions (co-occurrence patterns)
- Path statistics (late-decider rate, scope entropy)
- Behavioral signature (CORE_DRIVER, etc.)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

from ..types import FeatureProfile, FeatureInteraction, DecisionImpactProfile
from ..config import DEFAULT_CONFIG
from .trajectory_engine import TrajectoryEngine
from .shap_engine import ShapEngine
from .tree_engine import TreeEngine


class FeatureAnalyzer:
    """
    Build comprehensive feature profiles for all features in model.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target.
    feature_names : List[str]
        Human-readable feature names.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not specified).
    """
    
    def __init__(
        self,
        booster: xgb.Booster,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        config: Optional[Dict] = None,
    ):
        """Initialize feature analyzer."""
        self.booster = booster
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.feature_names = feature_names
        self.config = config or DEFAULT_CONFIG
        
        # Initialize sub-engines
        self.trajectory_engine = TrajectoryEngine(booster, feature_names, config)
        self.shap_engine = ShapEngine(booster, X_train, feature_names, config)
        self.tree_engine = TreeEngine(booster, feature_names)
    
    def build_profiles(self) -> Dict[str, FeatureProfile]:
        """
        Build comprehensive profiles for all features.
        
        Returns
        -------
        Dict[str, FeatureProfile]
            Maps feature name -> profile with all metrics.
            
        Examples
        --------
        >>> profiles = analyzer.build_profiles()
        >>> for feat_name, profile in profiles.items():
        ...     print(f"{feat_name}: {profile['behavior_signature']}, "
        ...           f"peak_iter={profile['importance_trajectory']['peak_iteration']}")
        """
        trajectories = self.trajectory_engine.compute_trajectories()
        impact_profiles = self.shap_engine.compute_impact_profiles()
        
        profiles = {}
        
        for feat_name, trajectory in trajectories.items():
            feat_idx = trajectory["feature_index"]
            
            # Get impact profile
            impact_profile = impact_profiles.get(
                feat_name,
                {
                    "mean_abs_shap": 0.0,
                    "high_value_effect": 0.0,
                    "low_value_effect": 0.0,
                    "total_swing": 0.0,
                    "interpretation": "No SHAP data",
                },
            )
            
            # Get feature interactions
            interactions = self._compute_feature_interactions(feat_idx)
            
            # Get path statistics
            path_stats = self._compute_path_statistics(feat_idx)
            
            profile: FeatureProfile = {
                "feature_name": feat_name,
                "feature_index": feat_idx,
                "importance_trajectory": trajectory,
                "decision_impact": impact_profile,
                "feature_interactions": interactions,
                "path_statistics": path_stats,
                "behavior_signature": trajectory["behavior_signature"],
            }
            
            profiles[feat_name] = profile
        
        return profiles
    
    def _compute_feature_interactions(
        self,
        feature_idx: int,
    ) -> List[FeatureInteraction]:
        """
        Compute co-occurrence patterns with other features in tree paths.
        
        Parameters
        ----------
        feature_idx : int
            Which feature to analyze.
            
        Returns
        -------
        List[FeatureInteraction]
            Top co-occurring features, sorted by co-occurrence rate.
        """
        co_occurrence = defaultdict(int)
        occurrence_count = 0
        
        # Walk all trees and count co-occurrences
        rules = self.tree_engine.extract_decision_rules_all_trees()
        
        for rule in rules:
            condition_features = {cond["feature_index"] for cond in rule["conditions"]}
            
            if feature_idx in condition_features:
                occurrence_count += 1
                
                # Count co-occurrences with other features
                for other_feat_idx in condition_features:
                    if other_feat_idx != feature_idx:
                        co_occurrence[other_feat_idx] += 1
        
        # Convert to FeatureInteraction objects
        interactions = []
        
        if occurrence_count > 0:
            for co_feat_idx, co_count in sorted(
                co_occurrence.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                co_feat_name = (
                    self.feature_names[co_feat_idx]
                    if co_feat_idx < len(self.feature_names)
                    else f"feature_{co_feat_idx}"
                )
                
                interaction: FeatureInteraction = {
                    "co_feature_name": co_feat_name,
                    "co_feature_index": co_feat_idx,
                    "co_occurrence_count": int(co_count),
                    "co_occurrence_rate": float(co_count / occurrence_count),
                }
                interactions.append(interaction)
        
        # Return top K
        return interactions[:self.config["top_k_interactions"]]
    
    def _compute_path_statistics(self, feature_idx: int) -> Dict[str, float]:
        """
        Compute path-level statistics for a feature.
        
        Parameters
        ----------
        feature_idx : int
            Which feature to analyze.
            
        Returns
        -------
        Dict[str, float]
            Statistics: late_decider_rate, scope_entropy, mean_leaf_value_when_late, etc.
        """
        rules = self.tree_engine.extract_decision_rules_all_trees()
        
        total_paths = len(rules)
        late_decider_count = 0
        leaf_values_when_late = []
        
        for rule in rules:
            conditions = rule["conditions"]
            
            # Check if this feature appears in the rule
            feature_in_rule = any(c["feature_index"] == feature_idx for c in conditions)
            
            if not feature_in_rule:
                continue
            
            # Find position of this feature in the path
            feature_position = next(
                (i for i, c in enumerate(conditions) if c["feature_index"] == feature_idx),
                None,
            )
            
            if feature_position is not None:
                # Check if it's late (in the last 30% of path)
                late_threshold = len(conditions) * 0.7
                if feature_position >= late_threshold:
                    late_decider_count += 1
                    leaf_values_when_late.append(rule["leaf_value"])
        
        late_decider_rate = (late_decider_count / total_paths) if total_paths > 0 else 0.0
        mean_leaf_value_when_late = (
            float(np.mean(leaf_values_when_late))
            if leaf_values_when_late
            else 0.0
        )
        
        return {
            "late_decider_rate": float(late_decider_rate),
            "late_decider_count": int(late_decider_count),
            "mean_leaf_value_when_late": mean_leaf_value_when_late,
            "total_path_occurrences": int(total_paths),
            "scope_entropy": self._compute_scope_entropy(feature_idx),
        }
    
    def _compute_scope_entropy(self, feature_idx: int) -> float:
        """
        Compute entropy of scope conditions when feature appears.
        
        Measures diversity of conditions that co-occur with this feature.
        """
        rules = self.tree_engine.extract_decision_rules_all_trees()
        scope_patterns = []
        for rule in rules:
            conditions = rule["conditions"]
            for position, condition in enumerate(conditions):
                if condition["feature_index"] == feature_idx:
                    scope_patterns.append(tuple(
                        item["feature_index"] for item in conditions[:position]
                    ))
                    break
        if not scope_patterns:
            return 0.0
        counts = pd.Series(scope_patterns).value_counts().to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities)))
    
    def get_top_features_by_shap(self, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get top K features by mean absolute SHAP value.
        
        Parameters
        ----------
        top_k : int, optional
            How many features to return. Uses config if not specified.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (feature_name, mean_abs_shap) sorted descending.
        """
        if top_k is None:
            top_k = self.config["top_k_features"]
        
        return self.shap_engine.get_feature_importance_ranking(top_k)
