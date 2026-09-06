"""
Trajectory engine for xgb-pathfinder.

Computes per-feature trajectory metrics across training iterations:
- velocity: Feature gain at each iteration
- acceleration: Change in velocity
- FCI: Feature Confusion Index (cover / split_count)
- cumulative_gain: Running sum of gains
- HVI: Hessian-based volatility index
- Behavioral signatures: CORE_DRIVER, LATE_LEARNER, etc.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import json

from ..types import FeatureTrajectory, TrajectoryMetric
from ..config import DEFAULT_CONFIG, BEHAVIOR_SIGNATURE_RULES
from .utils import (
    get_trees_from_booster,
    compute_acceleration,
    classify_early_late_phases,
    feature_index_from_split,
)


class TrajectoryEngine:
    """
    Compute feature trajectory metrics across XGBoost training iterations.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster with training history.
    feature_names : List[str]
        Human-readable feature names.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not provided).
    """
    
    def __init__(
        self,
        booster: xgb.Booster,
        feature_names: List[str],
        config: Optional[Dict] = None,
    ):
        """Initialize trajectory engine."""
        self.booster = booster
        self.feature_names = feature_names
        self.config = config or DEFAULT_CONFIG
        self._trees = None
        self._feature_counts = {}  # Cache: feature_index -> count of appearances
    
    @property
    def trees(self) -> List[Dict]:
        """Lazy-loaded trees from booster."""
        if self._trees is None:
            self._trees = get_trees_from_booster(self.booster)
        return self._trees
    
    def compute_trajectories(self) -> Dict[str, FeatureTrajectory]:
        """
        Compute complete trajectory for all features.
        
        Returns
        -------
        Dict[str, FeatureTrajectory]
            Maps feature name -> trajectory dict with metrics per iteration.
            
        Examples
        --------
        >>> trajectories = engine.compute_trajectories()
        >>> for feat_name, traj in trajectories.items():
        ...     print(f"{feat_name}: peak velocity={traj['peak_velocity']:.4f}")
        """
        trajectories = {}
        
        # Initialize trajectory data per feature
        n_features = len(self.feature_names)
        feature_data = {i: {"velocity": [], "fci": [], "hessian": []} for i in range(n_features)}
        
        # Iterate through trees (iterations)
        for tree_idx, tree in enumerate(self.trees):
            tree_metrics = self._analyze_tree(tree)
            
            for feat_idx, metrics in tree_metrics.items():
                feature_data[feat_idx]["velocity"].append(metrics.get("gain", 0.0))
                feature_data[feat_idx]["fci"].append(metrics.get("fci", 0.0))
                feature_data[feat_idx]["hessian"].append(metrics.get("cover", 0.0))
        
        # Build trajectory objects
        for feat_idx in range(n_features):
            feat_name = self.feature_names[feat_idx]
            
            velocity = np.array(feature_data[feat_idx]["velocity"])
            acceleration = compute_acceleration(velocity)
            
            # Compute additional metrics
            cumulative_gain = np.cumsum(velocity)
            fci = np.array(feature_data[feat_idx]["fci"])
            hessian_mean = np.mean(feature_data[feat_idx]["hessian"]) if feature_data[feat_idx]["hessian"] else 0.0
            
            # Find peak iteration
            if len(velocity) > 0 and velocity.max() > 0:
                peak_iter = np.argmax(velocity)
                peak_vel = float(velocity[peak_iter])
            else:
                peak_iter = 0
                peak_vel = 0.0
            
            # Build metric list
            metrics_list = []
            for it in range(len(velocity)):
                metrics_list.append({
                    "iteration": it,
                    "velocity": float(velocity[it]),
                    "acceleration": float(acceleration[it]) if not np.isnan(acceleration[it]) else 0.0,
                    "fci": float(fci[it]) if it < len(fci) else 0.0,
                    "cumulative_gain": float(cumulative_gain[it]),
                    "hessian_mean": float(hessian_mean),
                })
            
            # Compute behavior signature
            behavior_sig = self._classify_behavior(velocity, acceleration)
            
            trajectory: FeatureTrajectory = {
                "feature_name": feat_name,
                "feature_index": feat_idx,
                "metrics": metrics_list,
                "peak_iteration": peak_iter,
                "peak_velocity": peak_vel,
                "total_gain": float(cumulative_gain[-1]) if len(cumulative_gain) > 0 else 0.0,
                "stability_class": behavior_sig,
                "behavior_signature": behavior_sig,
            }
            
            trajectories[feat_name] = trajectory
        
        return trajectories
    
    def _analyze_tree(self, tree: Dict) -> Dict[int, Dict[str, float]]:
        """
        Analyze a single tree and extract per-feature metrics.
        
        Parameters
        ----------
        tree : Dict
            Tree node from XGBoost JSON format.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Maps feature_index -> {gain, cover, fci, ...}
        """
        feature_metrics = {}
        split_count = {}  # Feature -> split count
        
        def traverse(node: Dict):
            if "leaf" in node:
                return
            
            feat_idx = feature_index_from_split(node.get("split"))
            gain = node.get("gain", 0.0)
            cover = node.get("cover", 0.0)
            
            if feat_idx is not None:
                if feat_idx not in feature_metrics:
                    feature_metrics[feat_idx] = {"gain": 0.0, "cover": 0.0, "split_count": 0}
                    split_count[feat_idx] = 0
                
                # Accumulate gain and cover
                feature_metrics[feat_idx]["gain"] += gain
                feature_metrics[feat_idx]["cover"] += cover
                split_count[feat_idx] += 1
            
            # Recursively process children
            if "children" in node:
                for child in node["children"]:
                    traverse(child)
        
        traverse(tree)
        
        # Compute FCI (Feature Confusion Index = cover / split_count)
        for feat_idx, metrics in feature_metrics.items():
            splits = split_count[feat_idx]
            metrics["fci"] = metrics["cover"] / splits if splits > 0 else 0.0
        
        return feature_metrics
    
    def _classify_behavior(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> str:
        """
        Classify feature behavior signature based on velocity/acceleration pattern.
        
        Parameters
        ----------
        velocity : np.ndarray
            Velocity series (gain per iteration).
        acceleration : np.ndarray
            Acceleration series (ΔGain per iteration).
            
        Returns
        -------
        str
            One of: CORE_DRIVER, LATE_LEARNER, OUTLIER_SPECIALIST, UNSTABLE, STABLE_BASELINE
        """
        if len(velocity) == 0:
            return "STABLE_BASELINE"
        
        # Classify early vs late phases
        early_indices, late_indices = classify_early_late_phases(len(velocity))
        
        # Get valid acceleration values (excluding NaN from first element)
        valid_accel = acceleration[~np.isnan(acceleration)]
        if len(valid_accel) == 0:
            return "STABLE_BASELINE"
        
        # Check for UNSTABLE: high std dev indicates oscillation
        if np.std(valid_accel) > self.config["stability_threshold"]:
            return "UNSTABLE"
        
        # Compute mean acceleration in early and late phases
        early_accel_vals = valid_accel[1:len(early_indices)] if len(early_indices) > 1 else []
        late_accel_vals = valid_accel[len(early_indices):] if len(late_indices) > 0 else []
        
        early_accel_mean = np.mean(early_accel_vals) if len(early_accel_vals) > 0 else 0.0
        late_accel_mean = np.mean(late_accel_vals) if len(late_accel_vals) > 0 else 0.0
        
        thres_early = self.config["early_acceleration_threshold"]
        thres_late = self.config["late_acceleration_threshold"]
        
        # Apply classification rules
        if early_accel_mean > thres_early and late_accel_mean < -thres_late:
            return "CORE_DRIVER"
        elif early_accel_mean < thres_early and late_accel_mean > thres_late:
            return "LATE_LEARNER"
        elif early_accel_mean > thres_early and late_accel_mean > thres_late:
            return "OUTLIER_SPECIALIST"
        else:
            return "STABLE_BASELINE"
    
    def get_peak_features(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top K features by peak velocity.
        
        Parameters
        ----------
        top_k : int
            How many top features to return.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (feature_name, peak_velocity) sorted descending.
        """
        trajectories = self.compute_trajectories()
        
        feature_peaks = [
            (feat_name, traj["peak_velocity"])
            for feat_name, traj in trajectories.items()
        ]
        
        feature_peaks.sort(key=lambda x: x[1], reverse=True)
        return feature_peaks[:top_k]
    
    def get_behavior_signatures(self) -> Dict[str, str]:
        """
        Get behavior signature for all features.
        
        Returns
        -------
        Dict[str, str]
            Maps feature name -> behavior signature.
        """
        trajectories = self.compute_trajectories()
        return {
            feat_name: traj["behavior_signature"]
            for feat_name, traj in trajectories.items()
        }
