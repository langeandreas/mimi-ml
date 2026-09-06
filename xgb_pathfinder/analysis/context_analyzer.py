"""
Context analyzer for xgb-pathfinder.

Analyzes feature role in tree decision paths:
- Early gatekeeper: Appears early in path, gates access to downstream paths
- Late decider: Appears late in path, makes final decision
- Scope→decider graphs: Visualize co-occurrence relationships
- Outlier specialists: High late-decider rate with strong impact
"""

from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

from ..config import DEFAULT_CONFIG
from .tree_engine import TreeEngine
from .trajectory_engine import TrajectoryEngine


class ContextAnalyzer:
    """
    Analyze decision context and feature positioning in tree paths.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
    feature_names : List[str]
        Human-readable feature names.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not specified).
    """
    
    def __init__(
        self,
        booster: xgb.Booster,
        feature_names: List[str],
        config: Optional[Dict] = None,
    ):
        """Initialize context analyzer."""
        self.booster = booster
        self.feature_names = feature_names
        self.config = config or DEFAULT_CONFIG
        self.tree_engine = TreeEngine(booster, feature_names)
    
    def analyze_feature_positioning(
        self,
        top_k: int = 10,
    ) -> Dict[str, Dict[str, any]]:
        """
        Analyze how features are positioned in tree paths.
        
        Returns metrics for each feature:
        - early_gatekeeper_rate: Fraction of paths where feature appears in first 30%
        - late_decider_rate: Fraction of paths where feature appears in last 30%
        - average_depth: Average depth of feature in paths where it appears
        - scope_features: Features that appear before this one
        - decider_features: Features that appear after this one
        
        Parameters
        ----------
        top_k : int
            Return only top K features (by frequency).
            
        Returns
        -------
        Dict[str, Dict]
            Maps feature name -> positioning metrics.
        """
        rules = self.tree_engine.extract_decision_rules_all_trees()
        
        # Aggregate statistics per feature
        feature_stats = {fname: {
            "total_appearances": 0,
            "early_gatekeeper_count": 0,
            "late_decider_count": 0,
            "depths": [],
            "scope_features": defaultdict(int),
            "decider_features": defaultdict(int),
        } for fname in self.feature_names}
        
        for rule in rules:
            conditions = rule["conditions"]
            if not conditions:
                continue
            
            path_length = len(conditions)
            early_threshold = int(path_length * 0.3)
            late_threshold = int(path_length * 0.7)
            
            for depth, condition in enumerate(conditions):
                feat_name = condition["feature_name"]
                
                if feat_name not in feature_stats:
                    continue
                
                feature_stats[feat_name]["total_appearances"] += 1
                feature_stats[feat_name]["depths"].append(depth)
                
                # Check if early gatekeeper
                if depth < early_threshold:
                    feature_stats[feat_name]["early_gatekeeper_count"] += 1
                
                # Check if late decider
                if depth >= late_threshold:
                    feature_stats[feat_name]["late_decider_count"] += 1
                
                # Track scope features (before) and decider features (after)
                for i in range(depth):
                    scope_feat = conditions[i]["feature_name"]
                    feature_stats[feat_name]["scope_features"][scope_feat] += 1
                
                for i in range(depth + 1, len(conditions)):
                    decider_feat = conditions[i]["feature_name"]
                    feature_stats[feat_name]["decider_features"][decider_feat] += 1
        
        # Compute percentages and build results
        results = {}
        
        for fname in self.feature_names:
            stats = feature_stats[fname]
            appearances = stats["total_appearances"]
            
            if appearances == 0:
                continue
            
            positioning = {
                "feature_name": fname,
                "total_appearances": appearances,
                "early_gatekeeper_rate": float(stats["early_gatekeeper_count"] / appearances),
                "late_decider_rate": float(stats["late_decider_count"] / appearances),
                "average_depth": float(np.mean(stats["depths"])) if stats["depths"] else 0.0,
                "max_depth": int(max(stats["depths"])) if stats["depths"] else 0,
                "top_scope_features": sorted(
                    stats["scope_features"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
                "top_decider_features": sorted(
                    stats["decider_features"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
            }
            
            results[fname] = positioning
        
        # Sort by appearances and return top K
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["total_appearances"],
            reverse=True,
        )
        
        return dict(sorted_results[:top_k])
    
    def build_scope_to_decider_graph(
        self,
        threshold: float = 0.01,
    ) -> Dict[str, any]:
        """
        Build scope→decider graph showing co-occurrence relationships.
        
        Returns nodes and edges for visualization.
        
        Parameters
        ----------
        threshold : float
            Minimum co-occurrence rate to include an edge (0-1).
            
        Returns
        -------
        Dict with keys:
        - nodes: List[{id, label, type ("scope" or "decider")}]
        - edges: List[{source, target, weight}]
        """
        positioning = self.analyze_feature_positioning()
        
        # Identify scope features (high early_gatekeeper_rate) and deciders (high late_decider_rate)
        scope_features = set()
        decider_features = set()
        
        for fname, stats in positioning.items():
            if stats["early_gatekeeper_rate"] > 0.3:
                scope_features.add(fname)
            if stats["late_decider_rate"] > 0.3:
                decider_features.add(fname)
        
        # If no clear split, use top K scope and decider by rate
        if not scope_features or not decider_features:
            sorted_by_early = sorted(
                positioning.items(),
                key=lambda x: x[1]["early_gatekeeper_rate"],
                reverse=True,
            )
            sorted_by_late = sorted(
                positioning.items(),
                key=lambda x: x[1]["late_decider_rate"],
                reverse=True,
            )
            
            scope_features = {fname for fname, _ in sorted_by_early[:5]}
            decider_features = {fname for fname, _ in sorted_by_late[:5]}
        
        # Build nodes
        nodes = []
        for fname in scope_features:
            nodes.append({
                "id": fname,
                "label": fname,
                "type": "scope",
            })
        for fname in decider_features:
            if fname not in scope_features:  # Avoid duplicates
                nodes.append({
                    "id": fname,
                    "label": fname,
                    "type": "decider",
                })
        
        # Build edges (scope → decider co-occurrences)
        edges = []
        rules = self.tree_engine.extract_decision_rules_all_trees()
        edge_weights = defaultdict(int)
        
        for rule in rules:
            conditions = rule["conditions"]
            
            for i, cond_i in enumerate(conditions):
                feat_i = cond_i["feature_name"]
                if feat_i not in scope_features:
                    continue
                
                for cond_j in conditions[i+1:]:
                    feat_j = cond_j["feature_name"]
                    if feat_j in decider_features:
                        edge_weights[(feat_i, feat_j)] += 1
        
        # Normalize edge weights and create edge list
        total_paths = len(rules)
        for (source, target), weight in edge_weights.items():
            rate = weight / total_paths
            if rate >= threshold:
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": float(rate),
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "scope_features": list(scope_features),
            "decider_features": list(decider_features),
        }
    
    def identify_outlier_specialists(
        self,
        late_decider_threshold: float = 0.6,
        impact_threshold: float = 0.5,
    ) -> List[Tuple[str, Dict[str, any]]]:
        """
        Identify "outlier specialist" features.
        
        These are features with:
        - High late_decider_rate (appear late in paths)
        - Strong impact when they do appear
        - Unique decision patterns
        
        Parameters
        ----------
        late_decider_threshold : float
            Min late_decider_rate (0-1).
        impact_threshold : float
            Min relative impact score (0-1).
            
        Returns
        -------
        List[Tuple[str, Dict]]
            List of (feature_name, metrics) for outlier specialists, sorted by score.
        """
        positioning = self.analyze_feature_positioning()
        
        specialists = []
        
        for fname, stats in positioning.items():
            late_rate = stats["late_decider_rate"]
            
            if late_rate < late_decider_threshold:
                continue
            
            # Compute "specialization score" = late_decider_rate * unique_paths
            n_unique_deciders = len(stats["top_decider_features"])
            uniqueness_score = min(1.0, n_unique_deciders / 3.0)  # Normalize
            
            specialist_score = late_rate * uniqueness_score
            
            if specialist_score >= impact_threshold:
                specialists.append((fname, {
                    "specialist_score": float(specialist_score),
                    "late_decider_rate": late_rate,
                    "uniqueness_score": float(uniqueness_score),
                    "n_unique_deciders": n_unique_deciders,
                    "top_decider_features": stats["top_decider_features"],
                }))
        
        # Sort by specialist_score
        specialists.sort(key=lambda x: x[1]["specialist_score"], reverse=True)
        
        return specialists
