"""
Tree analysis engine for xgb-pathfinder.

Responsibilities:
- Extract and summarize tree structures
- Analyze split patterns and feature usage
- Extract decision rules from tree paths
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import json

from .utils import (
    get_trees_from_booster,
    traverse_tree_dfs,
    extract_split_conditions_from_path,
    feature_index_from_split,
)


class TreeEngine:
    """
    Analyze tree structures in XGBoost boosting models.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
    feature_names : List[str]
        Human-readable feature names.
    """
    
    def __init__(self, booster: xgb.Booster, feature_names: List[str]):
        """Initialize tree engine."""
        self.booster = booster
        self.feature_names = feature_names
        self._trees = None
        self._tree_dump_str = None
    
    @property
    def trees(self) -> List[Dict[str, Any]]:
        """Lazy-loaded trees from booster."""
        if self._trees is None:
            self._trees = get_trees_from_booster(self.booster)
        return self._trees
    
    def get_tree_count(self) -> int:
        """Return number of trees in booster."""
        return len(self.trees)
    
    def summarize_tree(self, tree_index: int) -> Dict[str, Any]:
        """
        Summarize structure of a single tree.
        
        Parameters
        ----------
        tree_index : int
            Which tree to summarize (0-indexed).
            
        Returns
        -------
        Dict[str, Any]
            Keys: split_count, leaf_count, max_depth, avg_depth, feature_usage, leaf_value_stats
        """
        if tree_index < 0 or tree_index >= len(self.trees):
            raise ValueError(f"Invalid tree index {tree_index}")
        
        tree = self.trees[tree_index]
        
        # Collect statistics
        split_count = 0
        leaf_count = 0
        depths = []
        feature_usage = {}
        leaf_values = []
        
        def count_nodes(node: Dict[str, Any], depth: int = 0):
            nonlocal split_count, leaf_count
            
            if "leaf" in node:
                leaf_count += 1
                depths.append(depth)
                leaf_values.append(node["leaf"])
            else:
                split_count += 1
                feature_idx = feature_index_from_split(node.get("split"))
                if feature_idx is not None:
                    feature_usage[feature_idx] = feature_usage.get(feature_idx, 0) + 1
                
                if "children" in node:
                    for child in node["children"]:
                        count_nodes(child, depth + 1)
        
        count_nodes(tree)
        
        # Map feature indices to names
        feature_usage_named = {}
        for feat_idx, count in feature_usage.items():
            if feat_idx < len(self.feature_names):
                feature_usage_named[self.feature_names[feat_idx]] = count
            else:
                feature_usage_named[f"feature_{feat_idx}"] = count
        
        return {
            "tree_index": tree_index,
            "split_count": split_count,
            "leaf_count": leaf_count,
            "max_depth": max(depths) if depths else 0,
            "avg_depth": np.mean(depths) if depths else 0.0,
            "feature_usage": feature_usage_named,
            "leaf_value_stats": {
                "mean": np.mean(leaf_values) if leaf_values else 0.0,
                "std": np.std(leaf_values) if leaf_values else 0.0,
                "min": min(leaf_values) if leaf_values else 0.0,
                "max": max(leaf_values) if leaf_values else 0.0,
            },
        }
    
    def summarize_all_trees(self) -> List[Dict[str, Any]]:
        """Summarize all trees in booster."""
        return [self.summarize_tree(i) for i in range(self.get_tree_count())]
    
    def get_feature_split_count_per_tree(self) -> pd.DataFrame:
        """
        Count how many times each feature is used as a split in each tree.
        
        Returns
        -------
        pd.DataFrame
            Rows: feature names, Columns: tree indices, Values: split count.
        """
        tree_count = self.get_tree_count()
        feature_split_counts = {fname: [0] * tree_count for fname in self.feature_names}
        
        for tree_idx, summary in enumerate(self.summarize_all_trees()):
            for feat_name, count in summary["feature_usage"].items():
                if feat_name in feature_split_counts:
                    feature_split_counts[feat_name][tree_idx] = count
        
        return pd.DataFrame(feature_split_counts).T
    
    def extract_all_paths(self, tree_index: int) -> List[List[Dict[str, Any]]]:
        """
        Extract all root-to-leaf paths from a tree.
        
        Parameters
        ----------
        tree_index : int
            Which tree to extract paths from.
            
        Returns
        -------
        List[List[Dict[str, Any]]]
            Each element is a path (list of nodes from root to leaf).
        """
        tree = self.trees[tree_index]
        paths = traverse_tree_dfs(tree)
        return paths
    
    def extract_all_rules(self, tree_index: int) -> List[Dict[str, Any]]:
        """
        Extract decision rules (paths) from a tree as human-readable rules.
        
        Parameters
        ----------
        tree_index : int
            Which tree to extract rules from.
            
        Returns
        -------
        List[Dict[str, Any]]
            Each rule has:
            - conditions: List of (feature_name, threshold, operator, direction)
            - leaf_value: Prediction at leaf
            - tree_index: This tree's index
            - rule_id: Unique identifier
        """
        paths = self.extract_all_paths(tree_index)
        rules = []
        
        for path_idx, path in enumerate(paths):
            # Extract conditions from non-leaf nodes
            conditions = []
            
            for path_position, node in enumerate(path[:-1]):
                feature_idx = feature_index_from_split(node.get("split"))
                threshold = node.get("split_condition")
                
                if feature_idx is not None and threshold is not None:
                    feature_name = (
                        self.feature_names[feature_idx]
                        if feature_idx < len(self.feature_names)
                        else f"feature_{feature_idx}"
                    )
                    
                    next_node = path[path_position + 1]
                    next_node_id = str(next_node.get("nodeid"))
                    operator = "<" if next_node_id == str(node.get("yes")) else ">="
                    conditions.append({
                        "feature_name": feature_name,
                        "feature_index": feature_idx,
                        "threshold": threshold,
                        "operator": operator,
                    })
            
            # Extract leaf value
            leaf_node = path[-1]
            leaf_value = leaf_node.get("leaf", 0.0)
            
            rule_id = f"tree_{tree_index}_path_{path_idx}"
            
            rules.append({
                "rule_id": rule_id,
                "tree_index": tree_index,
                "conditions": conditions,
                "leaf_value": float(leaf_value),
                "path_length": len(path) - 1,  # Exclude leaf
            })
        
        return rules
    
    def extract_decision_rules_all_trees(self) -> List[Dict[str, Any]]:
        """
        Extract decision rules from all trees in booster.
        
        Returns
        -------
        List[Dict[str, Any]]
            All rules from all trees.
        """
        all_rules = []
        for tree_idx in range(self.get_tree_count()):
            rules = self.extract_all_rules(tree_idx)
            all_rules.extend(rules)
        return all_rules
    
    def get_top_split_features(self, top_k: int = 10) -> Dict[str, int]:
        """
        Get top K features by total split count across all trees.
        
        Parameters
        ----------
        top_k : int
            How many top features to return.
            
        Returns
        -------
        Dict[str, int]
            Maps feature name -> total split count across all trees.
        """
        feature_split_counts = {fname: 0 for fname in self.feature_names}
        
        for summary in self.summarize_all_trees():
            for feat_name, count in summary["feature_usage"].items():
                feature_split_counts[feat_name] = feature_split_counts.get(feat_name, 0) + count
        
        # Sort and return top K
        sorted_features = sorted(
            feature_split_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return dict(sorted_features[:top_k])
