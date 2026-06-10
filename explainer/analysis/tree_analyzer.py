"""Tree structure analysis and summarization."""

import json
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any, Optional

import numpy as np

from .trajectory_utils import round_float, TOP_K_TREE_FEATURES, TOP_K_INTERACTIONS


class TreeAnalyzer:
    """Analyzes tree structures and generates tree-based summaries."""

    def __init__(self, classification):
        """Initialize with classification object containing feature names."""
        self.classification = classification
        self.feature_names = [str(c) for c in classification.train_test["X_train"].columns]

    def summarize_tree_step(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Summarizes a single tree step by extracting split and leaf information.
        
        Args:
            entry: Dictionary containing tree information for a specific iteration
            
        Returns:
            Dictionary summarizing the tree step, or None if no tree found
        """
        tree_raw = entry.get("tree")
        if tree_raw is None:
            return None

        tree_obj = json.loads(tree_raw) if isinstance(tree_raw, str) else tree_raw
        split_counter = Counter()
        split_depth_sum = Counter()
        pair_counter = Counter()
        leaf_values = []
        
        feature_to_index = {name: i for i, name in enumerate(self.feature_names)}

        max_depth = self._walk_tree(
            tree_obj,
            0,
            [],
            split_counter,
            split_depth_sum,
            pair_counter,
            leaf_values,
            feature_to_index,
        )

        top_features = sorted(split_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_TREE_FEATURES]
        top_pairs = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_INTERACTIONS]

        if leaf_values:
            leaf_stats = {
                "mean": round_float(np.mean(leaf_values)),
                "std": round_float(np.std(leaf_values)),
                "min": round_float(np.min(leaf_values)),
                "max": round_float(np.max(leaf_values)),
            }
        else:
            leaf_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "iteration": int(entry["iteration"]),
            "split_count": int(sum(split_counter.values())),
            "leaf_count": int(len(leaf_values)),
            "max_depth": int(max_depth),
            "top_split_features": [
                {
                    "feature_index": int(fi),
                    "split_count": int(cnt),
                    "avg_split_depth": round_float(split_depth_sum[fi] / cnt),
                }
                for fi, cnt in top_features
            ],
            "top_feature_interactions": [
                {
                    "left_feature_index": int(a),
                    "right_feature_index": int(b),
                    "cooccurrence_count": int(cnt),
                }
                for (a, b), cnt in top_pairs
            ],
            "leaf_value_stats": leaf_stats,
        }

    def _walk_tree(
        self,
        node,
        depth: int,
        path: List[int],
        split_counter: Counter,
        split_depth_sum: Counter,
        pair_counter: Counter,
        leaf_values: List[float],
        feature_to_index: Dict[str, int],
    ) -> int:
        """Recursively walks through tree structure to extract split and leaf information.
        
        Returns maximum depth encountered in the tree.
        """
        if not isinstance(node, dict):
            return depth

        if "leaf" in node:
            leaf_values.append(float(node["leaf"]))
            path_unique = sorted(set(path))
            for left, right in combinations(path_unique, 2):
                pair_counter[(left, right)] += 1
            return depth

        split_name = str(node.get("split", ""))
        feature_idx = feature_to_index.get(split_name, -1)
        next_path = path

        if feature_idx >= 0:
            split_counter[feature_idx] += 1
            split_depth_sum[feature_idx] += depth
            next_path = path + [feature_idx]

        max_depth = depth
        for child in node.get("children", []):
            max_depth = max(
                max_depth,
                self._walk_tree(child, depth + 1, next_path, split_counter, split_depth_sum, pair_counter, leaf_values, feature_to_index),
            )

        return max_depth
