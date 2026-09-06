"""Low-level, stateless tree-traversal utilities.

These are pure functions that operate on a raw XGBoost JSON node dict.
They are shared by TreeAnalyzer (summarize_tree_step) and
DecisionContextAnalyzer (analyze_feature_decision_contexts).
"""

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple


def walk_tree(
    node: Any,
    depth: int,
    path: List[int],
    split_counter: Counter,
    split_depth_sum: Counter,
    pair_counter: Counter,
    leaf_values: List[float],
    feature_to_index: Dict[str, int],
) -> int:
    """Recursively walk a tree node, collecting split and leaf statistics.

    Accumulates in-place into the counters / list passed in, then returns
    the maximum depth encountered in the subtree rooted at *node*.

    Args:
        node: Current tree node (dict) or terminal value.
        depth: Depth of *node* relative to the tree root.
        path: Feature indices on the root-to-*node* path (used for pair counting).
        split_counter: Counts how many times each feature index is used as a split.
        split_depth_sum: Accumulates the sum of depths at which each feature splits
                         (used to compute average split depth).
        pair_counter: Counts co-occurrence pairs of features along root-to-leaf paths.
        leaf_values: Collects every leaf value encountered.
        feature_to_index: Mapping of feature name -> feature index.

    Returns:
        Maximum depth reached in the subtree.
    """
    if not isinstance(node, dict):
        return depth

    if "leaf" in node:
        leaf_values.append(float(node["leaf"]))
        # Count every unique feature pair present on this root-to-leaf path.
        for left, right in combinations(sorted(set(path)), 2):
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
            walk_tree(
                child,
                depth + 1,
                next_path,
                split_counter,
                split_depth_sum,
                pair_counter,
                leaf_values,
                feature_to_index,
            ),
        )
    return max_depth


def collect_split_path_records(
    node: Any,
    depth: int,
    current_path: List[Tuple[int, int]],
    path_records: List[Dict[str, Any]],
    feature_to_index: Dict[str, int],
) -> None:
    """Recursively collect every root-to-leaf path as a (feature_idx, depth) sequence.

    Each completed path is appended to *path_records* together with the
    leaf value at its terminus.

    Args:
        node: Current tree node (dict) or terminal value.
        depth: Depth of *node* relative to the tree root.
        current_path: Accumulated list of (feature_idx, depth) tuples so far.
        path_records: Output list; each entry is
                      ``{'path': [...], 'leaf_value': float}``.
        feature_to_index: Mapping of feature name -> feature index.
    """
    if not isinstance(node, dict):
        return

    if "leaf" in node:
        path_records.append(
            {
                "path": list(current_path),
                "leaf_value": float(node.get("leaf", 0.0)),
            }
        )
        return

    split_name = str(node.get("split", ""))
    feature_idx = feature_to_index.get(split_name, -1)

    next_path = current_path
    if feature_idx >= 0:
        next_path = current_path + [(int(feature_idx), int(depth))]

    children = node.get("children", [])
    if not children:
        # Internal node with no children — treat as degenerate leaf.
        path_records.append({"path": list(next_path), "leaf_value": 0.0})
        return

    for child in children:
        collect_split_path_records(
            node=child,
            depth=depth + 1,
            current_path=next_path,
            path_records=path_records,
            feature_to_index=feature_to_index,
        )
