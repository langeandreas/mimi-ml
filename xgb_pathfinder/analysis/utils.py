"""
Shared utilities for analysis engines.

Includes:
- Tree traversal primitives
- Rule evaluation helpers
- Metric normalization functions
- Type conversions
"""

from typing import List, Dict, Tuple, Any, Callable, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import re


# ============================================================================
# Tree Traversal Utilities
# ============================================================================

def feature_index_from_split(split: Any) -> Optional[int]:
    """Return an integer feature index from XGBoost's split representation."""
    if isinstance(split, (int, np.integer)):
        return int(split)
    if isinstance(split, str):
        match = re.fullmatch(r"f(\d+)", split)
        if match:
            return int(match.group(1))
    return None

def get_trees_from_booster(booster: xgb.Booster) -> List[Dict[str, Any]]:
    """
    Extract trees from XGBoost booster as dictionaries.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
        
    Returns
    -------
    List[Dict[str, Any]]
        Each element is a tree node dictionary from XGBoost's tree structure.
    """
    # Convert trees to JSON format and back (standard XGBoost method)
    tree_json_list = booster.get_dump(with_stats=True, dump_format="json")
    import json
    trees = [json.loads(tree_str) for tree_str in tree_json_list]
    return trees


def traverse_tree_dfs(
    tree_node: Dict[str, Any],
    path: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Traverse tree via depth-first search, yielding root-to-leaf paths.
    
    Parameters
    ----------
    tree_node : Dict[str, Any]
        Current node in tree (from XGBoost JSON format).
    path : List[Dict[str, Any]], optional
        Accumulated path from root to current node.
    callback : Callable, optional
        Function to call on each leaf node. Receives path as argument.
        
    Returns
    -------
    List[List[Dict[str, Any]]]
        All root-to-leaf paths in this tree.
        Each path is a list of node dictionaries.
        
    Examples
    --------
    >>> tree = booster.get_dump(dump_format="json")[0]
    >>> paths = traverse_tree_dfs(json.loads(tree))
    >>> print(f"Found {len(paths)} paths in tree")
    """
    if path is None:
        path = []
    
    # Add current node to path
    current_path = path + [tree_node]
    
    # Check if leaf node
    if "leaf" in tree_node:
        if callback:
            callback(current_path)
        return [current_path]
    
    # Recursively traverse children
    all_paths = []
    
    if "children" in tree_node:
        for child_node in tree_node["children"]:
            paths = traverse_tree_dfs(child_node, current_path, callback)
            all_paths.extend(paths)
    else:
        # Leaf node (some XGBoost formats may not have explicit 'leaf' field)
        if callback:
            callback(current_path)
        all_paths.append(current_path)
    
    return all_paths


def extract_split_conditions_from_path(
    path: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract split conditions (feature, threshold, direction) from a path.
    
    Parameters
    ----------
    path : List[Dict[str, Any]]
        Root-to-leaf path in tree.
        
    Returns
    -------
    List[Dict[str, Any]]
        Split conditions (excluding leaf node).
        Each dict has keys: "split", "split_condition", "yes", "no", "depth".
        
    Examples
    --------
    >>> conditions = extract_split_conditions_from_path(path)
    >>> for cond in conditions:
    ...     print(f"Feature {cond['split']} < {cond['split_condition']}")
    """
    conditions = []
    
    # Process all but last node (last is leaf)
    for node in path[:-1]:
        if "split" in node:  # Not a leaf
            conditions.append(node)
    
    return conditions


# ============================================================================
# Metric Normalization & Aggregation
# ============================================================================

def normalize_metric(
    values: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize metric values to [0, 1] range.
    
    Parameters
    ----------
    values : np.ndarray
        Values to normalize.
    method : str
        "minmax" (default), "zscore", or "softmax".
        
    Returns
    -------
    np.ndarray
        Normalized values.
    """
    values = np.asarray(values, dtype=float)
    
    if method == "minmax":
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            return np.ones_like(values) * 0.5
        return (values - vmin) / (vmax - vmin)
    
    elif method == "zscore":
        mean, std = values.mean(), values.std()
        if std == 0:
            return np.zeros_like(values)
        return (values - mean) / std
    
    elif method == "softmax":
        e_x = np.exp(values - values.max())
        return e_x / e_x.sum()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_percentiles(
    values: np.ndarray,
    percentiles: List[float] = [25, 50, 75],
) -> Dict[float, float]:
    """
    Compute percentiles of values.
    
    Parameters
    ----------
    values : np.ndarray
        Values to analyze.
    percentiles : List[float]
        Which percentiles to compute (0-100).
        
    Returns
    -------
    Dict[float, float]
        Maps percentile -> value.
    """
    result = {}
    for p in percentiles:
        result[p] = np.percentile(values, p)
    return result


def robust_mean(values: np.ndarray, trim: float = 0.05) -> float:
    """
    Compute trimmed mean (robust to outliers).
    
    Parameters
    ----------
    values : np.ndarray
        Values to average.
    trim : float
        Proportion to trim from each end (default 5%).
        
    Returns
    -------
    float
        Trimmed mean.
    """
    from scipy import stats
    return stats.trim_mean(values, trim)


# ============================================================================
# Type Conversion Utilities
# ============================================================================

def ensure_array(data: Any, dtype: type = float) -> np.ndarray:
    """
    Convert input to numpy array with specified dtype.
    
    Parameters
    ----------
    data : Any
        Input data (list, array, scalar, etc.)
    dtype : type
        Target dtype.
        
    Returns
    -------
    np.ndarray
    """
    return np.asarray(data, dtype=dtype)


def ensure_dataframe(data: Any, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert input to pandas DataFrame.
    
    Parameters
    ----------
    data : Any
        Input data (array, dict, DataFrame, etc.)
    columns : List[str], optional
        Column names if creating from array.
        
    Returns
    -------
    pd.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    if isinstance(data, dict):
        return pd.DataFrame(data)
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if columns is None:
        columns = [f"col_{i}" for i in range(data.shape[1])]
    
    return pd.DataFrame(data, columns=columns)


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_acceleration(
    velocity_series: np.ndarray,
    window: int = 1,
) -> np.ndarray:
    """
    Compute acceleration (change in velocity) from a velocity time series.
    
    Parameters
    ----------
    velocity_series : np.ndarray
        Velocity values over time (iterations).
    window : int
        Look-back window for computing difference (default 1 for ΔV).
        
    Returns
    -------
    np.ndarray
        Acceleration values. First `window` values are set to NaN.
        
    Examples
    --------
    >>> velocity = np.array([0.1, 0.15, 0.2, 0.18, 0.15])
    >>> accel = compute_acceleration(velocity)
    >>> print(accel)  # [nan, 0.05, 0.05, -0.02, -0.03]
    """
    velocity_series = np.asarray(velocity_series, dtype=float)
    accel = np.full_like(velocity_series, np.nan)
    
    for i in range(window, len(velocity_series)):
        accel[i] = velocity_series[i] - velocity_series[i - window]
    
    return accel


def classify_early_late_phases(
    series_length: int,
    early_ratio: float = 0.3,
) -> Tuple[range, range]:
    """
    Partition training iterations into early and late phases.
    
    Parameters
    ----------
    series_length : int
        Total number of iterations.
    early_ratio : float
        Proportion of iterations considered "early" (default 30%).
        
    Returns
    -------
    Tuple[range, range]
        (early_indices, late_indices) as Python ranges.
        
    Examples
    --------
    >>> early, late = classify_early_late_phases(100, early_ratio=0.3)
    >>> print(f"Early: {list(early)}, Late: {list(late)}")
    """
    early_cutoff = max(1, int(series_length * early_ratio))
    early = range(0, early_cutoff)
    late = range(early_cutoff, series_length)
    return early, late


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_booster(booster: xgb.Booster) -> bool:
    """Check if booster is a valid trained XGBoost model."""
    try:
        trees = booster.get_dump()
        return len(trees) > 0
    except Exception:
        return False


def validate_feature_names(feature_names: List[str], n_features: int) -> bool:
    """Check if feature names list is valid."""
    return len(feature_names) == n_features and all(isinstance(name, str) for name in feature_names)
