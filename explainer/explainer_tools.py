"""Tooling helpers so trajectory agents can inspect JSON artifacts directly."""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.tools import tool


def _tokenize_path(path: str) -> List[str]:
    """Tokenize dotted/indexed paths like a.b[0].c into ['a','b','0','c']."""
    cleaned = path.strip()
    if not cleaned:
        return []

    tokens: List[str] = []
    current = []
    i = 0
    while i < len(cleaned):
        char = cleaned[i]
        if char == ".":
            if current:
                tokens.append("".join(current))
                current = []
            i += 1
            continue
        if char == "[":
            if current:
                tokens.append("".join(current))
                current = []
            end = cleaned.find("]", i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in path: {path}")
            bracket_value = cleaned[i + 1 : end].strip()
            if bracket_value:
                tokens.append(bracket_value)
            i = end + 1
            continue

        current.append(char)
        i += 1

    if current:
        tokens.append("".join(current))
    return tokens


def _resolve_path(payload: Any, path: str) -> Any:
    """Resolve a simple dotted/indexed path against a JSON-like payload."""
    value = payload
    for token in _tokenize_path(path):
        if isinstance(value, dict):
            if token not in value:
                raise KeyError(f"Key '{token}' not found")
            value = value[token]
            continue
        if isinstance(value, list):
            idx = int(token)
            value = value[idx]
            continue
        raise KeyError(f"Cannot descend into type {type(value).__name__} using token '{token}'")
    return value


def _iter_paths(payload: Any) -> Sequence[Tuple[str, Any]]:
    """Breadth-first iteration of paths and values."""
    results: List[Tuple[str, Any]] = []
    queue = deque([("", payload)])
    while queue:
        path, node = queue.popleft()
        results.append((path, node))
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                queue.append((child_path, value))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                child_path = f"{path}[{idx}]" if path else f"[{idx}]"
                queue.append((child_path, value))
    return results


def _summarize_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)[:5000]
    return str(value)


def build_trajectory_tools(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Build tool set bound to specific SHAP/tree payloads."""

    @tool
    def get_shap_json_at_path(path: str) -> str:
        """Return SHAP JSON value at a dotted/index path (example: iteration_progress[0].i)."""
        try:
            value = _resolve_path(shap_json, path)
            return _summarize_scalar(value)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            return f"PATH_ERROR: {exc}"

    @tool
    def get_tree_json_at_path(path: str) -> str:
        """Return tree JSON value at a dotted/index path (example: depth_stats.max_depth)."""
        try:
            value = _resolve_path(tree_json, path)
            return _summarize_scalar(value)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            return f"PATH_ERROR: {exc}"

    @tool
    def search_shap_json_keys(keyword: str, max_results: int = 25) -> str:
        """Find SHAP JSON paths whose key/path text contains a keyword (case-insensitive)."""
        needle = keyword.lower().strip()
        matches: List[str] = []
        for path, _ in _iter_paths(shap_json):
            if needle and needle in path.lower():
                matches.append(path)
            if len(matches) >= max_results:
                break
        return json.dumps(matches, ensure_ascii=False)

    @tool
    def search_tree_json_keys(keyword: str, max_results: int = 25) -> str:
        """Find tree JSON paths whose key/path text contains a keyword (case-insensitive)."""
        needle = keyword.lower().strip()
        matches: List[str] = []
        for path, _ in _iter_paths(tree_json):
            if needle and needle in path.lower():
                matches.append(path)
            if len(matches) >= max_results:
                break
        return json.dumps(matches, ensure_ascii=False)

    @tool
    def summarize_shap_progression() -> str:
        """Return compact SHAP trajectory summary: iterations, top features by max mean_abs_shap, and magnitude trend."""
        rows = shap_json.get("iteration_progress", [])
        if not isinstance(rows, list) or not rows:
            return "No SHAP iteration_progress rows found."

        feature_max: Dict[str, float] = {}
        magnitudes: List[float] = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            if "shap_magnitude" in row:
                try:
                    magnitudes.append(float(row.get("shap_magnitude", 0.0)))
                except (TypeError, ValueError):
                    pass
            elif "m" in row:
                try:
                    magnitudes.append(float(row.get("m", 0.0)))
                except (TypeError, ValueError):
                    pass

            if "top_shap_features" in row:
                candidates = row.get("top_shap_features", [])
                for feat in candidates:
                    if not isinstance(feat, dict):
                        continue
                    name = str(feat.get("feature_index"))
                    try:
                        score = float(feat.get("mean_abs_shap", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    feature_max[name] = max(feature_max.get(name, float("-inf")), score)
            else:
                compact_rows = row.get("f", [])
                iterable_rows = compact_rows
                if compact_rows and not isinstance(compact_rows[0], (list, tuple)):
                    iterable_rows = [compact_rows[i : i + 4] for i in range(0, len(compact_rows), 4)]
                for feat in iterable_rows:
                    if not isinstance(feat, (list, tuple)) or len(feat) < 2:
                        continue
                    name = str(feat[0])
                    try:
                        score = float(feat[1])
                    except (TypeError, ValueError):
                        score = 0.0
                    feature_max[name] = max(feature_max.get(name, float("-inf")), score)

        top_features = sorted(feature_max.items(), key=lambda kv: kv[1], reverse=True)[:10]
        magnitude_summary = {
            "count": len(magnitudes),
            "first": magnitudes[0] if magnitudes else None,
            "last": magnitudes[-1] if magnitudes else None,
            "min": min(magnitudes) if magnitudes else None,
            "max": max(magnitudes) if magnitudes else None,
        }

        return json.dumps(
            {
                "iteration_count": len(rows),
                "top_features_by_max_mean_abs_shap": top_features,
                "shap_magnitude_summary": magnitude_summary,
            },
            ensure_ascii=False,
        )

    @tool
    def get_feature_profile(feature_name: str) -> str:
        """Get comprehensive profile for a specific feature including trajectory, decision impact, and relationships."""
        if feature_profiles_json is None:
            return "FEATURE_PROFILES_NOT_AVAILABLE: Feature profiles JSON not provided"
        
        features = feature_profiles_json.get("features", {})
        if feature_name not in features:
            available = list(features.keys())[:10]
            return f"FEATURE_NOT_FOUND: '{feature_name}' not in profiles. Available features: {json.dumps(available, ensure_ascii=False)}"
        
        profile = features[feature_name]
        return json.dumps(profile, ensure_ascii=False)

    @tool
    def list_all_features() -> str:
        """List all tracked features in the profiles with their peak importance scores."""
        if feature_profiles_json is None:
            return "FEATURE_PROFILES_NOT_AVAILABLE: Feature profiles JSON not provided"
        
        features = feature_profiles_json.get("features", {})
        summary = [
            {
                "name": fname,
                "peak_importance": profile.get("importance_trajectory", {}).get("peak"),
                "stability": profile.get("importance_trajectory", {}).get("stability"),
            }
            for fname, profile in features.items()
        ]
        summary_sorted = sorted(summary, key=lambda x: x.get("peak_importance", 0), reverse=True)
        return json.dumps(summary_sorted, ensure_ascii=False)

    @tool
    def get_feature_impact_summary() -> str:
        """Get summary of all features' decision impacts and their interpretation."""
        if feature_profiles_json is None:
            return "FEATURE_PROFILES_NOT_AVAILABLE: Feature profiles JSON not provided"
        
        features = feature_profiles_json.get("features", {})
        impact_summary = [
            {
                "feature": fname,
                "high_value_effect": profile.get("decision_impact", {}).get("high_value_effect"),
                "low_value_effect": profile.get("decision_impact", {}).get("low_value_effect"),
                "total_swing": profile.get("decision_impact", {}).get("total_swing"),
                "interpretation": profile.get("decision_impact", {}).get("interpretation"),
            }
            for fname, profile in features.items()
        ]
        return json.dumps(impact_summary, ensure_ascii=False)

    @tool
    def search_feature_profiles(keyword: str) -> str:
        """Search feature profiles by feature name or characteristics (case-insensitive)."""
        if feature_profiles_json is None:
            return "FEATURE_PROFILES_NOT_AVAILABLE: Feature profiles JSON not provided"
        
        features = feature_profiles_json.get("features", {})
        needle = keyword.lower().strip()
        matches = []
        
        for fname in features.keys():
            if needle in fname.lower():
                matches.append({
                    "name": fname,
                    "peak": features[fname].get("importance_trajectory", {}).get("peak"),
                })
        
        if not matches:
            return f"NO_MATCHES: No features matched '{keyword}'"
        
        return json.dumps(matches, ensure_ascii=False)

    return [
        get_shap_json_at_path,
        get_tree_json_at_path,
        search_shap_json_keys,
        search_tree_json_keys,
        summarize_shap_progression,
        get_feature_profile,
        list_all_features,
        get_feature_impact_summary,
        search_feature_profiles,
    ]
