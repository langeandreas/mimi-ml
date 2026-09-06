"""Utility functions for trajectory analysis."""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


ROUND_DECIMALS = 4
TOP_K_SHAP_FEATURES = 12
TOP_K_TREE_FEATURES = 10
TOP_K_INTERACTIONS = 8


TARGET_VARIABLE_READABLE = {
    "overall_mar": "Overall Mean Adequacy Ratio",
    "zn_ai": "Zinc Intake",
    "vita_rae_mcg": "Vitamin A Intake (RAE mcg)",
    "folate_mcg": "Folate Intake (mcg)",
    "fe_mg": "Iron Intake (mg)",
    "vitb12_mcg": "Vitamin B12 Intake (mcg)",
}


def round_float(x: float) -> float:
    """Round a float to the configured decimal places."""
    return round(float(x), ROUND_DECIMALS)


def readable_target_name(target_variable: str) -> str:
    """Return a human-readable label for a target variable code."""
    return TARGET_VARIABLE_READABLE.get(target_variable, target_variable)


def generate_output_file(base_path: str, default_filename: str) -> Path:
    """Returns a writable output file path.
    
    If path points to a JSON file, that file is used directly.
    If path points to a directory, default_filename is created inside it.
    """
    base_path_obj = Path(base_path)
    if base_path_obj.suffix.lower() == ".json":
        base_path_obj.parent.mkdir(parents=True, exist_ok=True)
        return base_path_obj

    base_path_obj.mkdir(parents=True, exist_ok=True)
    return base_path_obj / default_filename


def generate_output_dir(base_path: str) -> Path:
    """Returns an output directory path for generated artifacts."""
    base_path_obj = Path(base_path)
    output_dir = base_path_obj.parent if base_path_obj.suffix.lower() == ".json" else base_path_obj
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def substitute_feature_names(
    df: pd.DataFrame,
    feature_index_col: str,
    feature_names: list,
    features_csv_path: str = 'data/features_explanations.csv',
) -> pd.DataFrame:
    """Substitutes feature identifiers in a column with human-readable names."""
    if feature_index_col not in df.columns:
        return df

    try:
        names_df = pd.read_csv(features_csv_path)
        if "codename" in names_df.columns and "explanation" in names_df.columns:
            name_map = dict(zip(names_df["codename"].astype(str), names_df["explanation"].astype(str)))
        elif "feature_index" in names_df.columns and "feature_name" in names_df.columns:
            name_map = dict(
                zip(names_df["feature_index"].astype(str), names_df["feature_name"].astype(str))
            )
        else:
            return df
    except FileNotFoundError:
        return df

    df_substituted = df.copy()

    def _resolve_feature_name(value):
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if 0 <= idx < len(feature_names):
                codename = str(feature_names[idx])
                return name_map.get(codename, codename)
            return value

        value_str = str(value)

        # Handle indices serialized as strings, e.g. "12".
        if value_str.isdigit():
            idx = int(value_str)
            if 0 <= idx < len(feature_names):
                codename = str(feature_names[idx])
                return name_map.get(codename, codename)

        return name_map.get(value_str, value)

    df_substituted[feature_index_col] = df_substituted[feature_index_col].apply(_resolve_feature_name)
    return df_substituted


def expand_shap_step(step: dict) -> dict:
    """Normalize SHAP step payloads."""
    if "top_shap_features" in step:
        return {
            "iteration": int(step["iteration"]),
            "base_value": float(step["base_value"]),
            "shap_magnitude": float(step["shap_magnitude"]),
            "top_shap_features": step.get("top_shap_features", []),
        }

    compact_features = []
    compact_rows = step.get("f", [])

    # Accept both grouped rows [[fi, mean, high, low], ...] and flattened rows
    if compact_rows and isinstance(compact_rows[0], (list, tuple)):
        iterable_rows = compact_rows
    else:
        iterable_rows = [compact_rows[i:i + 4] for i in range(0, len(compact_rows), 4)]

    for item in iterable_rows:
        if len(item) < 4:
            continue
        compact_features.append(
            {
                "feature_index": item[0],
                "mean_abs_shap": float(item[1]),
                "high_value_feature_impact": float(item[2]),
                "low_value_feature_impact": float(item[3]),
            }
        )

    return {
        "iteration": int(step.get("i", 0)),
        "base_value": float(step.get("b", 0.0)),
        "shap_magnitude": float(step.get("m", 0.0)),
        "top_shap_features": compact_features,
    }


def build_phase_summaries(progress: list) -> list:
    """Builds early/mid/late split-feature summaries from iteration progress."""
    
    phase_summaries = []
    if not progress:
        return phase_summaries

    n = len(progress)
    split_points = [0, n // 3, (2 * n) // 3, n]
    labels = ["early", "mid", "late"]

    for idx in range(3):
        start = split_points[idx]
        end = split_points[idx + 1]
        if end <= start:
            continue

        phase_steps = progress[start:end]
        counter = Counter()
        for step in phase_steps:
            for item in step.get("top_split_features", []):
                counter[item["feature_index"]] += item["split_count"]

        top_phase = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        phase_summaries.append(
            {
                "phase_name": labels[idx],
                "iteration_range": [int(phase_steps[0]["iteration"]), int(phase_steps[-1]["iteration"])],
                "top_phase_features": [
                    {"feature_index": int(fi), "split_count": int(cnt)}
                    for fi, cnt in top_phase
                ],
            }
        )

    return phase_summaries
