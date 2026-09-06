"""Shared helpers for trajectory agent and graph modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def json_for_prompt(payload: Dict[str, Any]) -> str:
    """Serialize JSON payloads deterministically for stable prompts."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def shap_schema_hint(shap_json: Dict[str, Any]) -> str:
    """Return a short schema hint when compact SHAP payloads are used."""
    if shap_json.get("summary_type") == "shap_llm":
        return (
            "\nLLM SHAP summary note: this is an aggregated trajectory summary, not full per-step detail. "
            "Use top_feature_trends, milestones, change_points, and magnitude fields for analysis.\n"
        )
    if int(shap_json.get("schema_version", 1)) != 2:
        return ""
    return (
        "\nCompact SHAP schema note: iteration_progress rows use short keys "
        "i=iteration, b=base_value, m=shap_magnitude, f=feature rows. "
        "Each feature row is [feature_index, mean_abs_shap, high_value_feature_impact, low_value_feature_impact].\n"
    )


def final_explainability_schema_hint(payload: Dict[str, Any]) -> str:
    """Return a short schema hint for compact final explainability payloads."""
    if payload.get("summary_type") != "final_model_explainability":
        return ""
    if int(payload.get("schema_version", 1)) != 2:
        return ""
    return (
        "\nCompact final explainability schema note: counts=train/test/features, "
        "balance=train_pos_rate/test_pos_rate, confusion.norm_true is row-normalized by actual class, "
        "shap.base is the expected value, shap.mean_abs is global mean absolute SHAP, and "
        "shap.top_features rows follow shap.feature_row_format.\n"
    )


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)
