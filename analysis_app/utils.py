"""General utility functions for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .config import DEFAULT_APP_CONFIG, DEFAULT_TRAJECTORY_PIPELINE_CONFIG, Y_OPTIONS


def get_llm_config() -> Dict[str, Any]:
    """Get or initialize LLM configuration from session state."""
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = DEFAULT_APP_CONFIG.copy()
    return st.session_state.llm_config


def get_trajectory_pipeline_config():
    """Get or initialize trajectory pipeline configuration from session state."""
    return (
        st.session_state.trajectory_pipeline_config
        if "trajectory_pipeline_config" in st.session_state
        else DEFAULT_TRAJECTORY_PIPELINE_CONFIG
    )


def load_shap_json(path: str) -> Dict[str, Any]:
    """Load SHAP JSON from file."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def shap_json_to_df(shap_json: Dict[str, Any]) -> pd.DataFrame:
    """Convert SHAP JSON to pandas DataFrame."""
    rows: List[Dict[str, Any]] = []
    for step in shap_json.get("iteration_progress", []):
        if "top_shap_features" in step:
            iteration = step.get("iteration")
            for feature in step.get("top_shap_features", []):
                rows.append(
                    {
                        "iteration": int(iteration),
                        "feature_index": str(feature.get("feature_index")),
                        "mean_abs_shap": float(feature.get("mean_abs_shap", 0.0)),
                        "high_value_feature_impact": float(
                            feature.get("high_value_feature_impact", 0.0)
                        ),
                        "low_value_feature_impact": float(
                            feature.get("low_value_feature_impact", 0.0)
                        ),
                    }
                )
            continue

        iteration = step.get("i")
        compact_rows = step.get("f", [])
        if compact_rows and isinstance(compact_rows[0], (list, tuple)):
            iterable_rows = compact_rows
        else:
            iterable_rows = [compact_rows[i : i + 4] for i in range(0, len(compact_rows), 4)]

        for feature_row in iterable_rows:
            if len(feature_row) < 4:
                continue
            rows.append(
                {
                    "iteration": int(iteration),
                    "feature_index": str(feature_row[0]),
                    "mean_abs_shap": float(feature_row[1]),
                    "high_value_feature_impact": float(feature_row[2]),
                    "low_value_feature_impact": float(feature_row[3]),
                }
            )
    return pd.DataFrame(rows)


def pick_top_features(df: pd.DataFrame, top_k: int) -> List[str]:
    """Pick top K features by mean_abs_shap."""
    if df.empty:
        return []
    top = (
        df.groupby("feature_index")["mean_abs_shap"]
        .max()
        .nlargest(top_k)
        .index.astype(str)
        .tolist()
    )
    return top


def format_chat_question(history: List[Dict[str, str]], latest_question: str) -> str:
    """Build a compact multi-turn prompt for one-shot graph/agent calls."""
    if not history:
        return latest_question

    recent_turns = history[-6:]
    lines: List[str] = [
        "Use the recent conversation context if relevant.",
        "If earlier context conflicts with the provided JSON artifacts, prioritize the artifacts.",
        "",
        "Conversation history:",
    ]
    for msg in recent_turns:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    lines.append("")
    lines.append(f"Current user question: {latest_question}")
    return "\n".join(lines)
