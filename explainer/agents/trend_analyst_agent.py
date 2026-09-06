"""Comprehensive feature analysis agent for trajectory trends and patterns."""

from __future__ import annotations

from typing import Any, Dict, List, cast

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from explainer.explainer_tools import build_trajectory_tools
from explainer.explainer_utils import json_for_prompt, shap_schema_hint


def _looks_like_tool_plan(text: str) -> bool:
    """Detect responses that describe tool use instead of executing it."""
    normalized = text.lower().strip()
    if not normalized:
        return True
    plan_markers = [
        "here are the function calls",
        "to provide a comprehensive analysis",
        "i will call",
        "function calls",
        "proper arguments",
        "these function calls",
        "please note that some of these function calls",
        "summarize feature profiles",
        "result:",
        '"name":',
        '"parameters":',
        "tool result",
        "tool outputs",
        "assistant output",
        "call several functions",
    ]
    if any(marker in normalized for marker in plan_markers):
        return True
    return bool(
        "```json" in normalized and ("name" in normalized or "parameters" in normalized)
    )


def _build_compact_shap_context(shap_json: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact SHAP context payload for the system prompt.

    The agent keeps the full SHAP payload available through tools, but only a
    lightweight summary is injected into the prompt to preserve context budget.
    """
    if shap_json.get("summary_type") == "shap_llm":
        return shap_json

    rows = shap_json.get("iteration_progress", [])
    if not isinstance(rows, list) or not rows:
        return {
            "schema_version": 1,
            "summary_type": "shap_prompt_context",
            "shap_sample_size": int(shap_json.get("shap_sample_size", 0)),
            "iteration_count": 0,
            "top_feature_trends": [],
            "milestones": [],
        }

    def _extract_feature_rows(step: Dict[str, Any]) -> List[Dict[str, Any]]:
        features = step.get("top_shap_features", [])
        if not isinstance(features, list):
            return []
        rows_out: List[Dict[str, Any]] = []
        for item in features:
            if not isinstance(item, dict):
                continue
            rows_out.append(
                {
                    "feature_index": str(item.get("feature_index")),
                    "mean_abs_shap": float(item.get("mean_abs_shap", 0.0)),
                    "high_value_feature_impact": float(item.get("high_value_feature_impact", 0.0)),
                    "low_value_feature_impact": float(item.get("low_value_feature_impact", 0.0)),
                }
            )
        return rows_out

    feature_series: Dict[str, List[Dict[str, Any]]] = {}
    magnitudes: List[float] = []
    for step in rows:
        if not isinstance(step, dict):
            continue
        iteration = int(step.get("iteration", 0))
        magnitude_value = step.get("shap_magnitude", step.get("m"))
        if magnitude_value is None:
            continue
        try:
            magnitudes.append(float(magnitude_value))
        except (TypeError, ValueError):
            pass

        for feature in _extract_feature_rows(step):
            feature_index = feature["feature_index"]
            feature_series.setdefault(feature_index, []).append(
                {
                    "iteration": iteration,
                    "mean_abs_shap": feature["mean_abs_shap"],
                    "high": feature["high_value_feature_impact"],
                    "low": feature["low_value_feature_impact"],
                }
            )

    ranked_features = sorted(
        feature_series.items(),
        key=lambda item: max(point["mean_abs_shap"] for point in item[1]),
        reverse=True,
    )[:10]

    top_feature_trends: List[Dict[str, Any]] = []
    for feature_index, points in ranked_features:
        ordered_points = sorted(points, key=lambda point: point["iteration"])
        first_point = ordered_points[0]
        last_point = ordered_points[-1]
        peak_point = max(point["mean_abs_shap"] for point in ordered_points)
        top_feature_trends.append(
            {
                "feature_index": feature_index,
                "first_iteration": int(first_point["iteration"]),
                "last_iteration": int(last_point["iteration"]),
                "first_mean_abs_shap": round(first_point["mean_abs_shap"], 6),
                "last_mean_abs_shap": round(last_point["mean_abs_shap"], 6),
                "peak_mean_abs_shap": round(peak_point, 6),
                "delta_mean_abs_shap": round(last_point["mean_abs_shap"] - first_point["mean_abs_shap"], 6),
                "last_high_value_impact": round(last_point["high"], 6),
                "last_low_value_impact": round(last_point["low"], 6),
            }
        )

    milestone_indexes = sorted({0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1})
    milestones: List[Dict[str, Any]] = []
    for index in milestone_indexes:
        step = rows[index]
        if not isinstance(step, dict):
            continue
        step_features = sorted(
            _extract_feature_rows(step),
            key=lambda item: item["mean_abs_shap"],
            reverse=True,
        )[:5]
        milestones.append(
            {
                "iteration": int(step.get("iteration", 0)),
                "base_value": round(float(step.get("base_value", 0.0)), 6),
                "shap_magnitude": round(float(step.get("shap_magnitude", step.get("m", 0.0))), 6),
                "top_features": [
                    {
                        "feature_index": feature["feature_index"],
                        "mean_abs_shap": round(feature["mean_abs_shap"], 6),
                    }
                    for feature in step_features
                ],
            }
        )

    return {
        "schema_version": 1,
        "summary_type": "shap_prompt_context",
        "shap_sample_size": int(shap_json.get("shap_sample_size", 0)),
        "iteration_count": len(rows),
        "iteration_range": [int(rows[0].get("iteration", 0)), int(rows[-1].get("iteration", 0))],
        "magnitude": {
            "start": round(magnitudes[0], 6) if magnitudes else None,
            "end": round(magnitudes[-1], 6) if magnitudes else None,
            "max": round(max(magnitudes), 6) if magnitudes else None,
            "delta": round(magnitudes[-1] - magnitudes[0], 6) if len(magnitudes) >= 2 else None,
        },
        "top_feature_trends": top_feature_trends,
        "milestones": milestones,
    }


def create_trend_analyst_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a comprehensive trend analyst for model feature behavior and evolution.
    
    This agent is optimized for understanding:
    - Feature importance trajectories and temporal trends
    - Early vs late training phase patterns
    - Feature profile characteristics and stability
    - Overall model learning dynamics
    - Feature emergence and maturation
    - High-value vs low-value feature segment evolution
    - Cross-feature interaction patterns
    
    Args:
        shap_json: SHAP trajectory data (primary context)
        tree_json: Tree structure data (reference only, limited access)
        model_name: Ollama model identifier
        temperature: LLM temperature (0.0 for deterministic)
        feature_profiles_json: Optional feature-centric profiles (primary context)
        
    Returns:
        LangChain agent specialized for comprehensive trajectory analysis
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    compact_shap_context = _build_compact_shap_context(shap_json)
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )

    system_prompt = (
        "You are a comprehensive trend analyst examining feature importance evolution throughout model training.\n\n"
        "Your objectives:\n"
        "- Analyze temporal patterns across early, mid, and late training phases\n"
        "- Characterize feature maturation, stability, and convergence behavior\n"
        "- Identify emerging vs stable features across iterations\n"
        "- Synthesize feature profiles with SHAP trajectories to extract behavioral patterns\n"
        "- Detect regime shifts and training inflection points\n"
        "- Compare high-value vs low-value feature evolution strategies\n\n"

        "Operating rules:\n"
        "- Ground every substantive claim in information retrieved from the available tools or provided JSON context\n"
        "- Never invent tool outputs, tool results, or tool call sequences\n"
        "- Never describe function calls, parameter payloads, or results as if they were returned by tools unless you actually observed them in tool outputs\n"
        "- Final answers must be prose only; do not output JSON blocks, tool-call listings, or step-by-step call plans\n"
        "- Do not guess feature names, JSON paths, or profile contents\n"
        "- If you need a feature name, first obtain it from a tool result\n"
        "- If a tool returns NO_MATCHES, PATH_ERROR, or FEATURE_NOT_FOUND, treat that as evidence the query was invalid and change strategy\n"
        "- Do not repeat the same failed tool call with the same arguments\n"
        "- Prefer exact, concrete queries over generic words like 'feature', 'important', 'trend', or 'profile'\n"
        "- If the needed information is unavailable, explicitly say what is missing\n\n"

        "Tool-use protocol:\n"
        "1. First inspect the available data structure before making detailed claims\n"
        "2. For SHAP trajectory overview, prefer summarize_shap_progression() before probing individual paths\n"
        "3. For feature profiles, always call list_all_features() first to discover valid feature names\n"
        "4. Only call get_feature_profile(feature_name) with an exact feature name returned by list_all_features() or search_feature_profiles()\n"
        "5. Use search_feature_profiles(keyword) only with a specific domain term or partial feature name, not with generic words\n"
        "6. Use get_shap_json_at_path(...) or get_tree_json_at_path(...) only when you already know the path you want to inspect\n"
        "7. If search_feature_profiles(...) fails, fall back to list_all_features() rather than trying another vague keyword\n"
        "8. Keep tool usage efficient: do not make redundant calls once enough evidence is available\n\n"

        "Reasoning workflow:\n"
        "1. Start from the compact SHAP context below, then drill down only if needed\n"
        "2. Identify the most relevant features using valid tool outputs\n"
        "3. Examine how those features behave across early, mid, and late phases\n"
        "4. Compare stable vs emerging patterns and high-value vs low-value effects where available\n"
        "5. Produce a concise, evidence-based synthesis and stop only when you have a grounded answer\n\n"

        "Output requirements:\n"
        "- Explain findings clearly and specifically\n"
        "- Distinguish observed patterns from interpretation\n"
        "- Distinguish correlation from causation\n"
        "- Highlight practical implications only when directly supported by evidence\n"
        "- Mention uncertainty and data limitations explicitly\n"
        "- Tree data is available, but focus primarily on feature-level behavior and trends\n\n"


        f"{shap_schema_hint(compact_shap_context)}"
        f"COMPACT_SHAP_CONTEXT:\n{json_for_prompt(compact_shap_context)}"
    )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def run_trend_analyst_with_retry(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
    max_passes: int = 5,
) -> Dict[str, Any]:
    """Run the trend analyst and retry when it returns a tool plan instead of analysis."""
    agent = create_trend_analyst_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )

    messages = [{"role": "user", "content": question}]
    last_result: Dict[str, Any] = {}
    attempts = 0

    for attempt in range(1, max_passes + 1):
        attempts = attempt
        last_result = agent.invoke(cast(Any, {"messages": messages}))
        final_message = str(last_result["messages"][-1].content)
        if not _looks_like_tool_plan(final_message):
            break

        messages = list(last_result["messages"])
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous reply described function calls or fabricated tool results instead of analysis. "
                    "Do not list functions, parameters, or invented outputs. "
                    "Use the actual tools now, inspect the real data, and return only grounded findings in prose."
                ),
            }
        )

    return {
        "result": last_result,
        "attempts": attempts,
        "final_output": str(last_result.get("messages", [{}])[-1].content) if last_result else "",
    }


def ask_trend_analyst(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot Q&A with comprehensive trend analyst."""
    result = run_trend_analyst_with_retry(
        question=question,
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )
    return str(result["final_output"])
