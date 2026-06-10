"""SHAP specialist agent for feature importance and impact analysis."""

from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from explainer.explainer_tools import build_trajectory_tools
from explainer.explainer_utils import json_for_prompt, shap_schema_hint


def create_shap_specialist_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a SHAP specialist agent for feature importance analysis.
    
    This agent is optimized for understanding:
    - Feature importance trajectories
    - Feature impact variations
    - SHAP magnitude changes
    - Feature drift patterns
    
    Args:
        shap_json: SHAP trajectory data (primary context)
        tree_json: Tree structure data (secondary context)
        model_name: Ollama model identifier
        temperature: LLM temperature (0.0 for deterministic)
        feature_profiles_json: Optional feature-centric profiles
        
    Returns:
        LangChain agent specialized for SHAP analysis
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )

    system_prompt = (
        "You are a SHAP specialist analyzing feature importance in model training trajectories.\n\n"
        "Your expertise:\n"
        "- Feature importance magnitude and trends\n"
        "- High-value vs low-value feature impacts\n"
        "- Feature emergence patterns\n"
        "- SHAP drift detection\n"
        "- Feature stability analysis\n\n"
        "Instructions:\n"
        "- Focus on SHAP-related questions; deflect tree-specific questions to your colleagues\n"
        "- Use tool calls to inspect JSON values\n"
        "- Ground all insights in the provided SHAP data\n"
        "- Provide concrete numbers and trends\n"
        "- If information is missing, say so explicitly\n\n"
        f"{shap_schema_hint(shap_json)}"
        f"SHAP_JSON:\n{json_for_prompt(shap_json)}"
    )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def ask_shap_specialist(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot Q&A with SHAP specialist."""
    agent = create_shap_specialist_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)
