"""Generalist explainer agent combining SHAP and tree analysis."""

from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from explainer.explainer_tools import build_trajectory_tools
from explainer.explainer_utils import json_for_prompt, shap_schema_hint


def create_generalist_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a generalist agent for explainer analysis (SHAP + tree).
    
    This agent has access to both SHAP and tree context and can answer
    questions about the overall model training process.
    
    Args:
        shap_json: SHAP trajectory data
        tree_json: Tree structure data
        model_name: Ollama model identifier
        temperature: LLM temperature (0.0 for deterministic)
        feature_profiles_json: Optional feature-centric profiles for LLM
        
    Returns:
        LangChain agent ready for invoke()
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )

    system_prompt = (
        "You are a machine learning expert analyzing model training trajectories. "
        "You have access to both SHAP (feature importance) and tree structure data.\n\n"
        "Instructions:\n"
        "- Use tool calls to inspect SHAP and tree trajectory JSON and answer questions\n"
        "- Tree context is available through tools; do not rely on any full tree JSON dump\n"
        "- Ground all responses in tool outputs; do not invent values\n"
        "- If information is missing, say so explicitly\n"
        "- Do not output code snippets or pseudo-code\n\n"
        f"{shap_schema_hint(shap_json)}"
        f"SHAP_JSON:\n{json_for_prompt(shap_json)}"
    )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def ask_generalist_agent(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot Q&A with the generalist agent."""
    agent = create_generalist_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)
