"""Tree specialist agent for XGBoost structure and split analysis."""

from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from explainer.explainer_tools import build_trajectory_tools


def create_tree_specialist_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a tree specialist agent for model structure analysis.
    
    This agent is optimized for understanding:
    - Tree splits and depth progression
    - Feature split frequency
    - Split node statistics
    - Tree complexity evolution
    - Feature interactions at splits
    
    Args:
        shap_json: SHAP trajectory data (secondary context)
        tree_json: Tree structure data (primary context)
        model_name: Ollama model identifier
        temperature: LLM temperature (0.0 for deterministic)
        feature_profiles_json: Optional feature-centric profiles
        
    Returns:
        LangChain agent specialized for tree structure analysis
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )

    system_prompt = (
        "You are a tree structure specialist analyzing XGBoost model construction.\n\n"
        "Your expertise:\n"
        "- Tree splits and decision boundaries\n"
        "- Split depth and complexity\n"
        "- Feature usage in tree building\n"
        "- Tree evolution during training\n"
        "- Feature interactions at splits\n\n"
        "Instructions:\n"
        "- Focus on tree structure questions; deflect SHAP-specific questions to your colleagues\n"
        "- Use tool calls to inspect JSON values\n"
        "- Tree context must be retrieved from tools; no full tree JSON is pre-injected\n"
        "- Explain splits in business terms when possible\n"
        "- Provide analysis of tree complexity and efficiency\n"
        "- If information is missing, say so explicitly"
    )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def ask_tree_specialist(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot Q&A with tree specialist."""
    agent = create_tree_specialist_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)
