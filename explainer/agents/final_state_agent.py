"""Agent focused only on final-model explainability metrics."""

from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from explainer.explainer_utils import final_explainability_schema_hint, json_for_prompt


def create_final_shap_agent(
    final_explainability_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
):
    """Create an agent that analyzes only final SHAP/performance metrics."""
    llm = ChatOllama(model=model_name, temperature=temperature)

    system_prompt = (
        "You are a machine learning explainability analyst focused only on a model's final-state explainability metrics.\n\n"
        "Your scope:\n"
        "- Analyze final SHAP feature importance metrics\n"
        "- Interpret the final confusion matrix and performance indicators\n"
        "- Explain what the final model appears to prioritize\n"
        "- Describe uncertainty and limits based only on the provided final-state summary\n\n"
        "Rules:\n"
        "- You do not have access to training trajectories, intermediate iterations, or tree history\n"
        "- Never imply temporal change, drift, or emergence over training\n"
        "- Ground every claim in the provided JSON only\n"
        "- Do not invent feature behavior beyond the provided SHAP, confusion matrix, and performance metrics\n"
        "- Explain technical terms clearly when needed\n"
        "- Prefer concise evidence-based prose over lists of raw values\n\n"
        f"{final_explainability_schema_hint(final_explainability_json)}"
        f"FINAL_EXPLAINABILITY_JSON:\n{json_for_prompt(final_explainability_json)}"
    )

    return create_agent(model=llm, tools=[], system_prompt=system_prompt)


def ask_final_shap_agent(
    question: str,
    final_explainability_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
) -> str:
    """Convenience helper for one-shot Q&A with final-state explainability only."""
    agent = create_final_shap_agent(
        final_explainability_json=final_explainability_json,
        model_name=model_name,
        temperature=temperature,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)