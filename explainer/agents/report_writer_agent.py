"""Professional report-writer agent for stakeholder-facing synthesis."""

from __future__ import annotations

from typing import Any, Dict, Literal

from langchain.agents import create_agent
from langchain_ollama import ChatOllama


ReportAudience = Literal[
    "stakeholder",
    "executive",
    "technical",
    "donor",
]

DEFAULT_REPORT_AUDIENCE: ReportAudience = "stakeholder"

_AUDIENCE_GUIDANCE: dict[ReportAudience, str] = {
    "stakeholder": (
        "Audience:\n"
        "- Senior stakeholders who make decisions based on analytical insights\n"
        "- Mixed technical/non-technical readership\n\n"
        "Style goals:\n"
        "- Prioritize strategic interpretation of evidence and uncertainty\n"
        "- Keep explanations accessible while preserving analytical precision\n"
        "- Do not use technical jargon without explanation, but do not oversimplify key concepts either"
    ),
    "executive": (
        "Audience:\n"
        "- Executive leadership with limited time\n"
        "- Readers focused on high-level implications and confidence\n\n"
        "Style goals:\n"
        "- Keep sections concise and high-signal\n"
        "- Surface core takeaways, caveats, and confidence without technical depth\n"
        "- Do not use technical jargon without explanation, but do not oversimplify key concepts either"
    ),
    "technical": (
        "Audience:\n"
        "- Technical analysts, data scientists, and model governance reviewers\n"
        "- Readers who expect methodological detail and assumptions\n\n"
        "Style goals:\n"
        "- Include model-behavior detail, data limitations, and interpretation caveats\n"
        "- Use precise technical wording while remaining structured and readable\n"
        "- Do not use technical jargon without explanation, but do not oversimplify key concepts either"
    ),
    "donor": (
        "Audience:\n"
        "- Donor and partner organizations reviewing evidence quality\n"
        "- Readers focused on accountability, transparency, and uncertainty\n\n"
        "Style goals:\n"
        "- Highlight traceability, confidence, and limitations clearly\n"
        "- Use neutral language suitable for external communication"
    ),
}


def _build_report_writer_prompt(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    feature_profiles_json: Dict[str, Any] | None,
    report_audience: ReportAudience,
) -> str:
    """Build report writer system prompt from shared and audience-specific blocks."""
    context_overview = (
        "Available analysis inputs:\n"
        f"- SHAP fields: {', '.join(list(shap_json.keys())[:6])}\n"
        f"- Tree fields: {', '.join(list(tree_json.keys())[:6])}\n"
        f"- Feature profiles included: {'yes' if feature_profiles_json is not None else 'no'}\n\n"
    )

    shared_rules = (
        "Rules:\n"
        "- Use the provided analysis context (country, target variable, prediction goal) to frame the report scope and narrative.\n"
        "- Expand upon the feature names the analyses provide, do not just restate them.\n"
        "- Do not mention the documents or tools you used\n"
        "- Do not provide concrete recommendations, action plans, or directives\n"
        "- Do not prescribe what stakeholders should do\n"
        "- Ground all claims in provided specialist content\n"
        "- If evidence is incomplete, state limitations and uncertainty explicitly\n"
        "- Write in professional report style with concise section headings\n"
        "- Prioritize factuality, clarity, neutrality, and decision relevance"
    )

    audience_guidance = _AUDIENCE_GUIDANCE[report_audience]
    return (
        "You are a professional report writer for an international organization.\n\n"
        f"{context_overview}"
        "Your role:\n"
        "- Convert expert analyses into a clear, structured report for decision audiences\n"
        "- Explain evidence, trends, and uncertainties in neutral, policy-facing language\n"
        "- Emphasize what the findings mean for situational understanding\n"
        "- Preserve traceability to specialist inputs and available quantitative details\n\n"
        f"Report audience mode: {report_audience}\n\n"
        f"{audience_guidance}\n\n"
        f"{shared_rules}"
    )


def create_report_writer_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
    report_audience: ReportAudience = DEFAULT_REPORT_AUDIENCE,
):
    """Create a professional report-writer agent for decision stakeholders.

    This agent synthesizes specialist insights into an executive-style report
    suitable for international organization stakeholders. It is intentionally
    non-prescriptive: it provides decision basis, not recommendations.

    Args:
        shap_json: SHAP trajectory data (context only)
        tree_json: Tree structure data (context only)
        model_name: Ollama model identifier
        temperature: LLM temperature
        feature_profiles_json: Optional feature-centric profiles
        report_audience: Audience-specific writing mode

    Returns:
        LangChain agent specialized in report writing
    """
    if report_audience not in _AUDIENCE_GUIDANCE:
        valid_modes = ", ".join(_AUDIENCE_GUIDANCE.keys())
        raise ValueError(f"Unsupported report_audience '{report_audience}'. Choose one of: {valid_modes}")

    llm = ChatOllama(model=model_name, temperature=temperature)

    system_prompt = _build_report_writer_prompt(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
        report_audience=report_audience,
    )

    return create_agent(model=llm, tools=[], system_prompt=system_prompt)


def ask_report_writer_agent(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
    report_audience: ReportAudience = DEFAULT_REPORT_AUDIENCE,
) -> str:
    """Convenience helper for one-shot report generation."""
    agent = create_report_writer_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
        report_audience=report_audience,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)
