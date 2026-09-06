"""LangGraph orchestration combining SHAP and tree specialist agents."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, NotRequired, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from explainer.agents.report_writer_agent import (
    DEFAULT_REPORT_AUDIENCE,
    ReportAudience,
    create_report_writer_agent,
)
from explainer.agents.final_state_agent import create_final_shap_agent
from explainer.agents.trend_analyst_agent import run_trend_analyst_with_retry
from explainer.explainer_tools import build_trajectory_tools


class ExplainerGraphState(TypedDict):
    """State passed across the explainer analysis graph."""

    question: str
    shap_json: NotRequired[Dict[str, Any]]
    tree_json: NotRequired[Dict[str, Any]]
    final_explainability_json: NotRequired[Dict[str, Any]]
    context_json: NotRequired[Dict[str, Any]]
    trend_analysis: NotRequired[str]
    final_shap_analysis: NotRequired[str]
    final_answer: NotRequired[str]


class ExplainerGraphEvent(TypedDict):
    """Lightweight event emitted during graph execution."""

    node: str
    phase: Literal["start", "end", "info"]
    message: str
    payload: NotRequired[Dict[str, Any]]


def _truncate_text(value: Any, max_chars: int = 4000) -> str:
    """Convert arbitrary content to readable truncated text."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15] + "\n...[truncated]"


def _message_text(content: Any) -> str:
    """Flatten LangChain message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(_truncate_text(item, max_chars=1000))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _tool_specs(feature_profiles_json: Dict[str, Any] | None, shap_json: Dict[str, Any], tree_json: Dict[str, Any]) -> list[Dict[str, str]]:
    """Describe tools available to the trend analyst."""
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )
    return [
        {
            "name": getattr(tool, "name", tool.__class__.__name__),
            "description": getattr(tool, "description", "").strip(),
        }
        for tool in tools
    ]


def _json_summary(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a compact summary of a JSON-like payload."""
    if not payload:
        return {"available": False}
    summary: Dict[str, Any] = {
        "available": True,
        "top_level_keys": list(payload.keys())[:20],
    }
    if "iteration_progress" in payload and isinstance(payload["iteration_progress"], list):
        summary["iteration_count"] = len(payload["iteration_progress"])
    if "features" in payload and isinstance(payload["features"], dict):
        summary["feature_count"] = len(payload["features"])
        summary["feature_examples"] = list(payload["features"].keys())[:10]
    if "feature_names" in payload and isinstance(payload["feature_names"], list):
        summary["feature_name_count"] = len(payload["feature_names"])
        summary["feature_name_examples"] = payload["feature_names"][:10]
    return summary


def _extract_agent_steps(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract assistant outputs, tool calls, and tool results from agent messages."""
    messages = result.get("messages", []) or []
    steps: list[Dict[str, Any]] = []
    final_output = ""

    for message in messages:
        message_type = getattr(message, "type", message.__class__.__name__.lower())
        content = _message_text(getattr(message, "content", ""))
        tool_calls = getattr(message, "tool_calls", None) or []

        if message_type in {"ai", "assistant"}:
            if content.strip():
                steps.append(
                    {
                        "kind": "assistant_output",
                        "content": _truncate_text(content),
                    }
                )
                final_output = content
            for tool_call in tool_calls:
                steps.append(
                    {
                        "kind": "tool_call",
                        "tool": str(tool_call.get("name", "unknown")),
                        "args": tool_call.get("args", {}),
                        "tool_call_id": tool_call.get("id"),
                    }
                )
            continue

        if message_type == "tool":
            steps.append(
                {
                    "kind": "tool_result",
                    "tool": getattr(message, "name", "unknown"),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "output": _truncate_text(content),
                }
            )
            continue

        if message_type in {"human", "user"} and content.strip():
            steps.append(
                {
                    "kind": "user_input",
                    "content": _truncate_text(content),
                }
            )

    return {
        "steps": steps,
        "final_output": final_output,
    }


def _format_event_markdown(event: ExplainerGraphEvent) -> str:
    """Render a single graph event into readable markdown."""
    lines = [f"## {event['node']} [{event['phase']}]: {event['message']}"]
    payload = event.get("payload") or {}
    if not payload:
        return "\n\n".join(lines)

    kind = payload.get("kind")
    if kind == "tool_call":
        lines.append(f"Tool: `{payload.get('tool', 'unknown')}`")
        lines.append("Arguments:")
        lines.append(f"```json\n{_truncate_text(payload.get('args', {}), max_chars=6000)}\n```")
        return "\n\n".join(lines)

    if kind == "tool_result":
        lines.append(f"Tool: `{payload.get('tool', 'unknown')}`")
        lines.append("Output:")
        lines.append(f"```text\n{payload.get('output', '')}\n```")
        return "\n\n".join(lines)

    if kind in {"assistant_output", "user_input"}:
        lines.append(f"```text\n{payload.get('content', '')}\n```")
        return "\n\n".join(lines)

    if "available_tools" in payload:
        lines.append("Available tools:")
        for tool in payload.get("available_tools", []):
            lines.append(f"- {tool.get('name', 'unknown')}: {tool.get('description', '')}")
    if "available_information" in payload:
        lines.append("Available information:")
        lines.append(f"```json\n{_truncate_text(payload.get('available_information', {}), max_chars=8000)}\n```")
    if "final_output" in payload:
        lines.append("Final output:")
        lines.append(f"```text\n{payload.get('final_output', '')}\n```")

    remaining = {
        key: value
        for key, value in payload.items()
        if key not in {"available_tools", "available_information", "final_output", "kind"}
    }
    if remaining:
        lines.append("Payload:")
        lines.append(f"```json\n{_truncate_text(remaining, max_chars=6000)}\n```")
    return "\n\n".join(lines)


def _format_agent_trace_markdown(question: str, events: list[ExplainerGraphEvent], answer: str) -> str:
    """Create a readable markdown execution trace."""
    sections = [
        "# Agent Execution Trace",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## User Question",
        "",
        question,
        "",
        "## Trace",
        "",
    ]
    for event in events:
        sections.append(_format_event_markdown(event))
        sections.append("")
    sections.extend([
        "## Final Answer",
        "",
        answer,
        "",
    ])
    return "\n".join(sections)


def _save_agent_trace_markdown(trace_markdown: str, output_dir: str = "results/reports") -> str:
    """Save agent execution trace under a dedicated logs directory."""
    directory = Path(output_dir) / "agent_logs"
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = directory / f"agent_trace_{timestamp}.md"
    output_file.write_text(trace_markdown + "\n", encoding="utf-8")
    return output_file.as_posix()





def _save_report_markdown(
    report_text: str,
    question: str,
    output_dir: str = "results/reports",
) -> str:
    """Save report text to markdown file and return the saved path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = directory / f"report_{timestamp}.md"

    header = (
        "# Stakeholder Insight Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Source Question\n\n"
        f"{question}\n\n"
        "## Report\n\n"
    )

    output_file.write_text(header + report_text + "\n", encoding="utf-8")
    return output_file.as_posix()


def _route_question(_question: str) -> Literal["shap"]:
    """Route question to trend analyst.

    All questions are routed to the comprehensive trend analyst,
    which handles feature importance, trajectories, patterns, and profiles.
    """
    return "shap"


def build_explainer_graph(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    final_explainability_json: Dict[str, Any] | None = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
    report_audience: ReportAudience = DEFAULT_REPORT_AUDIENCE,
):
    """Build a LangGraph pipeline with SHAP/tree specialists and report writing.

    Graph flow:
        START -> trend_analyst -> report_writer -> END
              -> final_shap_analyst -^

    The trend analyst extracts feature and training dynamics.
    The report writer turns those findings into stakeholder-facing prose.

    Args:
        shap_json: SHAP trajectory data
        tree_json: Tree structure data
        final_explainability_json: Compact final-state SHAP/performance payload
        model_name: Ollama model identifier
        temperature: LLM temperature
        on_event: Optional callback for graph events
        feature_profiles_json: Optional feature-centric profiles
        report_audience: Audience-specific report writing mode

    Returns:
        Compiled LangGraph
    """
    def emit(
        node: str,
        phase: Literal["start", "end", "info"],
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit graph events for monitoring."""
        if on_event is None:
            return
        event: ExplainerGraphEvent = {"node": node, "phase": phase, "message": message}
        if payload is not None:
            event["payload"] = payload
        on_event(event)

    trend_tools = _tool_specs(
        feature_profiles_json=feature_profiles_json,
        shap_json=shap_json,
        tree_json=tree_json,
    )

    # Create report writer agent
    report_writer_agent = create_report_writer_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
        report_audience=report_audience,
    )
    final_shap_agent = (
        create_final_shap_agent(
            final_explainability_json=final_explainability_json,
            model_name=model_name,
            temperature=temperature,
        )
        if final_explainability_json is not None
        else None
    )

    # Define graph nodes
    def trend_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Query trend analyst for comprehensive feature trajectory analysis."""
        emit("trend_analyst", "start", "Analyzing feature trajectories and patterns")
        emit(
            "trend_analyst",
            "info",
            "Trend analyst context and tool access",
            payload={
                "available_information": {
                    "question": state["question"],
                    "shap_json": _json_summary(shap_json),
                    "tree_json": _json_summary(tree_json),
                    "feature_profiles_json": _json_summary(feature_profiles_json),
                },
                "available_tools": trend_tools,
            },
        )
        retry_result = run_trend_analyst_with_retry(
            question=state["question"],
            shap_json=shap_json,
            tree_json=tree_json,
            model_name=model_name,
            temperature=temperature,
            feature_profiles_json=feature_profiles_json,
            max_passes=5,
        )
        result = retry_result["result"]
        attempts = int(retry_result.get("attempts", 1))
        if attempts > 1:
            emit(
                "trend_analyst",
                "info",
                f"Trend analyst required {attempts} passes",
                payload={"attempts": attempts},
            )
        trace_details = _extract_agent_steps(result)
        for step in trace_details["steps"]:
            emit("trend_analyst", "info", f"Trend analyst {step['kind'].replace('_', ' ')}", payload=step)
        analysis = str(trace_details["final_output"] or result["messages"][-1].content)
        emit(
            "trend_analyst",
            "info",
            "Trend analyst final output",
            payload={"final_output": _truncate_text(analysis, max_chars=8000)},
        )
        emit("trend_analyst", "end", "Trend analysis complete")
        return {"trend_analysis": analysis}

    def final_shap_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Analyze final-state SHAP metrics independently from trajectory trends."""
        if final_shap_agent is None or final_explainability_json is None:
            emit(
                "final_shap_analyst",
                "info",
                "Final explainability payload not available; skipping final SHAP analysis",
            )
            return {"final_shap_analysis": ""}

        emit("final_shap_analyst", "start", "Analyzing final-state SHAP and performance metrics")
        emit(
            "final_shap_analyst",
            "info",
            "Final SHAP analyst context and inputs",
            payload={
                "available_information": {
                    "question": state["question"],
                    "final_explainability_json": _json_summary(final_explainability_json),
                },
                "available_tools": [],
            },
        )

        result = final_shap_agent.invoke(
            {"messages": [{"role": "user", "content": state["question"]}]}
        )
        trace_details = _extract_agent_steps(result)
        for step in trace_details["steps"]:
            emit("final_shap_analyst", "info", f"Final SHAP analyst {step['kind'].replace('_', ' ')}", payload=step)
        analysis = str(trace_details["final_output"] or result["messages"][-1].content)
        emit(
            "final_shap_analyst",
            "info",
            "Final SHAP analyst output",
            payload={"final_output": _truncate_text(analysis, max_chars=8000)},
        )
        emit("final_shap_analyst", "end", "Final SHAP analysis complete")
        return {"final_shap_analysis": analysis}

    def report_writer_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Transform trend analysis into stakeholder-facing report text."""
        emit(
            "report_writer",
            "start",
            f"Drafting {report_audience} report from trend insights",
        )

        report_context = (
            f"User question: {state['question']}\n\n"
            f"Report audience mode: {report_audience}\n\n"
        )
        if "context_json" in state and state["context_json"]:
            report_context += f"Analysis context: {state['context_json']}\n\n"
        if "trend_analysis" in state:
            report_context += f"Trend analysis: {state['trend_analysis']}\n\n"
        final_shap_analysis = state.get("final_shap_analysis", "")
        if final_shap_analysis:
            report_context += f"Final SHAP analysis: {final_shap_analysis}\n\n"

        emit(
            "report_writer",
            "info",
            "Report writer context and inputs",
            payload={
                "available_information": {
                    "question": state["question"],
                    "context_json": state.get("context_json", {}),
                    "trend_analysis_preview": _truncate_text(state.get("trend_analysis", ""), max_chars=3000),
                    "final_shap_analysis_preview": _truncate_text(state.get("final_shap_analysis", ""), max_chars=3000),
                    "report_audience": report_audience,
                    "shap_json": _json_summary(shap_json),
                    "tree_json": _json_summary(tree_json),
                    "final_explainability_json": _json_summary(final_explainability_json),
                    "feature_profiles_json": _json_summary(feature_profiles_json),
                },
                "available_tools": [],
            },
        )

        result = report_writer_agent.invoke(
            {"messages": [{"role": "user", "content": report_context}]}
        )
        trace_details = _extract_agent_steps(result)
        for step in trace_details["steps"]:
            emit("report_writer", "info", f"Report writer {step['kind'].replace('_', ' ')}", payload=step)
        final_answer = str(trace_details["final_output"] or result["messages"][-1].content)
        emit(
            "report_writer",
            "info",
            "Report writer final output",
            payload={"final_output": _truncate_text(final_answer, max_chars=8000)},
        )
        emit("report_writer", "end", "Report generation complete")
        return {"final_answer": final_answer}

    # Build graph: START -> trend_analyst -> report_writer -> END
    #                         -> final_shap_analyst -^
    graph_builder = StateGraph(ExplainerGraphState)
    graph_builder.add_node("trend_analyst", trend_node)
    graph_builder.add_node("final_shap_analyst", final_shap_node)
    graph_builder.add_node("report_writer", report_writer_node)

    # Add edges
    graph_builder.add_edge(START, "trend_analyst")
    graph_builder.add_edge(START, "final_shap_analyst")
    graph_builder.add_edge("trend_analyst", "report_writer")
    graph_builder.add_edge("final_shap_analyst", "report_writer")
    graph_builder.add_edge("report_writer", END)

    return graph_builder.compile()


def run_explainer_graph(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    final_explainability_json: Dict[str, Any] | None = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
    context_json: Dict[str, Any] | None = None,
    report_audience: ReportAudience = DEFAULT_REPORT_AUDIENCE,
    save_report: bool = True,
    report_output_dir: str = "results/reports",
) -> Dict[str, Any]:
    """Execute explainer graph and return answer with events."""
    events: list[ExplainerGraphEvent] = []

    def collect(event: ExplainerGraphEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    graph = build_explainer_graph(
        shap_json=shap_json,
        tree_json=tree_json,
        final_explainability_json=final_explainability_json,
        model_name=model_name,
        temperature=temperature,
        on_event=collect,
        feature_profiles_json=feature_profiles_json,
        report_audience=report_audience,
    )

    state = graph.invoke(
        {
            "question": question,
            "shap_json": shap_json,
            "tree_json": tree_json,
            "final_explainability_json": final_explainability_json or {},
            "context_json": context_json or {},
        }
    )

    answer = str(state.get("final_answer", ""))
    trace_markdown = _format_agent_trace_markdown(question=question, events=events, answer=answer)
    trace_path = _save_agent_trace_markdown(
        trace_markdown=trace_markdown,
        output_dir=report_output_dir,
    )
    report_path: str | None = None
    if save_report and answer:
        report_path = _save_report_markdown(
            report_text=answer,
            question=question,
            output_dir=report_output_dir,
        )

    return {
        "answer": answer,
        "report_path": report_path,
        "trace_path": trace_path,
        "trace_markdown": trace_markdown,
        "events": events,
        "state": state,
    }


def ask_explainer_graph(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    final_explainability_json: Dict[str, Any] | None = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
    context_json: Dict[str, Any] | None = None,
    report_audience: ReportAudience = DEFAULT_REPORT_AUDIENCE,
    save_report: bool = True,
    report_output_dir: str = "results/reports",
) -> str:
    """Convenience helper for one-shot graph execution."""
    result = run_explainer_graph(
        question=question,
        shap_json=shap_json,
        tree_json=tree_json,
        final_explainability_json=final_explainability_json,
        model_name=model_name,
        temperature=temperature,
        on_event=on_event,
        feature_profiles_json=feature_profiles_json,
        context_json=context_json,
        report_audience=report_audience,
        save_report=save_report,
        report_output_dir=report_output_dir,
    )
    return str(result["answer"])
