"""LangGraph orchestration combining SHAP and tree specialist agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, NotRequired, Optional, TypedDict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from explainer.agents.shap_specialist_agent import create_shap_specialist_agent
from explainer.agents.tree_specialist_agent import create_tree_specialist_agent


class ExplainerGraphState(TypedDict):
    """State passed across the explainer analysis graph."""

    question: str
    route: NotRequired[Literal["shap", "tree", "both"]]
    shap_json: NotRequired[Dict[str, Any]]
    tree_json: NotRequired[Dict[str, Any]]
    shap_analysis: NotRequired[str]
    tree_analysis: NotRequired[str]
    final_answer: NotRequired[str]


class ExplainerGraphEvent(TypedDict):
    """Lightweight event emitted during graph execution."""

    node: str
    phase: Literal["start", "end", "info"]
    message: str
    payload: NotRequired[Dict[str, Any]]


def _route_question(question: str) -> Literal["shap", "tree", "both"]:
    """Route question to appropriate specialist(s) based on keywords."""
    q = question.lower()
    shap_keywords = ["shap", "feature impact", "importance", "drift", "magnitude"]
    tree_keywords = ["tree", "split", "depth", "leaf", "interaction"]

    asks_shap = any(keyword in q for keyword in shap_keywords)
    asks_tree = any(keyword in q for keyword in tree_keywords)

    if asks_shap and not asks_tree:
        return "shap"
    if asks_tree and not asks_shap:
        return "tree"
    return "both"


def build_explainer_graph(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Build a LangGraph pipeline with SHAP/tree specialists and synthesis.
    
    Graph flow:
        START -> router -> (shap_specialist|tree_specialist|both) -> synthesizer -> END
    
    The router directs questions to appropriate specialist(s) based on keywords.
    The synthesizer combines specialist outputs into a coherent answer.
    
    Args:
        shap_json: SHAP trajectory data
        tree_json: Tree structure data
        model_name: Ollama model identifier
        temperature: LLM temperature
        on_event: Optional callback for graph events
        feature_profiles_json: Optional feature-centric profiles
        
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

    llm = ChatOllama(model=model_name, temperature=temperature)

    # Create specialist agents
    shap_agent = create_shap_specialist_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )

    tree_agent = create_tree_specialist_agent(
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        feature_profiles_json=feature_profiles_json,
    )

    # Create synthesis agent
    synthesis_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=(
            "You are an orchestrator synthesizing insights from multiple specialist analyses.\n\n"
            "Your role:\n"
            "- Combine SHAP specialist and tree specialist outputs into coherent answers\n"
            "- Highlight connections between feature importance and tree splits\n"
            "- Reconcile any conflicting interpretations\n"
            "- Provide clear, concise final answers\n\n"
            "Rules:\n"
            "- Do not invent values; only use what specialists provided\n"
            "- Call out uncertainty when context is incomplete\n"
            "- Format responses for clarity and actionability"
        ),
    )

    # Define graph nodes
    def router_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Route question to specialists based on keywords."""
        emit("router", "start", "Analyzing question")
        route = _route_question(state["question"])
        emit("router", "end", f"Route: {route}", {"route": route})
        return {"route": route}

    def shap_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Query SHAP specialist."""
        emit("shap_specialist", "start", "Analyzing SHAP context")
        result = shap_agent.invoke({"messages": [{"role": "user", "content": state["question"]}]})
        analysis = str(result["messages"][-1].content)
        emit("shap_specialist", "end", "SHAP analysis complete")
        return {"shap_analysis": analysis}

    def tree_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Query tree specialist."""
        emit("tree_specialist", "start", "Analyzing tree context")
        result = tree_agent.invoke({"messages": [{"role": "user", "content": state["question"]}]})
        analysis = str(result["messages"][-1].content)
        emit("tree_specialist", "end", "Tree analysis complete")
        return {"tree_analysis": analysis}

    def synthesis_node(state: ExplainerGraphState) -> Dict[str, Any]:
        """Synthesize specialist analyses into final answer."""
        emit("synthesizer", "start", "Synthesizing specialist insights")
        
        synthesis_context = f"User question: {state['question']}\n\n"
        if "shap_analysis" in state:
            synthesis_context += f"SHAP specialist: {state['shap_analysis']}\n\n"
        if "tree_analysis" in state:
            synthesis_context += f"Tree specialist: {state['tree_analysis']}\n\n"
        
        result = synthesis_agent.invoke(
            {"messages": [{"role": "user", "content": synthesis_context}]}
        )
        final_answer = str(result["messages"][-1].content)
        emit("synthesizer", "end", "Synthesis complete")
        return {"final_answer": final_answer}

    # Build graph
    graph_builder = StateGraph(ExplainerGraphState)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("shap_specialist", shap_node)
    graph_builder.add_node("tree_specialist", tree_node)
    graph_builder.add_node("synthesizer", synthesis_node)

    # Add edges
    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "shap": "shap_specialist",
            "tree": "tree_specialist",
            "both": "shap_specialist",  # Start with SHAP when route="both"
        },
    )
    
    # After SHAP, check if we need tree too
    graph_builder.add_conditional_edges(
        "shap_specialist",
        lambda state: "tree_specialist" if state.get("route") == "both" else "synthesizer",
        {
            "tree_specialist": "tree_specialist",
            "synthesizer": "synthesizer",
        },
    )
    
    # Tree always goes to synthesizer
    graph_builder.add_edge("tree_specialist", "synthesizer")
    
    # Synthesizer always ends
    graph_builder.add_edge("synthesizer", END)

    return graph_builder.compile()


def run_explainer_graph(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
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
        model_name=model_name,
        temperature=temperature,
        on_event=collect,
        feature_profiles_json=feature_profiles_json,
    )
    
    state = graph.invoke(
        {
            "question": question,
            "shap_json": shap_json,
            "tree_json": tree_json,
        }
    )
    
    return {
        "answer": str(state.get("final_answer", "")),
        "events": events,
        "state": state,
    }


def ask_explainer_graph(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    on_event: Optional[Callable[[ExplainerGraphEvent], None]] = None,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot graph execution."""
    result = run_explainer_graph(
        question=question,
        shap_json=shap_json,
        tree_json=tree_json,
        model_name=model_name,
        temperature=temperature,
        on_event=on_event,
        feature_profiles_json=feature_profiles_json,
    )
    return str(result["answer"])
