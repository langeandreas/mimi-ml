# Explainer Agent Framework (documentation created using AI)

## New Structure

```
explainer/
├── agents/
│   ├── __init__.py                    (public API, re-exports)
│   ├── generalist_agent.py            (generalist agent)
│   ├── shap_specialist_agent.py       (SHAP specialist)
│   ├── tree_specialist_agent.py       (Tree specialist)
│   └── explainer_graph.py             (LangGraph orchestration)
├── explainer_tools.py                 (agent tools)
├── explainer_utils.py                 (utility functions)
```

## Factory Pattern

All agents follow the factory pattern:

```python
def create_[domain]_[type]_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a [domain] agent for [type] analysis."""
    ...
```

## Adding a New Specialist Agent

### Step 1: Create Agent File
Create `explainer/agents/your_specialist_agent.py`:

```python
"""Your specialist agent for [domain] analysis."""

from __future__ import annotations

from typing import Any, Dict
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from explainer.explainer_tools import build_trajectory_tools
from explainer.explainer_utils import json_for_prompt

def create_your_specialist_agent(
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
):
    """Create a specialist agent for [domain]."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    tools = build_trajectory_tools(
        shap_json=shap_json,
        tree_json=tree_json,
        feature_profiles_json=feature_profiles_json,
    )
    
    system_prompt = (
        "You are a specialist in [domain]. "
        "Focus on [domain]-specific insights. "
        "Use tools to ground your responses."
    )
    
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)

def ask_your_specialist(
    question: str,
    shap_json: Dict[str, Any],
    tree_json: Dict[str, Any],
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
    feature_profiles_json: Dict[str, Any] | None = None,
) -> str:
    """Convenience helper for one-shot Q&A."""
    agent = create_your_specialist_agent(...)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return str(result["messages"][-1].content)
```

### Step 2: Export in `agents/__init__.py`

Add to [agents/__init__.py](agents/__init__.py):

```python
from explainer.agents.your_specialist_agent import (
    ask_your_specialist,
    create_your_specialist_agent,
)

__all__ = [
    # ... existing exports ...
    "create_your_specialist_agent",
    "ask_your_specialist",
]
```

### Step 3: Integrate into Graph (Optional)

To add your specialist to the graph, modify `explainer_graph.py`:

```python
# In build_explainer_graph():
your_agent = create_your_specialist_agent(...)

# In graph_builder, add a node:
graph_builder.add_node("your_specialist", your_specialist_node)

# Connect it based on your routing logic
```

## Usage Examples

### Generalist Agent (Simple Queries)

```python
from explainer.agents import ask_generalist_agent

answer = ask_generalist_agent(
    question="What are the top 5 features?",
    shap_json=shap_data,
    tree_json=tree_data,
)
```

### Specialist Agents (Focused Analysis)

```python
from explainer.agents import ask_shap_specialist, ask_tree_specialist

# SHAP specialist
shap_analysis = ask_shap_specialist(
    question="How does water_access importance change over time?",
    shap_json=shap_data,
    tree_json=tree_data,
)

# Tree specialist
tree_analysis = ask_tree_specialist(
    question="What are the most common split features?",
    shap_json=shap_data,
    tree_json=tree_data,
)
```

### Graph Orchestration (Complex Queries)

```python
from explainer.agents import ask_explainer_graph

# Router automatically determines which specialists to use
answer = ask_explainer_graph(
    question="How do feature splits relate to SHAP values?",
    shap_json=shap_data,
    tree_json=tree_data,
    on_event=lambda evt: print(f"[{evt['node']}] {evt['message']}"),
)
```

## File Cleanup

Old files can now be deleted:
- `explainer/langchain_trajectory_agent.py`
- `explainer/langchain_trajectory_single_agent.py`
- `explainer/langchain_trajectory_utils.py`

They are replaced by:
- `explainer/agents/` (with 4 new files)
- `explainer/trajectory_tools.py` (renamed)
- `explainer/trajectory_utils.py` (renamed)

## Benefits

✅ **Modularity**: Each agent is independent and testable  
✅ **Extensibility**: Easy to add new specialists following the same pattern  
✅ **Clarity**: File names clearly indicate purpose  
✅ **Consistency**: All agents use the same factory pattern  
✅ **Composability**: Agents can be used individually or combined in graphs  
✅ **Maintainability**: No tangled dependencies between agents
