"""Explainer agents for model analysis.

This module provides specialized and generalist agents for analyzing XGBoost
models, specifically SHAP (feature importance) and tree structure analysis.

## Quick Start

```python
from explainer.agents import (
    create_generalist_agent,        # Generalist agent
    create_shap_specialist_agent,   # SHAP specialist
    create_tree_specialist_agent,   # Tree specialist
    build_explainer_graph,          # Graph orchestration
    ask_generalist_agent,           # One-shot Q&A (generalist)
    ask_shap_specialist,            # One-shot Q&A (SHAP)
    ask_tree_specialist,            # One-shot Q&A (tree)
)

# Use the generalist for simple queries
answer = ask_generalist_agent(
    question="Which features are most important?",
    shap_json=shap_data,
    tree_json=tree_data,
)

# Or use the graph for complex questions that need specialist input
from explainer.agents import run_explainer_graph
result = run_explainer_graph(
    question="How do water_access splits correlate with feature importance?",
    shap_json=shap_data,
    tree_json=tree_data,
)
print(result["answer"])
```

## Architecture

### Agents

- **Generalist** (`generalist_agent.py`): Analyzes both SHAP and tree data
- **SHAP Specialist** (`shap_specialist_agent.py`): Focused on feature importance
- **Tree Specialist** (`tree_specialist_agent.py`): Focused on model structure
- **Report Writer** (`report_writer_agent.py`): Produces stakeholder-facing, non-prescriptive reports
- **Graph** (`explainer_graph.py`): Routes questions to specialists and synthesizes answers

### Adding New Agents

To add a new specialist agent (e.g., for feature interactions):

1. Create `{domain}_specialist_agent.py` with a `create_{domain}_specialist_agent()` function
2. Import and use it in your analysis:

```python
from explainer.agents.your_specialist_agent import create_your_specialist_agent

agent = create_your_specialist_agent(shap_json, tree_json)
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

3. To integrate into the graph, modify `explainer_graph.py` to add the new specialist node

## Design Pattern

All agents follow the factory pattern:
```python
def create_[domain]_[specialist_type]_agent(
    shap_json, tree_json, model_name, temperature, feature_profiles_json
) -> LangChain Agent:
    ...create and return agent...
```

This makes it easy to:
- Add new specialists
- Test different LLM backends
- Configure agents independently
- Compose them in different graph topologies
"""

from __future__ import annotations

# Generalist agent
from explainer.agents.generalist_agent import (
    ask_generalist_agent,
    create_generalist_agent,
)

from explainer.agents.final_state_agent import (
    ask_final_shap_agent,
    create_final_shap_agent,
)

# Trend analyst agent
from explainer.agents.trend_analyst_agent import (
    ask_trend_analyst,
    create_trend_analyst_agent,
)



# Report writer agent
from explainer.agents.report_writer_agent import (
    DEFAULT_REPORT_AUDIENCE,
    ReportAudience,
    ask_report_writer_agent,
    create_report_writer_agent,
)

# Graph orchestration
from explainer.agents.explainer_graph import (
    ExplainerGraphEvent,
    ExplainerGraphState,
    ask_explainer_graph,
    build_explainer_graph,
    run_explainer_graph,
)

__all__ = [
    # Generalist
    "create_generalist_agent",
    "ask_generalist_agent",
    # Final SHAP
    "create_final_shap_agent",
    "ask_final_shap_agent",
    # Trend analyst
    "create_trend_analyst_agent",
    "ask_trend_analyst",
    # Report writer
    "ReportAudience",
    "DEFAULT_REPORT_AUDIENCE",
    "create_report_writer_agent",
    "ask_report_writer_agent",
    # Graph
    "build_explainer_graph",
    "run_explainer_graph",
    "ask_explainer_graph",
    "ExplainerGraphState",
    "ExplainerGraphEvent",
]
