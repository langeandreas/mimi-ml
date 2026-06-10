# Feature-Centric Format for LLM Analysis (documentation created using AI)

## Overview

The feature-centric format (`generate_llm_optimized_feature_profiles`) reorganizes SHAP and tree information by **feature** rather than iteration. This structure is optimized for LLM consumption because:

1. **Natural reasoning**: LLMs think about features as entities, not iterations
2. **Reduced tokens**: Hierarchical organization reduces redundancy vs. flattened arrays
3. **Interpretable structure**: Decision impacts and trajectories are pre-calculated
4. **Semantic relationships**: Feature interactions are explicit rather than implicit

## Data Structure

```json
{
  "schema": "feature_profiles_v1",
  "summary_type": "feature_profiles",
  "metadata": {
    "top_k_features": 15,
    "total_features_tracked": 50,
    "iterations_analyzed": 120,
    "shap_sample_size": 100
  },
  "features": {
    "water_access": {
      "importance_trajectory": {
        "start": 0.12,
        "end": 0.38,
        "peak": 0.45,
        "peak_at_iteration": 35,
        "delta": 0.26,
        "direction": "↑ increasing",
        "stability": "stable",
        "iteration_range": [0, 120]
      },
      "decision_impact": {
        "high_value_effect": 0.23,
        "low_value_effect": -0.19,
        "total_swing": 0.42,
        "interpretation": "High water_access → positive prediction; Low water_access → negative prediction"
      },
      "emergence_pattern": {
        "first_important_iteration": 0,
        "reached_peak_at": 35,
        "iterations_to_peak": 35,
        "stabilized_after": 75
      },
      "prevalence": {
        "appearances_in_trajectory": 85,
        "percentage_of_iterations": 70.8
      },
      "related_features": ["water_type", "distance_to_source", "seasonal_variation"]
    }
  }
}
```

## Key Fields

### `importance_trajectory`
- **start/end/peak**: SHAP values at different phases
- **direction**: Arrows (↑/↓/→) indicate trend
- **stability**: How consistent the feature's importance is
  - `very_stable` (CV < 0.1)
  - `stable` (CV < 0.25)
  - `moderate_variation` (CV < 0.5)
  - `highly_variable` (CV ≥ 0.5)

### `decision_impact`
- **high_value_effect**: SHAP contribution when feature value is high
- **low_value_effect**: SHAP contribution when feature value is low
- **interpretation**: Human-readable explanation of the feature's effect

### `emergence_pattern`
- Identifies when the feature became important
- How quickly it reached peak importance
- When stabilization occurred

### `prevalence`
- How often the feature appears in top-k across iterations
- Indicates reliability/consistency

## Usage in LangChain Agent

The feature profiles are automatically available to the LLM agent through new tools:

```python
# Pass feature profiles to the agent
agent = create_generalist_agent(
    shap_json=shap_json,
    tree_json=tree_json,
    feature_profiles_json=feature_profiles_json,
    model_name="llama3.1:8b",
)

# The agent now has access to these tools:
# - get_feature_profile(feature_name): Get full profile for a feature
# - list_all_features(): List all tracked features with peak importance
# - get_feature_impact_summary(): Summary of all features' decision impacts
# - search_feature_profiles(keyword): Find features matching a keyword
```

## Pipeline Integration

The feature profiles are generated automatically when enabled in the config:

```python
from explainer.analysis.trajectory_pipeline_config import TrajectoryPipelineConfig
from explainer.analysis.trajectory_pipeline import run_classification_trajectory_pipeline

config = TrajectoryPipelineConfig(
    type_target="overall_mar",
    best_hyperparams_path="path/to/hyperparams.csv",
    country_iso="LKA",
    features_path="data/lka_features.csv",
    targets_path="data/targets.csv",
    model_name="xgboost_v1",
    generate_feature_profiles_llm=True,  # Enable feature profiles
    feature_profiles_top_k=15,
)

summary_traj, artifacts = run_classification_trajectory_pipeline(
    y=targets_df,
    data_all=features_df,
    config=config,
)

# Access the feature profiles
feature_profiles = artifacts["feature_profiles_json"]
```

## Example LLM Queries

With this structure, the LLM can directly answer:

1. **"Which features drive positive predictions?"**
   - Uses `decision_impact.high_value_effect` and `interpretation`

2. **"What's the most stable feature in the model?"**
   - Queries `stability` across all features

3. **"When did water_access become important?"**
   - Uses `emergence_pattern` to pinpoint iteration range

4. **"How does water_access interact with other factors?"**
   - Uses `related_features` to identify co-splits in trees

5. **"Which features consistently stay in top-k?"**
   - Uses `prevalence.percentage_of_iterations`

## Token Efficiency

Compared to raw SHAP/tree JSON:

- **Compact size**: ~60-70% smaller than iteration-based format
- **Reduced redundancy**: No repeated feature information across iterations
- **Semantic depth**: Pre-computed statistics reduce LLM computation
- **Fast retrieval**: Features are indexed by name for instant lookups

## Output Files

When saved, the feature profiles are written to:
```
trajectory_feature_profiles_llm.json
```

This file can be used standalone for visualization, further processing, or as the primary interface to the model's decision logic.
