# xgb-pathfinder

A generalized, SHAP-like package for explaining XGBoost tree-based models through trajectory metrics, cohort analysis, and geographic impact visualization.

**Status**: ✅ Phases 1-4 complete (Core + Analysis + Geographic)  
**Version**: 0.1.0  
**Python**: 3.8+

---

## Overview

`xgb-pathfinder` extracts actionable insights from boosting models by analyzing:

1. **Trajectory Metrics** — How feature importance evolves across training iterations
   - Velocity: Feature gain per iteration
   - Acceleration: Change in velocity (identifies turning points)
   - Feature Confusion Index (FCI): Conditional probability metric
   - Behavioral Signature: CORE_DRIVER, LATE_LEARNER, OUTLIER_SPECIALIST, etc.

2. **Cohort Analysis** — Population segments defined by decision tree patterns
   - Scope signature: Early-stage conditions defining segment
   - Decider feature: Primary decision-making feature
   - Enriched metrics: Support count, risk rate, decision impact, local SHAP summary

3. **Geographic Impact** — Regional aggregation prioritizing prediction impact over frequency
   - Impact = prediction magnitude × support count
   - Outlier specialist identification
   - Scope→decider relationship graphs

---

## Installation

### Development Install (from project root)

```bash
cd xgb_pathfinder
pip install -e .
```

### Dependencies

```bash
pip install xgboost numpy pandas scikit-learn shap

# Optional: Geographic features
pip install geopandas folium matplotlib
```

---

## Quick Start

### Basic API

```python
from xgb_pathfinder import ModelExplainer

# Initialize with trained XGBoost booster
explainer = ModelExplainer(
    booster=trained_model,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
)

# Compute feature trajectories
trajectories = explainer.compute_trajectory_metrics()
for feat_name, traj in trajectories.items():
    print(f"{feat_name}: peak_velocity={traj['peak_velocity']:.4f}, "
          f"behavior={traj['behavior_signature']}")

# Extract cohorts (population segments)
cohorts = explainer.extract_cohorts(min_support=20)
for cohort in cohorts[:5]:
    print(f"Cohort {cohort['cohort_id']}: "
          f"{cohort['support_count']} samples, "
          f"risk={cohort['risk_rate']:.2%}, "
          f"impact={cohort['decision_impact']:.3f}")

# Get comprehensive feature profiles
profiles = explainer.get_feature_profiles()
for feat_name, profile in profiles.items():
    print(f"{feat_name}:")
    print(f"  - Behavior: {profile['behavior_signature']}")
    print(f"  - Peak importance: {profile['importance_trajectory']['peak_velocity']:.4f}")
    print(f"  - SHAP impact: {profile['decision_impact']['total_swing']:.4f}")

# Geographic aggregation
regions = explainer.aggregate_by_region(
    geodata_path="admin_boundaries.geojson",
    sample_region_mapping=sample_to_region_mapping,
    impact_metric="magnitude_x_frequency",
)

# Visualize impact
fig = explainer.plot_geographic_impact(
    geodata_path="admin_boundaries.geojson",
    sample_region_mapping=sample_to_region_mapping,
    output_path="impact_map.html",
)

# Export metrics
explainer.to_csv("output_dir/")
explainer.to_json("output.json")
data = explainer.to_dict()
```

---

## Module Structure

### Core API (`explainer.py`)

**`ModelExplainer`** — Main entry point

Public methods:
- `compute_trajectory_metrics()` → Dict[feature → trajectory metrics]
- `extract_cohorts(min_support)` → List[CohortRecord]
- `get_feature_profiles()` → Dict[feature → profile]
- `get_segments(rules)` → DataFrame with sample-to-cohort mapping
- `aggregate_by_region(geodata, mapping)` → Dict[region → impact metrics]
- `plot_geographic_impact(geodata, mapping)` → Map object
- `to_dict() / to_json() / to_csv()` → Export metrics

### Analysis Engines (`analysis/`)

| Module | Responsibility |
|--------|-----------------|
| `tree_engine.py` | Tree structure analysis, rule extraction |
| `trajectory_engine.py` | Velocity, acceleration, FCI, behavioral signatures |
| `shap_engine.py` | SHAP value computation, impact profiles |
| `cohort_engine.py` | Cohort extraction, sample assignment, enrichment |
| `feature_analyzer.py` | Feature profiles, interactions, path statistics |
| `context_analyzer.py` | Early/late positioning, scope→decider graphs |
| `utils.py` | Tree traversal, normalization, statistics |

### Geographic Modules (`geographic/`)

| Module | Responsibility |
|--------|-----------------|
| `impact_aggregator.py` | Aggregate cohorts by region |
| `impact_prioritizer.py` | Rank regions by positive prediction impact |
| `geo_plotter.py` | Choropleth visualization (matplotlib/folium) |
| `mappings.py` | Admin hierarchy & sample→region utilities |

### Type Definitions (`types.py`)

TypedDict definitions for strong typing:
- `TrajectoryMetric` — Per-iteration metrics
- `FeatureTrajectory` — Complete feature trajectory
- `DecisionRule` — Root-to-leaf path rules
- `CohortRecord` — Aggregated cohort
- `FeatureProfile` — Comprehensive feature analysis
- `RegionImpactMetric` — Geographic aggregation
- `PathfinderConfig` — Configuration parameters

### Configuration (`config.py`)

All thresholds are configurable:
- Acceleration thresholds (early/late phase detection)
- Cohort support & path count minimums
- Geographic aggregation settings
- Top-K export values

---

## Data Models

### FeatureTrajectory

```python
{
    "feature_name": "age",
    "feature_index": 0,
    "metrics": [
        {
            "iteration": 0,
            "velocity": 0.1234,       # Gain at iteration
            "acceleration": nan,      # ΔGain (NaN for iteration 0)
            "fci": 0.567,            # Feature Confusion Index
            "cumulative_gain": 0.1234,
            "hessian_mean": 0.05,
        },
        # ... more iterations
    ],
    "peak_iteration": 15,
    "peak_velocity": 0.8901,
    "total_gain": 5.432,
    "behavior_signature": "CORE_DRIVER",  # or LATE_LEARNER, OUTLIER_SPECIALIST, etc.
}
```

### CohortRecord

```python
{
    "cohort_id": "cohort_0",
    "scope_signature": "age < 35 AND income < 1000",  # Early conditions
    "decider_feature_name": "education_level",        # Primary late decider
    "decider_feature_index": 5,
    "support_count": 245,          # Samples in cohort
    "risk_rate": 0.62,             # Positive rate
    "mean_confidence": 0.71,       # Model's mean prediction
    "late_decider_rate": 0.73,     # Fraction of paths where decider is late
    "decision_impact": 0.45,       # Combined impact score
    "impact_magnitude_ratio": 1.23, # Relative to global mean
    "leaf_values": [0.2, 0.3, 0.25],  # Leaf values when decider appears
    "top_features_by_shap": [("income", 0.12), ("age", 0.08)],
    "contributing_rules": ["tree_0_path_0", "tree_1_path_2"],
}
```

### RegionImpactMetric

```python
{
    "region_name": "northern_district",
    "region_id": "district_001",
    "cohort_count": 12,            # Unique cohorts in region
    "total_impact": 5.67,          # Sum of impacts
    "avg_impact_per_cohort": 0.47,
    "avg_impact_per_sample": 0.023,
    "total_support": 245,          # Total samples
    "top_cohorts": [
        {"cohort_id": "cohort_3", "support": 45, "impact": 0.89},
        # ... top 5 cohorts
    ],
}
```

---

## Behavior Signatures

Features are classified based on acceleration patterns:

| Signature | Pattern | Meaning |
|-----------|---------|---------|
| **CORE_DRIVER** | Early accel → late stabilization | Strong early, importance plateaus |
| **LATE_LEARNER** | Weak early, late acceleration | Importance grows over time |
| **OUTLIER_SPECIALIST** | High early AND late acceleration | Continuous improvement, special case handling |
| **UNSTABLE** | High variance | Oscillating importance, possible collinearity |
| **STABLE_BASELINE** | Consistent velocity | Steady contribution |

---

## Configuration

Customize via `config` parameter or environment variables:

```python
config = {
    "acceleration_threshold": 0.1,
    "early_acceleration_threshold": 0.05,
    "late_acceleration_threshold": 0.05,
    "min_cohort_support": 10,
    "min_paths_per_cohort": 2,
    "admin_level": 2,
    "top_k_features": 15,
    "json_indent": 2,
}

explainer = ModelExplainer(
    booster, X_train, y_train,
    feature_names=features,
    config=config,
)
```

---

## Example: Complete Workflow

```python
import pandas as pd
from xgb_pathfinder import ModelExplainer

# 1. Initialize
explainer = ModelExplainer(
    booster=model,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
)

# 2. Analyze trajectories
trajectories = explainer.compute_trajectory_metrics()
print(f"Found {len(trajectories)} features")

# 3. Extract cohorts
cohorts = explainer.extract_cohorts(min_support=50)
print(f"Extracted {len(cohorts)} cohorts")

# 4. Get feature profiles
profiles = explainer.get_feature_profiles()
for name, prof in profiles.items():
    print(f"{name}: {prof['behavior_signature']}")

# 5. Geographic analysis
sample_to_region = pd.read_csv("mapping.csv", index_col="sample_id")

regions = explainer.aggregate_by_region(
    geodata_path="boundaries.geojson",
    sample_region_mapping=sample_to_region,
)

print("Top regions by impact:")
for i, (region, metrics) in enumerate(list(regions.items())[:5], 1):
    print(f"{i}. {region}: impact={metrics['total_impact']:.2f}")

# 6. Visualize
fig = explainer.plot_geographic_impact(
    geodata_path="boundaries.geojson",
    sample_region_mapping=sample_to_region,
    output_path="impact_map.html",
)

# 7. Export
explainer.to_csv("results/")
explainer.to_json("results/pathfinder.json")
```

---

## Next Steps (Phases 5-6)

1. **Phase 5: Export & Integration**
   - Integrate all engines into ModelExplainer
   - Implement to_dict(), to_json(), to_csv() export
   - Integration tests with real XGBoost model

2. **Phase 6: Documentation & Polish**
   - Comprehensive docstrings
   - Example Jupyter notebooks
   - API reference documentation

---

## Design Principles

1. ✅ **Generalized**: No hardcoded data-specific references
2. ✅ **SHAP-like Interface**: Familiar API for users of SHAP package
3. ✅ **Configurable**: All thresholds and parameters customizable
4. ✅ **Layered**: Extract → Trajectory → Enrichment → Visualization
5. ✅ **Impact-Driven**: Geographic ranking by prediction impact, not frequency
6. ✅ **Exportable**: JSON, CSV, Dict formats for integration
7. ✅ **XGBoost-Specific** (with future extensibility)

---

## License

(Specify as needed)

---

## Contributing

Contributions welcome! Key areas for enhancement:
- Support for LightGBM, CatBoost
- Advanced Shapley value metrics
- Model drift tracking
- Interactive dashboard integration
