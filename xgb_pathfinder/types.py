"""
Data models and type definitions for xgb-pathfinder.

Uses TypedDict for lightweight, runtime-compatible type hints.
All metrics are designed to be serializable to JSON/CSV.
"""

from typing import TypedDict, List, Dict, Optional, Any, Tuple
import numpy as np


# ============================================================================
# Metric & Trajectory Types
# ============================================================================

class TrajectoryMetric(TypedDict):
    """Per-iteration trajectory metric for a single feature."""
    iteration: int
    velocity: float  # Gain at this iteration (from XGBoost)
    acceleration: float  # Change in velocity (ΔGain)
    fci: float  # Feature Confusion Index = cover / split_count
    cumulative_gain: float  # Running sum of gains
    hessian_mean: float  # Hessian mean (HVI signal)


class FeatureTrajectory(TypedDict):
    """Complete trajectory for a feature across all training iterations."""
    feature_name: str
    feature_index: int
    metrics: List[TrajectoryMetric]
    peak_iteration: int
    peak_velocity: float
    total_gain: float
    stability_class: str  # "CORE_DRIVER", "LATE_LEARNER", "OUTLIER_SPECIALIST", "UNSTABLE", "STABLE_BASELINE"
    behavior_signature: str  # Same as stability_class


# ============================================================================
# Decision Rule & Cohort Types
# ============================================================================

class SplitCondition(TypedDict):
    """A single condition in a decision rule (feature < threshold or feature >= threshold)."""
    feature_name: str
    feature_index: int
    threshold: float
    operator: str  # "<" or ">="
    direction: int  # 0 (left/false) or 1 (right/true)


class DecisionRule(TypedDict):
    """Complete decision rule: root-to-leaf path conditions."""
    rule_id: str  # Unique identifier
    conditions: List[SplitCondition]  # List of AND-ed conditions
    leaf_value: float  # Prediction value at leaf
    tree_index: int  # Which tree this rule came from
    support_count: Optional[int]  # How many samples satisfy this rule (filled after assignment)


class CohortRecord(TypedDict):
    """Aggregated cohort: group of samples sharing a decision pattern."""
    cohort_id: str  # Unique identifier
    scope_signature: str  # AND-joined scope conditions (scope features)
    decider_feature_name: str  # Primary decider feature
    decider_feature_index: int
    support_count: int  # Number of samples in cohort
    risk_rate: float  # Empirical positive rate (0-1)
    mean_confidence: float  # Model's mean predicted probability
    late_decider_rate: float  # Fraction of paths where decider is late in tree
    decision_impact: float  # Combined impact score (late_decider_rate * impact_magnitude)
    impact_magnitude_ratio: float  # Leaf impact relative to global mean
    leaf_values: List[float]  # Leaf values when decider is present
    top_features_by_shap: List[Tuple[str, float]]  # Top features by SHAP in this cohort
    contributing_rules: List[str]  # IDs of rules that form this cohort


# ============================================================================
# Feature Profile Types
# ============================================================================

class DecisionImpactProfile(TypedDict):
    """SHAP-based decision impact for a feature."""
    mean_abs_shap: float  # Average absolute SHAP value
    high_value_effect: float  # SHAP effect for high feature values (tail)
    low_value_effect: float  # SHAP effect for low feature values (tail)
    total_swing: float  # |high_value_effect| + |low_value_effect|
    interpretation: str  # Human-readable summary


class FeatureInteraction(TypedDict):
    """Co-occurrence pattern between two features."""
    co_feature_name: str
    co_feature_index: int
    co_occurrence_count: int  # How many paths have both features
    co_occurrence_rate: float  # Proportion of paths with this feature that also have co_feature


class FeatureProfile(TypedDict):
    """Comprehensive per-feature analysis."""
    feature_name: str
    feature_index: int
    importance_trajectory: FeatureTrajectory  # Trajectory over iterations
    decision_impact: DecisionImpactProfile  # SHAP-based impact
    feature_interactions: List[FeatureInteraction]  # Top co-occurring features
    path_statistics: Dict[str, float]  # late_decider_rate, scope_entropy, etc.
    behavior_signature: str  # CORE_DRIVER, LATE_LEARNER, etc.


# ============================================================================
# Geographic & Region Types
# ============================================================================

class RegionImpactMetric(TypedDict):
    """Impact metrics aggregated at the region level."""
    region_name: str
    region_id: Optional[str]  # Admin level identifier
    cohort_count: int  # Number of cohorts in region
    total_impact: float  # Sum of impact across cohorts
    avg_impact_per_cohort: float  # Mean impact per cohort
    avg_impact_per_sample: float  # Impact normalized by population
    total_support: int  # Total samples in region
    top_cohorts: List[Dict[str, Any]]  # Top 5 cohorts by impact


# ============================================================================
# Model State & Summary Types
# ============================================================================

class ModelExplainerState(TypedDict, total=False):
    """Complete state of a fitted ModelExplainer."""
    booster: Any  # XGBoost booster (xgboost.Booster)
    X_train: np.ndarray  # Training feature matrix
    y_train: np.ndarray  # Training target
    feature_names: List[str]  # Feature name list
    trajectory_metrics: Dict[str, FeatureTrajectory]  # All trajectory metrics
    feature_profiles: Dict[str, FeatureProfile]  # All feature profiles
    cohorts: List[CohortRecord]  # Extracted cohorts
    decision_rules: List[DecisionRule]  # All decision rules
    shap_values: Optional[np.ndarray]  # SHAP values from final model
    sample_to_cohort_mapping: Optional[Dict[int, List[str]]]  # Sample index -> cohort IDs
    geographic_aggregation: Optional[Dict[str, RegionImpactMetric]]  # Region -> impact metrics


# ============================================================================
# Configuration Types
# ============================================================================

class PathfinderConfig(TypedDict, total=False):
    """Configuration parameters for xgb-pathfinder."""
    # Trajectory computation
    acceleration_threshold: float  # Min |acceleration| to trigger behavior signature change
    early_acceleration_threshold: float  # Min acceleration in early iterations
    late_acceleration_threshold: float  # Min acceleration in late iterations
    stability_threshold: float  # Std dev threshold to classify as UNSTABLE
    
    # Cohort extraction
    min_cohort_support: int  # Minimum samples in cohort
    min_paths_per_cohort: int  # Minimum distinct paths to form cohort
    late_decider_percentile: float  # Percentile threshold for late-decider classification
    
    # Geographic aggregation
    admin_level: int  # Which administrative level (1=state, 2=district, etc.)
    impact_metric: str  # "magnitude", "frequency", "magnitude_x_frequency"
    
    # Output & export
    top_k_features: int  # Top K features to include in profiles/exports
    top_k_interactions: int  # Top K feature interactions to include
    json_indent: Optional[int]  # JSON formatting (None for compact, 2/4 for pretty)
