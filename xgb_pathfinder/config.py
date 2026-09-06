"""
Default configuration for xgb-pathfinder.

All thresholds and parameters are customizable at instantiation time.
"""

from .types import PathfinderConfig

# Default configuration
DEFAULT_CONFIG: PathfinderConfig = {
    # Trajectory computation thresholds
    "acceleration_threshold": 0.1,  # Min |acceleration| to indicate change in importance
    "early_acceleration_threshold": 0.05,  # Early-phase threshold for behavior classification
    "late_acceleration_threshold": 0.05,  # Late-phase threshold for behavior classification
    "stability_threshold": 0.15,  # Std dev threshold for UNSTABLE classification
    
    # Cohort extraction parameters
    "min_cohort_support": 10,  # At least 10 samples per cohort
    "min_paths_per_cohort": 2,  # At least 2 distinct paths define a cohort
    "late_decider_percentile": 0.75,  # Feature is "late decider" if it appears in ≥75th percentile of path depth
    
    # Geographic aggregation
    "admin_level": 2,  # Default to admin level 2 (district/county equivalent)
    "impact_metric": "magnitude_x_frequency",  # Combine impact magnitude with frequency
    
    # Output & export
    "top_k_features": 15,  # Include top 15 features in reports
    "top_k_interactions": 15,  # Include top 15 feature interactions
    "json_indent": 2,  # Pretty-print JSON with 2-space indent
}


# Behavior signature classification thresholds
BEHAVIOR_SIGNATURE_RULES = {
    "CORE_DRIVER": {
        "description": "Early acceleration → stabilization. Strong early impact.",
        "early_accel": (0.05, float('inf')),  # (min, max) acceleration in early phase
        "late_accel": (-float('inf'), -0.05),  # Stabilizes/decelerates in late phase
    },
    "LATE_LEARNER": {
        "description": "Weak early, strong late. Importance grows over time.",
        "early_accel": (-float('inf'), 0.05),
        "late_accel": (0.05, float('inf')),
    },
    "OUTLIER_SPECIALIST": {
        "description": "Continuous optimization throughout. High volatility.",
        "early_accel": (0.05, float('inf')),
        "late_accel": (0.05, float('inf')),
    },
    "UNSTABLE": {
        "description": "High oscillation. Possible collinearity or interaction effects.",
        "std_dev_threshold": 0.15,  # High variance in velocity
    },
    "STABLE_BASELINE": {
        "description": "Steady, consistent contribution.",
    },
}


# Metric computation constants
TAIL_PERCENTILE = 0.30  # Use bottom 30% and top 30% of SHAP values for tail effects
SHAP_SAMPLE_SIZE_MAX = 1000  # Cap SHAP computation sample size for efficiency
MIN_SAMPLES_FOR_SHAP = 100  # Minimum samples needed to compute SHAP (otherwise use all)


# Feature names constants (for validation/logging)
RESERVED_KEYWORDS = {"feature", "sample", "iteration", "cohort", "rule"}
