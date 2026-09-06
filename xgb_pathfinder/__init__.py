"""
xgb-pathfinder: Generalized XGBoost model explanation package.

Provides trajectory metrics, cohort analysis, and geographic visualization
of model decisions and impact.

Main API:
---------
from xgb_pathfinder import ModelExplainer

explainer = ModelExplainer(booster, X_train, y_train, feature_names)
trajectories = explainer.compute_trajectory_metrics()
cohorts = explainer.extract_cohorts(min_support=20)
profiles = explainer.get_feature_profiles()
data = explainer.to_dict()
"""

from .explainer import ModelExplainer
from .types import (
    TrajectoryMetric,
    FeatureTrajectory,
    CohortRecord,
    FeatureProfile,
    DecisionRule,
    RegionImpactMetric,
    PathfinderConfig,
    ModelExplainerState,
)
from .config import DEFAULT_CONFIG

__version__ = "0.1.0"
__author__ = "xgb-pathfinder Contributors"

__all__ = [
    "ModelExplainer",
    "TrajectoryMetric",
    "FeatureTrajectory",
    "CohortRecord",
    "FeatureProfile",
    "DecisionRule",
    "RegionImpactMetric",
    "PathfinderConfig",
    "ModelExplainerState",
    "DEFAULT_CONFIG",
]
