"""
Analysis engines for xgb-pathfinder.

Modules:
--------
- trajectory_engine: Velocity, acceleration, FCI metrics
- tree_engine: Tree structure analysis and rule extraction
- shap_engine: SHAP value computation and analysis
- cohort_engine: Cohort extraction and enrichment
- feature_analyzer: Feature-level profiles and statistics
- context_analyzer: Decision path context and scope→decider graphs
- utils: Shared utilities (tree traversal, rule evaluation, etc.)
"""

from .tree_engine import TreeEngine
from .trajectory_engine import TrajectoryEngine
from .shap_engine import ShapEngine
from .cohort_engine import CohortEngine
from .feature_analyzer import FeatureAnalyzer
from .context_analyzer import ContextAnalyzer

__all__ = [
	"TreeEngine",
	"TrajectoryEngine",
	"ShapEngine",
	"CohortEngine",
	"FeatureAnalyzer",
	"ContextAnalyzer",
]
