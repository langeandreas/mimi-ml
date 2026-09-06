"""Analysis module for trajectory-based model explainability.

This module contains all trajectory analysis, SHAP analysis, and tree analysis code,
organized into specialized analyzers for maintainability and clarity.

Main exports:
    - TrajectorySummarizer: Main orchestrator for all analysis
    - TrajectoryPipelineConfig: Configuration container
    - run_classification_trajectory_pipeline: High-level pipeline function
"""

from .feature_profiles_analyzer import FeatureProfilesAnalyzer
from .shap_analyzer import ShapAnalyzer
from .training_process_trajectories import TrajectorySummarizer
from .trajectory_pipeline import run_classification_trajectory_pipeline
from .trajectory_pipeline_config import TrajectoryPipelineConfig
from .trajectory_utils import (
    round_float,
    expand_shap_step,
    build_phase_summaries,
    substitute_feature_names,
    generate_output_file,
    generate_output_dir,
)
from .behavioral_quadrants import BehavioralQuadrantAnalyzer
from .behavioral_plotter import plot_behavioral_signatures
from .cohort_attribution_analyzer import CohortAttributionAnalyzer
from .decision_context_analyzer import DecisionContextAnalyzer
from .decision_graph import (
    build_decision_narrowing_graph,
    decision_narrowing_to_graphviz,
)
from .feature_story import FeatureStoryBuilder
from .shap_behavior_correlator import ShapBehaviorCorrelator
from .tree_analyzer import TreeAnalyzer
from .tree_walker import walk_tree, collect_split_path_records

__all__ = [
    "TrajectorySummarizer",
    "TrajectoryPipelineConfig",
    "run_classification_trajectory_pipeline",
    "TreeAnalyzer",
    "BehavioralQuadrantAnalyzer",
    "CohortAttributionAnalyzer",
    "DecisionContextAnalyzer",
    "FeatureStoryBuilder",
    "build_decision_narrowing_graph",
    "decision_narrowing_to_graphviz",
    "ShapBehaviorCorrelator",
    "plot_behavioral_signatures",
    "walk_tree",
    "collect_split_path_records",
    "ShapAnalyzer",
    "FeatureProfilesAnalyzer",
    "round_float",
    "expand_shap_step",
    "build_phase_summaries",
    "substitute_feature_names",
    "generate_output_file",
    "generate_output_dir",
]
