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
from .tree_analyzer import TreeAnalyzer

__all__ = [
    "TrajectorySummarizer",
    "TrajectoryPipelineConfig",
    "run_classification_trajectory_pipeline",
    "TreeAnalyzer",
    "ShapAnalyzer",
    "FeatureProfilesAnalyzer",
    "round_float",
    "expand_shap_step",
    "build_phase_summaries",
    "substitute_feature_names",
    "generate_output_file",
    "generate_output_dir",
]
