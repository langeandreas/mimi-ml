"""Configuration constants and defaults for the analysis app."""

from pathlib import Path

from explainer.analysis.trajectory_pipeline_config import TrajectoryPipelineConfig

# Compute project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent

# Define data paths as absolute paths
DEFAULT_JSON_PATH = str(PROJECT_ROOT / "data" / "results" / "trajectory_shap.json")
DEFAULT_TREE_JSON_PATH = str(PROJECT_ROOT / "data" / "results" / "trajectory_tree.json")

Y_OPTIONS = {
    "Mean Absolute SHAP": "mean_abs_shap",
    "High Value Feature Impact": "high_value_feature_impact",
    "Low Value Feature Impact": "low_value_feature_impact",
}

DEFAULT_APP_CONFIG = {
    "shap_json_path": DEFAULT_JSON_PATH,
    "tree_json_path": DEFAULT_TREE_JSON_PATH,
    "y_label": "Mean Absolute SHAP",
    "top_k": 8,
    "chat_execution_mode": "graph",
    "chat_show_agent_room": True,
    "model_name": "llama3.1:8b",
    "temperature": 0.0,
}

DEFAULT_TRAJECTORY_PIPELINE_CONFIG = TrajectoryPipelineConfig(
    type_target="overall_mar",
    best_hyperparams_path=str(PROJECT_ROOT / "data" / "results" / "besthyper_overall_mar_LKA_undersampling_0.1_xgboost.csv"),
    country_iso="LKA",
    features_path=str(PROJECT_ROOT / "data" / "lka" / "new_features_lka.csv"),
    targets_path=str(PROJECT_ROOT / "data" / "lka" / "ML_targets_lka.csv"),
    model_name="xgboost",
)
