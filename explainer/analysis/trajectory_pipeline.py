"""Reusable training + trajectory analysis pipeline.

This module exposes a single helper that mirrors the notebook workflow:
1) create ExplainerCallback
2) train Classification with best hyperparameters
3) build trajectory summaries/json artifacts

No visualization is performed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from ..callbacks import ExplainerCallback
from .trajectory_pipeline_config import TrajectoryPipelineConfig
from .training_process_trajectories import TrajectorySummarizer
from predictor.classification_class import Classification


def run_classification_trajectory_pipeline(
    *,
    y: pd.DataFrame,
    data_all: pd.DataFrame,
    config: TrajectoryPipelineConfig,
) -> Tuple[TrajectorySummarizer, Dict[str, Any]]:
    """Run classification and trajectory analysis without plotting.

    Args:
        y: Target dataframe with index aligned to ``data_all``.
        data_all: Feature dataframe used for training.
        config: Pipeline settings container from ``TrajectoryPipelineConfig``.

    Returns:
        (summary_traj, artifacts) where:
            - summary_traj is the initialized ``TrajectorySummarizer`` instance
            - artifacts is a dict containing generated JSON payloads
    """
    explain = ExplainerCallback(
        X_train=data_all.to_numpy(),
        shap_sample_size=config.shap_sample_size,
        shap_interval=config.shap_interval,
        shap_epsilon=config.shap_epsilon,
        shap_interval_min=config.shap_interval_min,
        shap_interval_max=config.shap_interval_max,
    )

    classification = Classification(
        y=y,
        data_all=data_all,
        type_target=config.type_target,
        device=config.device,
        verbose=config.verbose,
        random_state=config.random_state,
        cross_country=config.cross_country,
        sampling=config.sampling,
        sampling_strategy=config.sampling_strategy,
        callbacks=[explain],
    )

    model = classification.xgbclassification_best_model(config.best_hyperparams_path)
    predictions = classification.predictions(model)
    performance = classification.perf_ind_classification(predictions)

    summary_traj = TrajectorySummarizer(
        explain=explain,
        classification=classification,
        save=config.save_artifacts,
        path=config.output_path,
        change_point_epsilon=config.change_point_epsilon,
    )

    artifacts: Dict[str, Any] = {
        "performance": performance,
    }

    if config.generate_shap_llm_summary:
        shap_llm_summary_json = summary_traj.generate_shap_llm_summary_json(
            save=config.save_artifacts,
            top_k_features=config.llm_top_k_features,
            milestone_count=config.llm_milestone_count,
        )
        artifacts["shap_llm_summary_json"] = shap_llm_summary_json

    if config.generate_feature_profiles_llm:
        feature_profiles_json = summary_traj.generate_llm_optimized_feature_profiles(
            save=config.save_artifacts,
            top_k_features=config.feature_profiles_top_k,
            readable_feature_names=True,
        )
        artifacts["feature_profiles_json"] = feature_profiles_json

    return summary_traj, artifacts
