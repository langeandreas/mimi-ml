"""Configuration models for the trajectory pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd


@dataclass
class TrajectoryPipelineConfig:
    """All tunable parameters for classification + trajectory pipeline."""

    type_target: str
    best_hyperparams_path: str
    country_iso: str
    features_path: str
    targets_path: str
    model_name: str

    device: Optional[str] = 'cuda'
    random_state: int = 42
    cross_country: bool = False
    sampling: Optional[str] = 'undersampling'
    sampling_strategy: Optional[float] = 0.5
    verbose: bool = False

    shap_sample_size: int = 100
    shap_interval: int = 20
    shap_epsilon: Optional[float] = 0.1
    shap_interval_min: int = 5
    shap_interval_max: int = 100

    use_change_point_detection: bool = False
    change_point_epsilon: Optional[float] = 0.1
    output_path: str = "data/results/"
    save_artifacts: bool = True

    generate_shap_llm_summary: bool = True
    llm_top_k_features: int = 10
    llm_milestone_count: int = 6

    generate_feature_profiles_llm: bool = True
    feature_profiles_top_k: int = 15


    def __getitem__(self, key: str) -> Optional[Any]:
        return getattr(self, key)
    
    def __getattr__(self, name: str) -> Optional[Any]:
        return super().__getattribute__(name)
