"""Setup page for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib

import streamlit as st

from analysis_app.config import DEFAULT_TRAJECTORY_PIPELINE_CONFIG
from analysis_app.utils import get_trajectory_pipeline_config
from explainer.analysis.trajectory_pipeline_config import TrajectoryPipelineConfig
from explainer.analysis.trajectory_pipeline import run_classification_trajectory_pipeline
from predictor.sri_lanka_functions import lka_data_preparation, THRESHOLDS, TargetTypeLKA


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ANALYSES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "analyses"


def _ensure_configs_dir() -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIGS_DIR


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return asdict(config)


def _dict_to_pipeline_config(data: Dict[str, Any]) -> TrajectoryPipelineConfig:
    base = asdict(DEFAULT_TRAJECTORY_PIPELINE_CONFIG)
    base.update(data)
    return TrajectoryPipelineConfig(**base)


def _list_saved_config_names() -> list[str]:
    configs_dir = _ensure_configs_dir()
    return sorted(path.stem for path in configs_dir.glob("*.json"))


def _load_saved_config(config_name: str) -> TrajectoryPipelineConfig:
    config_path = _ensure_configs_dir() / f"{config_name}.json"
    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return _dict_to_pipeline_config(payload)


def _save_config(config_name: str, config: Dict[str, Any]) -> Path:
    safe_name = config_name.strip().replace(" ", "_")
    if not safe_name:
        raise ValueError("Config name cannot be empty.")

    config_path = _ensure_configs_dir() / f"{safe_name}.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    return config_path


def _ensure_analyses_dir() -> Path:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSES_DIR


def _save_analysis(summary_traj: Any) -> Path:
    analyses_dir = _ensure_analyses_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = analyses_dir / f"analysis_{timestamp}.joblib"
    joblib.dump(summary_traj, out_path)
    return out_path


def _list_saved_analyses() -> list[str]:
    analyses_dir = _ensure_analyses_dir()
    return sorted(
        (p.stem for p in analyses_dir.glob("analysis_*.joblib")),
        reverse=True,  # most recent first
    )


def _load_analysis(stem: str) -> Any:
    analyses_dir = _ensure_analyses_dir()
    return joblib.load(analyses_dir / f"{stem}.joblib")


def run_setup() -> Dict[str, Any]:
    """Execute trajectory analysis and store results in session state."""
    config_data = _config_to_dict(get_trajectory_pipeline_config())
    config = _dict_to_pipeline_config(config_data)

    try:
        y, data_all = lka_data_preparation(
            new_features_fp=config.features_path,
            targets_fp=config.targets_path,
            t=config.type_target,
        )
        data_all.drop(columns=["r3q", "rfh_avg", "food_exp_share"], inplace=True)
        summary_traj, artifacts = run_classification_trajectory_pipeline(
            config=config, y=y, data_all=data_all
        )
        # Store in session state for other pages to access
        st.session_state.summary_traj = summary_traj
        st.session_state.artifacts = artifacts
        # Persist summary_traj to disk so it can be reloaded later
        _save_analysis(summary_traj)
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Failed to load data: {e}",
            "artifacts": {},
            "summary_traj": None,
        }
    return {
        "status": "ok",
        "message": "Training executed. Go to the different pages for insights",
        "artifacts": artifacts,
        "summary_traj": summary_traj,
    }


def render_setup_page() -> None:
    """Render the Setup page."""
    st.header("Trajectory Analyzer Setup")
    st.caption("Run one-time tasks and adjust shared app configuration.")

    saved_config_names = _list_saved_config_names()
    selected_saved_config = st.selectbox(
        "Saved configurations",
        options=["(current session)", *saved_config_names],
        index=0,
    )
    if selected_saved_config != "(current session)" and st.button("Load selected configuration"):
        st.session_state.trajectory_pipeline_config = _load_saved_config(selected_saved_config)
        st.success(f"Loaded configuration '{selected_saved_config}'.")
        st.rerun()

    saved_analyses = _list_saved_analyses()
    selected_analysis = st.selectbox(
        "Recent analyses",
        options=["(none)", *saved_analyses],
        index=0,
        format_func=lambda s: s if s == "(none)" else s.replace("analysis_", "").replace("_", " "),
    )
    if selected_analysis != "(none)" and st.button("Load recent analysis"):
        with st.spinner("Loading analysis..."):
            st.session_state.summary_traj = _load_analysis(selected_analysis)
        st.success(f"Loaded analysis '{selected_analysis}'.")

    pipeline_config = get_trajectory_pipeline_config()
    pipeline_config_data = _config_to_dict(pipeline_config)

    if st.button("Run Trajectory Analysis"):
        result = run_setup()
        st.success(result.get("message", "Analysis completed."))
        performance = result["artifacts"].get("performance")
        if isinstance(performance, dict):
            cols = st.columns(len(performance))
            for col, (metric_name, metric_value) in zip(cols, performance.items()):
                col.metric(label=metric_name.replace("_", " ").title(), value=metric_value)
        else:
            st.metric(label="Training performance", value=performance or "N/A")
        with st.expander("Raw results"):
            st.json(result, expanded=False)

    with st.container(border=True) as config_container:
        st.subheader("Trajectory Pipeline Configuration")
        with st.form("trajectory_pipeline_config_form"):
            st.text("General Settings")
            type_target = st.text_input(f"Target Variable, available: {', '.join(THRESHOLDS.keys())}", value=str(pipeline_config_data["type_target"]))
            st.text(f"Threshold value: {THRESHOLDS.get(type_target, 'N/A')}")
            country_iso = st.text_input("Country ISO", value=str(pipeline_config_data["country_iso"]))
            features_path = st.text_input(
                "Features dataset path", value=str(pipeline_config_data["features_path"])
            )
            targets_path = st.text_input(
                "Targets dataset path", value=str(pipeline_config_data["targets_path"])
            )

            st.divider()
            st.text("Prediction Model Settings")
            model_name = st.text_input("Model Name", value=str(pipeline_config_data["model_name"]))
            best_hyperparams_path = st.text_input(
                "Model Hyperparameters Path", value=str(pipeline_config_data["best_hyperparams_path"])
            )
            device = st.text_input("Device (optional)", value=str(pipeline_config_data["device"]))
            use_best_random_state = st.checkbox(
                "Use best trained random state for classification",
                value=bool(pipeline_config_data["use_best_random_state"]),
            )
            random_state = st.number_input(
                "Custom Random State", value=int(pipeline_config_data["random_state"])
            )
            cross_country = st.checkbox("Cross Country", value=bool(pipeline_config_data["cross_country"]))
            sampling = st.text_input("Sampling (optional)", value=str(pipeline_config_data["sampling"]))
            sampling_strategy = st.number_input(
                "sampling_strategy (optional float)",
                value=float(pipeline_config_data["sampling_strategy"]),
                step=0.1,
            )
            verbose = st.checkbox("verbose", value=bool(pipeline_config_data["verbose"]))

            st.divider()
            st.text("SHAP Trajectory Analysis Settings")
            shap_sample_size = st.number_input(
                "shap_sample_size",
                min_value=1,
                value=int(pipeline_config_data["shap_sample_size"]),
                step=1,
            )
            shap_interval = st.number_input(
                "shap_interval", min_value=1, value=int(pipeline_config_data["shap_interval"]), step=1
            )
            shap_epsilon = st.number_input(
                "shap_epsilon (optional float)",
                value=float(pipeline_config_data["shap_epsilon"]),
                step=0.1,
            )
            shap_interval_min = st.number_input(
                "shap_interval_min",
                min_value=1,
                value=int(pipeline_config_data["shap_interval_min"]),
                step=1,
            )
            shap_interval_max = st.number_input(
                "shap_interval_max",
                min_value=1,
                value=int(pipeline_config_data["shap_interval_max"]),
                step=1,
            )

            use_change_point_detection = st.checkbox(
                "use_change_point_detection",
                value=bool(pipeline_config_data["use_change_point_detection"]),
            )
            change_point_epsilon = st.number_input(
                "change_point_epsilon (optional float)",
                value=float(pipeline_config_data["change_point_epsilon"]),
                step=0.1,
                label_visibility="collapsed",
                disabled=not use_change_point_detection,
            )

            generate_shap_llm_summary = st.checkbox(
                "Generate condensed SHAP summary for LLM input",
                value=bool(pipeline_config_data["generate_shap_llm_summary"]),
            )
            st.divider()
            st.text("Output Settings")
            output_path = st.text_input("Output Path", value=str(pipeline_config_data["output_path"]))
            save_artifacts = st.checkbox(
                "Save Artifacts", value=bool(pipeline_config_data["save_artifacts"])
            )

            llm_top_k_features = st.number_input(
                "LLM Top K Features",
                min_value=1,
                value=int(pipeline_config_data["llm_top_k_features"]),
                step=1,
            )
            llm_milestone_count = st.number_input(
                "LLM Milestone Count",
                min_value=1,
                value=int(pipeline_config_data["llm_milestone_count"]),
                step=1,
            )

            config_name = st.text_input("Config name", value="trajectory_pipeline")

            pipeline_saved = st.form_submit_button("Save trajectory pipeline configuration")
            if pipeline_saved:
                config_payload = {
                    "type_target": type_target,
                    "best_hyperparams_path": best_hyperparams_path,
                    "country_iso": country_iso,
                    "features_path": features_path,
                    "targets_path": targets_path,
                    "model_name": model_name,
                    "device": device,
                    "use_best_random_state": use_best_random_state,
                    "random_state": random_state,
                    "cross_country": cross_country,
                    "sampling": sampling,
                    "sampling_strategy": sampling_strategy,
                    "verbose": verbose,
                    "shap_sample_size": int(shap_sample_size),
                    "shap_interval": int(shap_interval),
                    "shap_epsilon": shap_epsilon,
                    "shap_interval_min": int(shap_interval_min),
                    "shap_interval_max": int(shap_interval_max),
                    "use_change_point_detection": use_change_point_detection,
                    "change_point_epsilon": change_point_epsilon,
                    "output_path": output_path,
                    "save_artifacts": save_artifacts,
                    "generate_shap_llm_summary": generate_shap_llm_summary,
                    "llm_top_k_features": int(llm_top_k_features),
                    "llm_milestone_count": int(llm_milestone_count),
                }

                st.session_state.trajectory_pipeline_config = _dict_to_pipeline_config(config_payload)
                saved_path = _save_config(config_name, config_payload)

                st.success(f"Trajectory pipeline configuration saved to {saved_path}.")
                

        if st.button("Reset trajectory pipeline config to defaults"):
            st.session_state.trajectory_pipeline_config = DEFAULT_TRAJECTORY_PIPELINE_CONFIG
            st.rerun()

    with st.expander("Show trajectory pipeline configuration", expanded=False):
        st.json(
            _config_to_dict(get_trajectory_pipeline_config()),
            expanded=False,
        )
