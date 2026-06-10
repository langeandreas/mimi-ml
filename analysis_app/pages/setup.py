"""Setup page for the Streamlit app."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import streamlit as st

from analysis_app.config import DEFAULT_TRAJECTORY_PIPELINE_CONFIG
from analysis_app.utils import get_trajectory_pipeline_config
from explainer.analysis.trajectory_pipeline import run_classification_trajectory_pipeline
from predictor.sri_lanka_functions import lka_data_preparation, THRESHOLDS, TargetTypeLKA


def run_setup() -> Dict[str, Any]:
    """Execute trajectory analysis and store results in session state."""
    config = get_trajectory_pipeline_config()
    try:
        y, data_all = lka_data_preparation(
            new_features_fp=config["features_path"],
            targets_fp=config["targets_path"],
            t=config["type_target"],
        )
        summary_traj, artifacts = run_classification_trajectory_pipeline(
            config=config, y=y, data_all=data_all
        )
        # Store in session state for other pages to access
        st.session_state.summary_traj = summary_traj
        st.session_state.artifacts = artifacts
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Failed to load data: {e}",
            "artifacts": {},
            "summary_traj": None,
        }
    return {
        "status": "ok",
        "message": "Training executed. You can now chat with the LLM about the model.",
        "artifacts": artifacts,
        "summary_traj": summary_traj,
    }


def render_setup_page() -> None:
    """Render the Setup page."""
    st.header("Trajectory Analyzer Setup")
    st.caption("Run one-time tasks and adjust shared app configuration.")

    pipeline_config = get_trajectory_pipeline_config()

    if st.button("Run Trajectory Analysis"):
        result = run_setup()
        st.success(result.get("message", "Analysis completed."))
        with st.expander("Result"):
            st.json(result, expanded=False)

    with st.container(border=True) as config_container:
        st.subheader("Trajectory Pipeline Configuration")
        with st.form("trajectory_pipeline_config_form"):
            st.text("General Settings")
            type_target = st.text_input(f"Target Variable, available: {', '.join(THRESHOLDS.keys())}", value=str(pipeline_config["type_target"]))
            st.text(f"Threshold value: {THRESHOLDS.get(type_target, 'N/A')}")
            country_iso = st.text_input("Country ISO", value=str(pipeline_config["country_iso"]))
            features_path = st.text_input(
                "Features dataset path", value=str(pipeline_config["features_path"])
            )
            targets_path = st.text_input(
                "Targets dataset path", value=str(pipeline_config["targets_path"])
            )

            st.divider()
            st.text("Prediction Model Settings")
            model_name = st.text_input("Model Name", value=str(pipeline_config["model_name"]))
            best_hyperparams_path = st.text_input(
                "Model Hyperparameters Path", value=str(pipeline_config["best_hyperparams_path"])
            )
            device = st.text_input("Device (optional)", value=str(pipeline_config["device"]))
            random_state = st.text_input(
                "Random State (default: best trained)", value=str(pipeline_config["random_state"])
            )
            cross_country = st.checkbox("Cross Country", value=bool(pipeline_config["cross_country"]))
            sampling = st.text_input("Sampling (optional)", value=str(pipeline_config["sampling"]))
            sampling_strategy = st.number_input(
                "sampling_strategy (optional float)",
                value=float(pipeline_config["sampling_strategy"]),
                step=0.1,
            )
            verbose = st.checkbox("verbose", value=bool(pipeline_config["verbose"]))

            st.divider()
            st.text("SHAP Trajectory Analysis Settings")
            shap_sample_size = st.number_input(
                "shap_sample_size",
                min_value=1,
                value=int(pipeline_config["shap_sample_size"]),
                step=1,
            )
            shap_interval = st.number_input(
                "shap_interval", min_value=1, value=int(pipeline_config["shap_interval"]), step=1
            )
            shap_epsilon = st.number_input(
                "shap_epsilon (optional float)",
                value=float(pipeline_config["shap_epsilon"]),
                step=0.1,
            )
            shap_interval_min = st.number_input(
                "shap_interval_min",
                min_value=1,
                value=int(pipeline_config["shap_interval_min"]),
                step=1,
            )
            shap_interval_max = st.number_input(
                "shap_interval_max",
                min_value=1,
                value=int(pipeline_config["shap_interval_max"]),
                step=1,
            )

            use_change_point_detection = st.checkbox(
                "use_change_point_detection",
                value=bool(pipeline_config["use_change_point_detection"]),
            )
            change_point_epsilon = st.number_input(
                "change_point_epsilon (optional float)",
                value=float(pipeline_config["change_point_epsilon"]),
                step=0.1,
                label_visibility="collapsed",
                disabled=not use_change_point_detection,
            )

            generate_shap_llm_summary = st.checkbox(
                "Generate condensed SHAP summary for LLM input",
                value=bool(pipeline_config["generate_shap_llm_summary"]),
            )
            st.divider()
            st.text("Output Settings")
            output_path = st.text_input("Output Path", value=str(pipeline_config["output_path"]))
            save_artifacts = st.checkbox(
                "Save Artifacts", value=bool(pipeline_config["save_artifacts"])
            )

            llm_top_k_features = st.number_input(
                "LLM Top K Features",
                min_value=1,
                value=int(pipeline_config["llm_top_k_features"]),
                step=1,
            )
            llm_milestone_count = st.number_input(
                "LLM Milestone Count",
                min_value=1,
                value=int(pipeline_config["llm_milestone_count"]),
                step=1,
            )

            pipeline_saved = st.form_submit_button("Save trajectory pipeline configuration")
            if pipeline_saved:
                st.session_state.trajectory_pipeline_config = {
                    "type_target": type_target,
                    "best_hyperparams_path": best_hyperparams_path,
                    "country_iso": country_iso,
                    "features_path": features_path,
                    "targets_path": targets_path,
                    "model_name": model_name,
                    "device": device,
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
                st.success("Trajectory pipeline configuration saved.")

        if st.button("Reset trajectory pipeline config to defaults"):
            st.session_state.trajectory_pipeline_config = DEFAULT_TRAJECTORY_PIPELINE_CONFIG
            st.rerun()

    with st.expander("Show trajectory pipeline configuration", expanded=False):
        st.json(
            (
                st.session_state.trajectory_pipeline_config
                if "trajectory_pipeline_config" in st.session_state
                else asdict(DEFAULT_TRAJECTORY_PIPELINE_CONFIG)
            ),
            expanded=False,
        )
