"""SHAP Viewer page for the Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import shap

from analysis_app.config import Y_OPTIONS
from analysis_app.utils import get_llm_config, load_shap_json, pick_top_features, shap_json_to_df
from explainer.analysis.trajectory_utils import substitute_feature_names


def _normalize_shap_matrix(shap_values_raw: object) -> np.ndarray:
    """Normalize SHAP outputs to a 2D matrix (samples x features)."""
    if isinstance(shap_values_raw, list):
        shap_matrix = np.asarray(shap_values_raw[-1])
    else:
        shap_matrix = np.asarray(shap_values_raw)

    if shap_matrix.ndim == 1:
        return shap_matrix.reshape(1, -1)
    if shap_matrix.ndim == 2:
        return shap_matrix
    if shap_matrix.ndim == 3:
        return np.asarray(shap_matrix[:, :, -1])
    raise ValueError(f"Unsupported SHAP value shape: {shap_matrix.shape}")


def _render_shap_summary_plot(
    summary_traj, selected_iteration: int, iteration_entry: dict
) -> None:
    """Render SHAP summary plot for a specific iteration.
    
    Args:
        summary_traj: The summary trajectory object.
        selected_iteration: The selected iteration number.
        iteration_entry: The entry containing SHAP values for the iteration.
    """

    # Normalize SHAP values into a 2D sample x feature matrix.
    shap_values = _normalize_shap_matrix(iteration_entry["shap_values"])
    feature_names = summary_traj.feature_names

    # SHAP values in the callback are computed on explain.X_train, not on classification.train_test["X_train"].
    callback_matrix = np.asarray(summary_traj.explain.X_train)
    if callback_matrix.ndim == 1:
        callback_matrix = callback_matrix.reshape(1, -1)

    n_rows = min(shap_values.shape[0], callback_matrix.shape[0])
    n_features = min(shap_values.shape[1], callback_matrix.shape[1], len(feature_names))
    if n_rows == 0 or n_features == 0:
        st.warning("Could not align SHAP values with callback sample data for plotting.")
        return

    # Align both matrices to the same shape so feature values and SHAP attributions match.
    shap_values = shap_values[:n_rows, :n_features]
    X_sample = pd.DataFrame(
        callback_matrix[:n_rows, :n_features],
        columns=feature_names[:n_features],
    )

    # Replace technical feature names with readable explanations where available.
    rename_df = pd.DataFrame({"feature_index": feature_names[:n_features]})
    rename_df = substitute_feature_names(
        rename_df,
        feature_index_col="feature_index",
        feature_names=feature_names,
    )
    X_sample.columns = rename_df["feature_index"].astype(str).tolist()

    # Create SHAP summary plot with larger canvas and smaller tick labels.
    fig_shap = plt.figure(figsize=(15, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=15,
        plot_size=(15, 8),
    )
    ax = plt.gca()
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    plt.title(
        f"SHAP Summary Plot - Iteration {selected_iteration}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    st.pyplot(fig_shap, width="content")

    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean |SHAP|", f"{np.abs(shap_values).mean():.4f}")
    with col2:
        st.metric("Max |SHAP|", f"{np.abs(shap_values).max():.4f}")
    with col3:
        st.metric("Samples", shap_values.shape[0])


def render_shap_viewer_page() -> None:
    """Render the SHAP Viewer page."""
    st.header("SHAP Trajectory Viewer")
    st.caption("Visualize SHAP feature trajectories across training iterations.")

    config = get_llm_config()

    st.subheader("Trajectory Plot")
    st.caption("Configure and view SHAP feature trajectories.")

    json_path = st.text_input(
        "(Optional) Custom Path to SHAP JSON",
        value=str(config["shap_json_path"]),
        key="shap_json_path_input",
    )
    y_default = str(config["y_label"])
    y_options = list(Y_OPTIONS.keys())
    y_index = y_options.index(y_default) if y_default in y_options else 0
    controls_col1, controls_col2 = st.columns([2, 1])
    with controls_col1:
        y_label = st.selectbox(
            "Y-axis metric",
            options=y_options,
            index=y_index,
            key="shap_y_label_input",
        )
    with controls_col2:
        top_k = st.slider(
            "Top features (by max mean_abs_shap)",
            min_value=1,
            max_value=20,
            value=int(config["top_k"]),
            key="shap_top_k_input",
        )

    config["shap_json_path"] = json_path
    config["y_label"] = y_label
    config["top_k"] = top_k

    shap_json = load_shap_json(json_path)

    df = shap_json_to_df(shap_json)
    if df.empty:
        st.warning("No SHAP trajectory rows found in the provided JSON.")
        return

    selected_features = pick_top_features(df, top_k)
    plot_df = df[df["feature_index"].isin(selected_features)].copy()

    if plot_df.empty:
        st.warning("No matching features to plot.")
        return

    feature_names_for_plot = []
    if "summary_traj" in st.session_state and st.session_state.summary_traj is not None:
        feature_names_for_plot = list(getattr(st.session_state.summary_traj, "feature_names", []))

    plot_df = substitute_feature_names(
        plot_df,
        feature_index_col="feature_index",
        feature_names=feature_names_for_plot,
    )

    y_col = Y_OPTIONS[y_label]

    fig, ax = plt.subplots(figsize=(12, 6))
    for feature_name in sorted(plot_df["feature_index"].unique()):
        feature_data = plot_df[plot_df["feature_index"] == feature_name].sort_values("iteration")
        ax.plot(
            feature_data["iteration"],
            feature_data[y_col],
            label=feature_name,
            alpha=0.8,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.set_title(f"SHAP Trajectory ({y_label})")
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    st.pyplot(fig, width="content")

    with st.expander("Show plotted data"):
        st.dataframe(plot_df.sort_values(["feature_index", "iteration"]), width="content")

    # === SHAP Summary Plot at Specific Iteration ===
    st.divider()
    st.subheader("SHAP Summary Plot at Iteration")
    st.caption("View traditional SHAP summary plot for a specific training iteration.")

    if "summary_traj" in st.session_state and st.session_state.summary_traj is not None:
        summary_traj = st.session_state.summary_traj

        # Get available iterations from callback (only those with SHAP values)
        shap_iterations = sorted(
            [int(entry["iteration"]) for entry in summary_traj.explain.shap_values_history]
        )
        if shap_iterations:
            # Create mapping of iteration to entry for quick lookup
            iteration_to_entry = {
                int(entry["iteration"]): entry
                for entry in summary_traj.explain.shap_values_history
            }

            selected_iteration = st.select_slider(
                "Select iteration",
                options=shap_iterations,
                value=shap_iterations[-1],
                key="shap_summary_iteration_slider",
                format_func=lambda iteration: f"Iter {iteration}",
            )


            # Get the entry for the selected iteration
            iteration_entry = iteration_to_entry.get(selected_iteration)

            if iteration_entry is not None:
                _render_shap_summary_plot(summary_traj, selected_iteration, iteration_entry)
        else:
            st.info("No SHAP data available. Run the Setup page first to generate SHAP values.")
    else:
        st.info("No trained model available. Please run the trajectory analysis in the **Setup** page first.")
