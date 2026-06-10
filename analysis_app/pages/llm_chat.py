"""LLM Chat page for the Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import streamlit as st

from explainer.agents import (
    ask_explainer_graph,
    ask_generalist_agent,
    run_explainer_graph,
)
from analysis_app.utils import format_chat_question, get_llm_config


def render_llm_chat_page() -> None:
    """Render the LLM Chat page."""
    st.header("Trajectory LLM Chatbot")
    st.caption("Ask questions about SHAP and tree trajectories using the model from the Setup page.")

    # Check if summary_traj is available
    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.warning(
            "No trained model available. Please run the trajectory analysis in the **Setup** page first."
        )
        return

    summary_traj = st.session_state.summary_traj
    config = get_llm_config()

    # Generate JSONs from summary_traj on-the-fly
    with st.spinner("Preparing model data..."):
        shap_json = summary_traj.generate_shap_json(save=False, readable_feature_names=True)
        tree_json = summary_traj.generate_tree_json(save=False)
        shap_llm_summary_json = summary_traj.generate_shap_llm_summary_json(
            save=False,
            top_k_features=10,
            milestone_count=6,
        )
        feature_profiles_json = summary_traj.generate_llm_optimized_feature_profiles(
            save=False,
            top_k_features=15,
            readable_feature_names=True,
        )

    mode_options = ["graph", "agent"]
    mode_default = str(config["chat_execution_mode"])
    mode_index = mode_options.index(mode_default) if mode_default in mode_options else 0
    execution_mode = st.sidebar.selectbox(
        "Chat execution mode",
        options=mode_options,
        index=mode_index,
        key="chat_execution_mode_input",
    )
    show_agent_room = st.sidebar.checkbox(
        "Show graph agent room",
        value=bool(config.get("chat_show_agent_room", True)),
        key="chat_show_agent_room_input",
    )
    use_feature_profiles = st.sidebar.checkbox(
        "Use feature-centric profiles",
        value=True,
        help="Use the feature-profiles format for more efficient LLM processing",
        key="chat_use_feature_profiles_input",
    )
    model_name = st.sidebar.text_input("Ollama model", value=str(config["model_name"]), key="chat_model_name_input")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(config["temperature"]),
        step=0.1,
        key="chat_temperature_input",
    )

    config["chat_execution_mode"] = execution_mode
    config["chat_show_agent_room"] = show_agent_room
    config["model_name"] = model_name
    config["temperature"] = temperature

    with st.expander("Show model context available to the LLM", expanded=False):
        st.caption("Source: Generated from trained model")
        tab1, tab2, tab3, tab4 = st.tabs(["SHAP JSON", "LLM Summary", "Feature Profiles", "Tree JSON"])
        with tab1:
            st.json(shap_json, expanded=False)
        with tab2:
            st.json(shap_llm_summary_json, expanded=False)
        with tab3:
            st.json(feature_profiles_json, expanded=False)
        with tab4:
            st.json(tree_json, expanded=False)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about feature drift, split behavior, or specific iterations...")
    if user_prompt is None:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        room_placeholder = st.empty()
        room_events: List[Dict[str, Any]] = []
        answer = ""

        def _render_room() -> None:
            if not show_agent_room:
                return
            lines = []
            for event in room_events:
                icon = "[RUN]" if event.get("phase") == "start" else "[DONE]"
                node = event.get("node", "unknown")
                message = event.get("message", "")
                lines.append(f"{icon} **{node}**: {message}")

                payload = event.get("payload") or {}
                msg_text = payload.get("message")
                if msg_text:
                    lines.append(f"   - {msg_text}")
            room_placeholder.markdown("\n\n".join(lines) if lines else "No graph events yet.")

        def on_graph_event(event) -> None:
            room_events.append(event)
            if event.get("phase") == "start":
                status_placeholder.info(f"Running: {event.get('node', 'unknown')}")
            elif event.get("node") == "synthesis" and event.get("phase") == "end":
                status_placeholder.success("Graph execution completed")
            _render_room()

        with st.spinner("Thinking..."):
            formatted_question = format_chat_question(
                st.session_state.chat_messages[:-1], user_prompt
            )
            # Use feature profiles if enabled, otherwise use LLM summary
            primary_shap = feature_profiles_json if use_feature_profiles else shap_llm_summary_json
            llm_kwargs = {
                "question": formatted_question,
                "shap_json": primary_shap,
                "tree_json": tree_json,
                "model_name": model_name,
                "temperature": temperature,
                "feature_profiles_json": feature_profiles_json if use_feature_profiles else None,
            }

            try:
                if execution_mode == "agent":
                    status_placeholder.info("Running: single agent")
                    answer = ask_generalist_agent(**llm_kwargs)
                    status_placeholder.success("Single-agent execution completed")
                else:
                    status_placeholder.info("Running: graph")
                    if show_agent_room:
                        trace_result = run_explainer_graph(
                            **llm_kwargs,
                            on_event=on_graph_event,
                        )
                        answer = str(trace_result.get("answer", ""))
                    else:
                        answer = ask_explainer_graph(**llm_kwargs)
                    status_placeholder.success("Graph execution completed")
            except httpx.ConnectError:
                status_placeholder.error("Cannot connect to Ollama. Start Ollama and try again.")
                answer = (
                    "I could not reach Ollama (connection refused). "
                    "Please start Ollama first (for example, run `ollama serve`) and retry."
                )
            except Exception as exc:
                status_placeholder.error(f"LLM request failed: {exc}")
                answer = "The request failed due to an unexpected error. See the message above and try again."

        st.markdown(answer)

        if execution_mode == "graph" and show_agent_room and room_events:
            with st.expander("Agent room", expanded=True):
                _render_room()

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear chat history"):
            st.session_state.chat_messages = []
            st.rerun()
