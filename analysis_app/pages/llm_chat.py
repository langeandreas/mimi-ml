"""LLM Chat page for the Streamlit app."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import httpx
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from explainer.agents import (
    ask_final_shap_agent,
    ask_generalist_agent,
    run_explainer_graph,
)
from analysis_app.utils import format_chat_question, get_llm_config


def render_llm_chat_page() -> None:
    """Render the LLM Chat page."""
    st.header("Explainability Chat")
    st.caption("Ask me anything about the model, its predictions, which features are most important in general. Which ones are ambiguous or more tricky. Which features mostly depend on each other, etc.")

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
        final_explainability_json = getattr(summary_traj, "final_explainability_json", None)
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

    mode_options = ["graph", "agent", "final_shap_agent"]
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["SHAP JSON", "LLM Summary", "Feature Profiles", "Tree JSON", "Final Explainability"])
        with tab1:
            st.json(shap_json, expanded=False)
        with tab2:
            st.json(shap_llm_summary_json, expanded=False)
        with tab3:
            st.json(feature_profiles_json, expanded=False)
        with tab4:
            st.json(tree_json, expanded=False)
        with tab5:
            if final_explainability_json is None:
                st.info("Final explainability JSON is not available for this analysis.")
            else:
                st.json(final_explainability_json, expanded=False)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about feature importance, split behavior, final SHAP drivers, or specific iterations...")
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
        script_run_ctx = get_script_run_ctx()

        def _render_room() -> None:
            if not show_agent_room:
                return
            lines = []
            for event in room_events:
                phase = event.get("phase")
                icon = "[RUN]" if phase == "start" else "[DONE]" if phase == "end" else "[INFO]"
                node = event.get("node", "unknown")
                message = event.get("message", "")
                lines.append(f"{icon} **{node}**: {message}")

                payload = event.get("payload") or {}
                if payload.get("tool"):
                    lines.append(f"   - tool: `{payload['tool']}`")
                if payload.get("args"):
                    lines.append(f"   - args: `{str(payload['args'])[:300]}`")
                if payload.get("content"):
                    lines.append(f"   - content: {str(payload['content'])[:300]}")
                if payload.get("output"):
                    lines.append(f"   - output: {str(payload['output'])[:300]}")
                if payload.get("final_output"):
                    lines.append(f"   - final output: {str(payload['final_output'])[:300]}")
                if payload.get("available_tools"):
                    tool_names = [tool.get("name", "unknown") for tool in payload["available_tools"]]
                    lines.append(f"   - tools available: {', '.join(tool_names)}")
                if payload.get("available_information"):
                    lines.append(f"   - available information: {str(payload['available_information'])[:500]}")
            room_placeholder.markdown("\n\n".join(lines) if lines else "No graph events yet.")

        def on_graph_event(event) -> None:
            if script_run_ctx is not None:
                add_script_run_ctx(threading.current_thread(), script_run_ctx)
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
            context_json = summary_traj.generate_context_json(save=False)
            llm_kwargs = {
                "question": formatted_question,
                "shap_json": primary_shap,
                "tree_json": tree_json,
                "model_name": model_name,
                "temperature": temperature,
                "feature_profiles_json": feature_profiles_json if use_feature_profiles else None,
            }
            graph_kwargs = {
                **llm_kwargs,
                "final_explainability_json": final_explainability_json,
                "context_json": context_json,
            }

            try:
                if execution_mode == "agent":
                    status_placeholder.info("Running: single agent")
                    answer = ask_generalist_agent(**llm_kwargs)
                    status_placeholder.success("Single-agent execution completed")
                elif execution_mode == "final_shap_agent":
                    if final_explainability_json is None:
                        status_placeholder.error("Final explainability JSON is not available for this analysis.")
                        answer = (
                            "This analysis does not include the final explainability payload needed by the final SHAP agent. "
                            "Run the setup pipeline again to regenerate it."
                        )
                    else:
                        status_placeholder.info("Running: final SHAP agent")
                        answer = ask_final_shap_agent(
                            question=formatted_question,
                            final_explainability_json=final_explainability_json,
                            model_name=model_name,
                            temperature=temperature,
                        )
                        status_placeholder.success("Final-SHAP-agent execution completed")
                else:
                    status_placeholder.info("Running: graph")
                    if show_agent_room:
                        trace_result = run_explainer_graph(
                            **graph_kwargs,
                            on_event=on_graph_event,
                        )
                        answer = str(trace_result.get("answer", ""))
                        report_path = trace_result.get("report_path")
                        trace_path = trace_result.get("trace_path")
                        if report_path:
                            st.session_state.last_report_path = str(report_path)
                            st.caption(f"Saved report: {report_path}")
                        if trace_path:
                            st.session_state.last_trace_path = str(trace_path)
                            st.caption(f"Saved agent trace: {trace_path}")
                    else:
                        trace_result = run_explainer_graph(**graph_kwargs)
                        answer = str(trace_result.get("answer", ""))
                        report_path = trace_result.get("report_path")
                        trace_path = trace_result.get("trace_path")
                        if report_path:
                            st.session_state.last_report_path = str(report_path)
                            st.caption(f"Saved report: {report_path}")
                        if trace_path:
                            st.session_state.last_trace_path = str(trace_path)
                            st.caption(f"Saved agent trace: {trace_path}")
                    status_placeholder.success("Graph execution completed")
            except httpx.ConnectError:
                status_placeholder.error("Cannot connect to Ollama. Start Ollama and try again.")
                answer = (
                    "I could not reach Ollama (connection refused). "
                    "Please start Ollama first (for example, run `ollama serve`) and retry."
                )
            except httpx.HTTPError as exc:
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
