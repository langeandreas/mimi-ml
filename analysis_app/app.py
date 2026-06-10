"""Main entry point for the Streamlit app."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LLM Analyzer v0", layout="wide")
st.title("LLM Analyzer v0")

st.header("Welcome")
st.write(
    """
    This is the LLM-based model analysis and visualization tool.
    
    Use the sidebar to navigate between pages:
    - **Trajectory Setup**: Configure and run trajectory analysis
    - **SHAP Viewer**: Visualize SHAP feature trajectories
    - **LLM Chat**: Ask questions about your model using LLM agents
    """
)
