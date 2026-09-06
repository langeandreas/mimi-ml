"""Geographic view of cohort attribution and regional decision paths."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from analysis_app.config import DEFAULT_GEODATA_PATH, DEFAULT_HH_GEO_MAPPING_PATH
from explainer.visualizations import (
    aggregate_cohorts_by_region,
    build_cohort_membership,
    load_admin_geodata,
    load_household_admin_mapping,
    visualise_cohort_map,
    visualise_region_decision_paths,
)

MODE_LABELS = {
    "Dominant cohort per region": "dominant",
    "Share of a single cohort": "cohort_share",
    "Cohort risk rate": "risk_rate",
}


@st.cache_data(show_spinner=False)
def _load_geo(geodata_path: str, mapping_path: str, admin_level: str):
    return (
        load_admin_geodata(geodata_path, admin_level=admin_level),
        load_household_admin_mapping(mapping_path, admin_level=admin_level),
    )


def _membership(
    summary_traj: Any,
    tree_entries: List[Dict[str, Any]],
    min_support: int,
    min_paths: int,
    top_k: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Cache membership per parameter set; the classification object is unhashable."""
    key = ("cohort_geo_membership", id(summary_traj), min_support, min_paths, top_k)
    if st.session_state.get("cohort_geo_key") != key:
        st.session_state["cohort_geo_key"] = key
        st.session_state["cohort_geo_value"] = build_cohort_membership(
            summary_traj.classification,
            tree_entries,
            model=summary_traj.model,
            min_support=min_support,
            min_paths=min_paths,
            top_k=top_k,
        )
    return st.session_state["cohort_geo_value"]


def render_cohort_geography_page() -> None:
    st.header("Cohort Geography")
    st.caption(
        "Map the cohorts, decider features and decision paths from the trained trees "
        "onto administrative regions."
    )

    summary_traj = st.session_state.get("summary_traj")
    if summary_traj is None or not getattr(summary_traj, "model", None):
        st.info("No trained model available. Run the Setup page first.")
        return

    tree_entries = getattr(getattr(summary_traj, "explain", None), "trees", None)
    if not tree_entries:
        st.warning("No in-memory tree objects available for cohort analysis.")
        return

    with st.expander("Data sources", expanded=False):
        geodata_path = st.text_input("Admin geodata CSV", DEFAULT_GEODATA_PATH, key="geo_geodata_path")
        mapping_path = st.text_input("Household geo mapping CSV", DEFAULT_HH_GEO_MAPPING_PATH, key="geo_mapping_path")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        admin_level = st.selectbox("Admin level", ["adm2", "adm1"], key="geo_admin_level")
    with c2:
        min_support = st.slider("Minimum cohort support", 10, 500, 60, key="geo_min_support")
    with c3:
        min_paths = st.slider("Minimum path occurrences", 1, 30, 5, key="geo_min_paths")
    with c4:
        top_k = st.slider("Top cohorts", 5, 50, 20, key="geo_top_k")

    try:
        geo_df, hh_admin_df = _load_geo(geodata_path, mapping_path, admin_level)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load geographic data: {exc}")
        return

    try:
        with st.spinner("Assigning households to cohorts over the full dataset..."):
            membership_df, cohorts = _membership(summary_traj, tree_entries, min_support, min_paths, top_k)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error building cohort membership: {exc}")
        return

    if membership_df.empty:
        st.info("No cohorts met the thresholds. Lower minimum support or path count.")
        return

    cohort_ids = [c.get("cohort_id") for c in cohorts]
    m1, m2 = st.columns(2)
    with m1:
        mode = MODE_LABELS[st.selectbox("Map metric", list(MODE_LABELS), key="geo_mode")]
    with m2:
        selected_cohort = st.selectbox(
            "Cohort (for share mode)",
            cohort_ids,
            key="geo_cohort_id",
            disabled=mode != "cohort_share",
        )

    exclusive = st.checkbox(
        "Assign each household to one cohort only",
        value=False,
        key="geo_exclusive",
        help="Keeps the highest local-impact cohort per household instead of counting overlaps.",
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Cohorts", len(cohorts))
    k2.metric("Households assigned", int(membership_df["household_id"].nunique()))
    k3.metric("Household/cohort pairs", len(membership_df))

    try:
        region_df = aggregate_cohorts_by_region(
            membership_df,
            hh_admin_df,
            mode=mode,
            cohort_id=selected_cohort if mode == "cohort_share" else None,
            exclusive=exclusive,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error aggregating cohorts by region: {exc}")
        return

    comparison = st.checkbox(
        "Fix colour scale to [0, 1]",
        value=(mode == "cohort_share"),
        key="geo_comparison",
        disabled=mode == "dominant",
    )

    ax = visualise_cohort_map(geo_df, region_df, mode=mode, comparison=comparison, show=False)
    st.pyplot(ax.get_figure())

    st.subheader("Regional summary")
    st.dataframe(region_df.sort_values("value", ascending=False), width="content")

    st.subheader("Regional decision paths")
    region_name = st.selectbox(
        "Region",
        region_df.sort_values("n_households", ascending=False)["Name"].tolist(),
        key="geo_region_name",
    )
    top_n = st.slider("Cohorts to explain", 1, 10, 5, key="geo_region_top_n")

    try:
        ax_bar, _ = visualise_region_decision_paths(
            membership_df, cohorts, hh_admin_df, region_name, top_n=top_n, show=False
        )
        st.pyplot(ax_bar.get_figure())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render decision paths for {region_name}: {exc}")
