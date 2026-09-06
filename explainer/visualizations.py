"""Geographic visualisation of tree-derived cohort and decision-path insights.

Projects the cohorts produced by
:class:`explainer.analysis.cohort_attribution_analyzer.CohortAttributionAnalyzer`
onto administrative polygons, following the plotting conventions of
:func:`predictor.visualisations.visualise_actual_predicted_map`.

Pipeline::

    load_admin_geodata          -> polygons (Name, geometry)
    load_household_admin_mapping-> household_id -> admin region
    build_cohort_membership     -> household_id -> cohort_id (full dataset)
    aggregate_cohorts_by_region -> per-region metric
    visualise_cohort_map        -> choropleth
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely import wkt

from .analysis.cohort_attribution_analyzer import CohortAttributionAnalyzer
from .analysis.trajectory_utils import substitute_feature_names

__all__ = [
    "load_admin_geodata",
    "load_household_admin_mapping",
    "build_cohort_membership",
    "aggregate_cohorts_by_region",
    "visualise_cohort_map",
    "visualise_region_decision_paths",
    "plot_cohort_geography",
]

TITLE_FONT = {"fontsize": "17"}

MODE_TITLES = {
    "dominant": "Dominant cohort",
    "cohort_share": "Cohort share",
    "risk_rate": "Cohort risk rate",
}

_ADMIN_COLUMNS = {
    "adm1": {"geometry": "adm1Geometry", "code": "adm1Code", "name": "adm1Name"},
    "adm2": {"geometry": "adm2Geometry", "code": "adm2Code", "name": "adm2Name_x"},
}


def _normalise_name(value: Any) -> str:
    return str(value).strip().casefold()


def _household_key(values: Any) -> pd.Series:
    """Household ids are string-like in some countries and integer in others."""
    return pd.Series(values).astype(str).str.strip()


def _feature_rename_map(
    cohorts: List[Dict[str, Any]],
    feature_names: Sequence[str],
    features_csv_path: str,
) -> Dict[str, str]:
    """Map every codename referenced by a cohort to its human-readable label."""
    codes = set()
    for cohort in cohorts:
        codes.add(str(cohort.get("decider_feature_name", "")))
        for cond in cohort.get("scope_conditions", []) or []:
            codes.add(str(cond.get("feature_name", "")))
        for tp in cohort.get("top_paths", []) or []:
            for cond in tp.get("path_conditions", []) or []:
                codes.add(str(cond.get("feature_name", "")))
    codes.discard("")
    ordered = sorted(codes)
    if not ordered:
        return {}

    lookup = substitute_feature_names(
        pd.DataFrame({"feature_name": ordered}),
        feature_index_col="feature_name",
        feature_names=list(feature_names),
        features_csv_path=features_csv_path,
    )
    return dict(zip(ordered, lookup["feature_name"]))


def _render_scope_signature(conditions: Sequence[Dict[str, Any]]) -> str:
    return " AND ".join(f"{c.get('feature_name')} {c.get('op')} {c.get('threshold')}" for c in conditions)


def _wrap_and_joined(text: str, indent: str = "        ") -> str:
    """Break an ' AND '-joined condition string onto one line per condition."""
    return f"\n{indent}AND ".join(text.split(" AND "))


def _decider_condition_text(cohort: Dict[str, Any]) -> str:
    """Render the decider's own split (e.g. 'food_group8 >= 1.0'), not just its scope."""
    top_paths = cohort.get("top_paths") or []
    conditions = top_paths[0].get("path_conditions") if top_paths else None
    if not conditions:
        return str(cohort.get("decider_feature_name", ""))
    decider = conditions[-1]
    return f"{decider.get('feature_name')} {decider.get('op')} {decider.get('threshold')}"


def _cohort_prediction_text(cohort: Dict[str, Any]) -> str:
    """Render what the cohort predicts, preferring the model's mean confidence."""
    confidence = cohort.get("mean_confidence")
    if confidence is not None:
        return f"{float(confidence):.0%} predicted chance of inadequacy"
    return f"{float(cohort.get('risk_rate', 0.0)):.0%} observed chance of inadequacy"


def _apply_readable_feature_names(
    cohorts: List[Dict[str, Any]],
    feature_names: Sequence[str],
    features_csv_path: str,
) -> List[Dict[str, Any]]:
    """Rewrite decider/scope/path feature names in place with readable labels."""
    rename_map = _feature_rename_map(cohorts, feature_names, features_csv_path)
    if not rename_map:
        return cohorts

    for cohort in cohorts:
        cohort["decider_feature_name"] = rename_map.get(
            str(cohort.get("decider_feature_name", "")), cohort.get("decider_feature_name", "")
        )

        scope_conditions = []
        for cond in cohort.get("scope_conditions", []) or []:
            cond = dict(cond)
            cond["feature_name"] = rename_map.get(str(cond.get("feature_name", "")), cond.get("feature_name"))
            scope_conditions.append(cond)
        cohort["scope_conditions"] = scope_conditions
        cohort["scope_signature"] = _render_scope_signature(scope_conditions)

        top_paths = []
        for tp in cohort.get("top_paths", []) or []:
            tp = dict(tp)
            path_conditions = []
            for cond in tp.get("path_conditions", []) or []:
                cond = dict(cond)
                cond["feature_name"] = rename_map.get(str(cond.get("feature_name", "")), cond.get("feature_name"))
                path_conditions.append(cond)
            tp["path_conditions"] = path_conditions
            top_paths.append(tp)
        cohort["top_paths"] = top_paths

    return cohorts


# ----------------------------------------------------------------------
# Phase 1 - geo loaders
# ----------------------------------------------------------------------


def load_admin_geodata(geodata_path: str, admin_level: str = "adm2") -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of admin polygons from a WKT geodata CSV.

    :param geodata_path: path to a geodata CSV holding WKT polygons per admin unit
    :param admin_level: 'adm1' or 'adm2'
    :return: GeoDataFrame with columns ``adm_code``, ``Name``, ``geometry`` (EPSG:4326)
    """
    if admin_level not in _ADMIN_COLUMNS:
        raise ValueError(f"admin_level must be one of {sorted(_ADMIN_COLUMNS)}, got {admin_level!r}")

    cols = _ADMIN_COLUMNS[admin_level]
    raw = pd.read_csv(geodata_path)

    name_col = cols["name"]
    if name_col not in raw.columns:  # adm2Name_x only exists in the merged export
        name_col = cols["name"].replace("_x", "")
    missing = [c for c in (cols["geometry"], cols["code"], name_col) if c not in raw.columns]
    if missing:
        raise KeyError(f"{geodata_path} is missing required columns: {missing}")

    geo = raw[[cols["code"], name_col, cols["geometry"]]].copy()
    geo.columns = ["adm_code", "Name", "wkt"]
    geo = geo.dropna(subset=["wkt", "adm_code"]).drop_duplicates(subset=["adm_code"])
    geo["geometry"] = geo["wkt"].map(wkt.loads)
    geo = geo.drop(columns=["wkt"])
    geo["adm_code"] = geo["adm_code"].astype(int)
    geo["Name"] = geo["Name"].astype(str).str.strip()

    return gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)


def load_household_admin_mapping(
    mapping_path: str,
    admin_level: str = "adm2",
    household_col: str = "household_id",
) -> pd.DataFrame:
    """Load the household -> admin region lookup.

    :param mapping_path: path to the household/geography mapping CSV
    :param admin_level: 'adm1' or 'adm2'
    :param household_col: household identifier column in the mapping CSV
    :return: DataFrame with columns ``household_id``, ``adm_code``, ``Name``
    """
    if admin_level not in _ADMIN_COLUMNS:
        raise ValueError(f"admin_level must be one of {sorted(_ADMIN_COLUMNS)}, got {admin_level!r}")

    code_col = _ADMIN_COLUMNS[admin_level]["code"]
    name_col = _ADMIN_COLUMNS[admin_level]["name"].replace("_x", "")
    raw = pd.read_csv(mapping_path)

    missing = [c for c in (household_col, code_col, name_col) if c not in raw.columns]
    if missing:
        raise KeyError(f"{mapping_path} is missing required columns: {missing}")

    mapping = raw[[household_col, code_col, name_col]].copy()
    mapping.columns = ["household_id", "adm_code", "Name"]
    mapping = mapping.dropna().drop_duplicates(subset=["household_id"])
    mapping["household_id"] = _household_key(mapping["household_id"])
    mapping["adm_code"] = mapping["adm_code"].astype(int)
    mapping["Name"] = mapping["Name"].astype(str).str.strip()

    return mapping.reset_index(drop=True)


# ----------------------------------------------------------------------
# Phase 2 - cohort membership over the full dataset
# ----------------------------------------------------------------------


def build_cohort_membership(
    classification: Any,
    tree_entries: List[Dict[str, Any]],
    *,
    model: Optional[Any] = None,
    final_shap_values: Optional[Any] = None,
    min_support: int = 50,
    min_paths: int = 5,
    top_k: int = 20,
    top_local_shap: int = 8,
    features_csv_path: str = "data/features_explanations.csv",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Assign every household in the full dataset to the cohorts it belongs to.

    Unlike ``TreeAnalyzer.analyze_subpopulation_cohorts`` this runs over
    ``classification.data`` (all rows, hhid-indexed) rather than the resampled
    training split, and keeps the full membership instead of a 20-row preview.

    :param classification: fitted ``Classification`` instance
    :param tree_entries: ``[{"iteration": int, "tree": json}]`` from the explainer callback
    :param model: optional fitted estimator, used for cohort mean confidence
    :param final_shap_values: optional SHAP matrix aligned to the full dataset
    :param min_support: minimum households for a cohort to be retained
    :param min_paths: minimum repeated scope->decider paths for a cohort
    :param top_k: number of cohorts to keep, ranked as in ``to_json_schema``
    :param top_local_shap: number of local SHAP features summarised per cohort
    :param features_csv_path: codename -> human-readable feature name lookup CSV
    :return: ``(membership_df, cohorts)``; membership rows are household/cohort pairs
    """
    X_full = classification.data.iloc[:, 1:]
    y_full = classification.data[classification.type_target]

    analyzer = CohortAttributionAnalyzer(list(X_full.columns))

    y_proba = None
    if model is not None:
        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_full)[:, -1]

    rules = analyzer.extract_path_rules(tree_entries)
    assigned = analyzer.assign_samples_to_rules(X_full, rules)
    cohorts = analyzer.build_cohorts(assigned, min_support=min_support, min_paths=min_paths)
    enriched = analyzer.enrich_cohorts_with_metrics(
        cohorts,
        y_true=np.asarray(y_full).reshape(len(y_full), -1)[:, 0],
        y_proba=y_proba,
        final_shap_values=final_shap_values,
        top_local_shap=top_local_shap,
    )

    # Same ranking as CohortAttributionAnalyzer.to_json_schema so cohort ids match the app.
    ranked = sorted(
        enriched,
        key=lambda c: (
            -float(c.get("support_count", 0)) * float(c.get("decision_impact_local", 0.0)),
            -float(c.get("support_count", 0)),
        ),
    )[: max(int(top_k), 1)]
    ranked = _apply_readable_feature_names(ranked, list(X_full.columns), features_csv_path)

    household_ids = np.asarray(X_full.index)
    rows: List[Dict[str, Any]] = []
    for idx, cohort in enumerate(ranked, start=1):
        cohort_id = f"C{idx:03d}"
        cohort["cohort_id"] = cohort_id
        positions = sorted(int(i) for i in cohort.get("support_ids", set()))
        positions = [p for p in positions if 0 <= p < len(household_ids)]
        for hh in household_ids[positions]:
            rows.append(
                {
                    "household_id": hh,
                    "cohort_id": cohort_id,
                    "decider_feature_name": cohort.get("decider_feature_name", ""),
                    "scope_signature": cohort.get("scope_signature", ""),
                    "risk_rate": float(cohort.get("risk_rate", 0.0)),
                    "decision_impact_local": float(cohort.get("decision_impact_local", 0.0)),
                    "support_count": int(cohort.get("support_count", 0)),
                }
            )

    membership = pd.DataFrame(
        rows,
        columns=[
            "household_id",
            "cohort_id",
            "decider_feature_name",
            "scope_signature",
            "risk_rate",
            "decision_impact_local",
            "support_count",
        ],
    )
    membership["household_id"] = _household_key(membership["household_id"])
    membership.attrs["household_ids"] = list(_household_key(household_ids))
    return membership, ranked


# ----------------------------------------------------------------------
# Phase 3 - regional aggregation
# ----------------------------------------------------------------------


def aggregate_cohorts_by_region(
    membership_df: pd.DataFrame,
    hh_admin_df: pd.DataFrame,
    mode: str = "dominant",
    cohort_id: Optional[str] = None,
    exclusive: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate household-level cohort membership into a per-region metric.

    :param membership_df: output of :func:`build_cohort_membership`
    :param hh_admin_df: output of :func:`load_household_admin_mapping`
    :param mode: 'dominant', 'cohort_share' or 'risk_rate'
    :param cohort_id: required for mode 'cohort_share'
    :param exclusive: keep only each household's highest-impact cohort
    :param verbose: print household coverage of the geographic mapping
    :return: DataFrame with ``Name``, ``adm_code``, ``value``, ``n_households`` (+ cohort labels)
    """
    if mode not in MODE_TITLES:
        raise ValueError(f"mode must be one of {sorted(MODE_TITLES)}, got {mode!r}")
    if mode == "cohort_share" and not cohort_id:
        raise ValueError("mode='cohort_share' requires a cohort_id")

    members = membership_df.copy()
    if exclusive:
        members = (
            members.sort_values("decision_impact_local", ascending=False)
            .drop_duplicates(subset=["household_id"])
        )

    members["household_id"] = _household_key(members["household_id"])
    admin = hh_admin_df.copy()
    admin["household_id"] = _household_key(admin["household_id"])

    universe = _household_key(membership_df.attrs.get("household_ids", members["household_id"].unique()))
    region_totals = (
        admin[admin["household_id"].isin(universe)]
        .groupby(["adm_code", "Name"], as_index=False)
        .agg(n_households=("household_id", "nunique"))
    )
    if verbose:
        covered = admin["household_id"].isin(universe).sum()
        print(f"Geographic coverage: {covered}/{len(universe)} households mapped to an admin region.")

    joined = members.merge(admin, on="household_id", how="inner")
    if joined.empty:
        raise ValueError("No households matched between cohort membership and the admin mapping.")

    if mode == "risk_rate":
        agg = (
            joined.groupby(["adm_code", "Name"], as_index=False)
            .agg(value=("risk_rate", "mean"))
        )
        return region_totals.merge(agg, on=["adm_code", "Name"], how="left")

    counts = (
        joined.groupby(["adm_code", "Name", "cohort_id", "decider_feature_name", "scope_signature"], as_index=False)
        .agg(cohort_households=("household_id", "nunique"))
    )
    counts = counts.merge(region_totals, on=["adm_code", "Name"], how="left")
    counts["value"] = counts["cohort_households"] / counts["n_households"].replace(0, np.nan)

    if mode == "cohort_share":
        selected = counts[counts["cohort_id"] == cohort_id]
        out = region_totals.merge(
            selected[["adm_code", "Name", "cohort_id", "decider_feature_name", "scope_signature", "value"]],
            on=["adm_code", "Name"],
            how="left",
        )
        out["cohort_id"] = out["cohort_id"].fillna(cohort_id)
        return out.fillna({"value": 0.0})

    dominant = counts.sort_values(["adm_code", "value"], ascending=[True, False]).drop_duplicates(subset=["adm_code"])
    return region_totals.merge(
        dominant[["adm_code", "Name", "cohort_id", "decider_feature_name", "scope_signature", "value"]],
        on=["adm_code", "Name"],
        how="left",
    )


# ----------------------------------------------------------------------
# Phase 4 - plots
# ----------------------------------------------------------------------


def visualise_cohort_map(
    geo_df: gpd.GeoDataFrame,
    region_df: pd.DataFrame,
    mode: str = "dominant",
    color: str = "Reds",
    adminsfontsize: int = 10,
    comparison: bool = True,
    iso3: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = False,
    path: Optional[str] = None,
    show: bool = True,
):
    """Draw the per-region cohort choropleth.

    :param geo_df: admin polygons from :func:`load_admin_geodata`
    :param region_df: aggregated regions from :func:`aggregate_cohorts_by_region`
    :param mode: 'dominant' (categorical), 'cohort_share' or 'risk_rate' (continuous)
    :param color: matplotlib colormap name for the continuous modes
    :param adminsfontsize: font size of the region annotations
    :param comparison: fix the continuous colour range to [0, 1] for cross-map comparison
    :param iso3: country ISO3 code, used in the saved file name
    :param title: overrides the generated title
    :param save: save the figure
    :param path: directory prefix for the saved figure
    :param show: call ``plt.show()``; disable when embedding the figure elsewhere
    :return: the matplotlib axes
    """
    if mode not in MODE_TITLES:
        raise ValueError(f"mode must be one of {sorted(MODE_TITLES)}, got {mode!r}")

    geo = geo_df.copy()
    regions = region_df.copy()
    geo["_key"] = geo["Name"].map(_normalise_name)
    regions["_key"] = regions["Name"].map(_normalise_name)

    unmatched = sorted(set(regions["_key"]) - set(geo["_key"]))
    if unmatched:
        print(f"Warning: {len(unmatched)} region(s) had no matching polygon: {unmatched}")

    to_vis = geo.merge(regions.drop(columns=["Name"]), on="_key", how="left").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title(title or f"{MODE_TITLES[mode]} by admin region", loc="center", pad=10, fontdict=TITLE_FONT)

    if mode == "dominant":
        labels = sorted(to_vis["cohort_id"].dropna().unique())
        palette = plt.get_cmap("tab20")
        colours = {label: palette(i % 20) for i, label in enumerate(labels)}
        to_vis["_color"] = to_vis["cohort_id"].map(colours).fillna("lightgrey")
        to_vis.plot(color=list(to_vis["_color"]), ax=ax, linewidth=0.8, edgecolors="grey")
        ax.legend(
            handles=[Patch(facecolor=colours[label], edgecolor="grey", label=label) for label in labels],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=10,
            title="Cohort",
        )
    else:
        vmin, vmax = (0, 1) if comparison else (to_vis["value"].min(), to_vis["value"].max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=color, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.3, pad=0.05)
        cbar.ax.tick_params(labelsize=11)
        cbar.outline.set_visible(False)
        to_vis.plot(column="value", cmap=color, norm=norm, ax=ax, linewidth=0.8, edgecolors="grey", missing_kwds={"color": "lightgrey"})

    centroids = to_vis.geometry.to_crs(3857).centroid.to_crs(to_vis.crs)
    for i in range(len(to_vis)):
        label = to_vis["Name"].iloc[i]
        if mode == "dominant" and pd.notna(to_vis["cohort_id"].iloc[i]):
            label = f"{label}\n{to_vis['cohort_id'].iloc[i]}"
        ax.text(centroids.x.iloc[i], centroids.y.iloc[i], label, size=adminsfontsize, ha="center", fontname="Open sans")

    if save:
        plt.savefig(f"{path or ''}{iso3 or 'map'}_cohort_{mode}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    return ax


def visualise_region_decision_paths(
    membership_df: pd.DataFrame,
    cohorts: Sequence[Dict[str, Any]],
    hh_admin_df: pd.DataFrame,
    region_name: str,
    top_n: int = 5,
    max_conditions: int = 4,
    save: bool = False,
    path: Optional[str] = None,
    show: bool = True,
):
    """Show the cohorts, decider features and decision paths driving one region.

    :param membership_df: output of :func:`build_cohort_membership`
    :param cohorts: ranked cohorts returned alongside ``membership_df``
    :param hh_admin_df: output of :func:`load_household_admin_mapping`
    :param region_name: admin region to explain
    :param top_n: number of cohorts to show
    :param max_conditions: conditions rendered per decision path
    :param save: save the figure
    :param path: directory prefix for the saved figure
    :param show: call ``plt.show()``; disable when embedding the figure elsewhere
    :return: the matplotlib axes pair
    """
    admin = hh_admin_df[hh_admin_df["Name"].map(_normalise_name) == _normalise_name(region_name)].copy()
    if admin.empty:
        raise ValueError(f"Region {region_name!r} not found in the household admin mapping.")
    admin["household_id"] = _household_key(admin["household_id"])

    members = membership_df.copy()
    members["household_id"] = _household_key(members["household_id"])
    joined = members.merge(admin[["household_id"]], on="household_id", how="inner")
    if joined.empty:
        raise ValueError(f"No cohort households fall in region {region_name!r}.")

    total = admin["household_id"].nunique()
    ranking = (
        joined.groupby(["cohort_id", "decider_feature_name"], as_index=False)
        .agg(households=("household_id", "nunique"))
        .assign(share=lambda d: d["households"] / total)
        .sort_values("share", ascending=False)
        .head(top_n)
    )

    by_id = {c.get("cohort_id"): c for c in cohorts}
    fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(16, 1.6 * max(len(ranking), 3) + 2), gridspec_kw={"width_ratios": [1, 1.4]})

    ax_bar.barh(
        [f"{r.cohort_id}\n{_decider_condition_text(by_id.get(r.cohort_id, {}))}" for r in ranking.itertuples()],
        ranking["share"],
        color="orange",
        alpha=0.7,
        edgecolor="grey",
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Share of region households", fontsize=12, fontname="Open sans", fontweight="bold")
    ax_bar.set_title(f"Most common cohorts - {region_name}", loc="left", pad=10, fontdict=TITLE_FONT)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

    lines: List[str] = []
    for row in ranking.itertuples():
        cohort = by_id.get(row.cohort_id, {})
        decider_condition = _decider_condition_text(cohort)
        prediction = _cohort_prediction_text(cohort)
        lines.append(f"{row.cohort_id} - if {decider_condition} -> {prediction} ({row.share:.0%} of households)")
        lines.append(f"    scope: {_wrap_and_joined(cohort.get('scope_signature') or '(root split)')}")
        for rank, tp in enumerate(cohort.get("top_paths", [])[:2], start=1):
            conds = tp.get("path_conditions", [])[:max_conditions]
            rendered = _wrap_and_joined(
                " AND ".join(f"{c.get('feature_name')} {c.get('op')} {c.get('threshold')}" for c in conds)
            )
            suffix = " ..." if len(tp.get("path_conditions", [])) > max_conditions else ""
            lines.append(f"    path {rank} (leaf {tp.get('leaf_value')}): {rendered}{suffix}")
        lines.append("")

    ax_text.axis("off")
    ax_text.set_title("Top conditional features and decision paths", loc="left", pad=10, fontdict=TITLE_FONT)
    ax_text.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    plt.tight_layout()
    if save:
        plt.savefig(f"{path or ''}decision_paths_{_normalise_name(region_name).replace(' ', '_')}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    return ax_bar, ax_text


def plot_cohort_geography(
    classification: Any,
    tree_entries: List[Dict[str, Any]],
    geodata_path: str,
    mapping_path: str,
    *,
    admin_level: str = "adm2",
    mode: str = "dominant",
    cohort_id: Optional[str] = None,
    model: Optional[Any] = None,
    min_support: int = 50,
    min_paths: int = 5,
    top_k: int = 20,
    exclusive: bool = False,
    features_csv_path: str = "data/features_explanations.csv",
    **plot_kwargs: Any,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """Run loaders, cohort assignment, aggregation and plotting in one call.

    :return: ``(membership_df, cohorts, region_df)``
    """
    geo_df = load_admin_geodata(geodata_path, admin_level=admin_level)
    hh_admin_df = load_household_admin_mapping(mapping_path, admin_level=admin_level)
    membership_df, cohorts = build_cohort_membership(
        classification,
        tree_entries,
        model=model,
        min_support=min_support,
        min_paths=min_paths,
        top_k=top_k,
        features_csv_path=features_csv_path,
    )
    region_df = aggregate_cohorts_by_region(
        membership_df,
        hh_admin_df,
        mode=mode,
        cohort_id=cohort_id,
        exclusive=exclusive,
    )
    visualise_cohort_map(geo_df, region_df, mode=mode, **plot_kwargs)
    return membership_df, cohorts, region_df
