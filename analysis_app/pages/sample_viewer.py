"""Sample-level attribution explorer page.

Combines final-step SHAP values with a tree-path weighted leaf attribution for
one selected sample. The weighted-leaf view allocates each tree leaf value to
features along that sample's decision path, with a controllable depth emphasis.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


def _get_booster(summary_traj: Any) -> Optional[Any]:
    """Return fitted booster from either estimator or GridSearchCV wrapper."""
    model = getattr(summary_traj, "model", None)
    if model is None:
        return None
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if estimator is None or not hasattr(estimator, "get_booster"):
        return None
    return estimator.get_booster()


def _safe_sample_value(sample_row: pd.Series, feature_name: str) -> float:
    """Read a numeric sample value safely from one row."""
    if feature_name not in sample_row.index:
        return float("nan")
    value = pd.to_numeric(sample_row[feature_name], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _traverse_tree_for_sample(
    tree_obj: Dict[str, Any],
    sample_row: pd.Series,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Traverse one XGBoost tree and return leaf value + visited split steps."""
    node = tree_obj
    steps: List[Dict[str, Any]] = []

    while isinstance(node, dict) and "leaf" not in node:
        split_name = str(node.get("split", ""))
        threshold = float(node.get("split_condition", 0.0))
        value = _safe_sample_value(sample_row, split_name)

        children = node.get("children", [])
        child_by_id = {
            int(child.get("nodeid", -1)): child
            for child in children
            if isinstance(child, dict)
        }

        yes_id = int(node.get("yes", -1))
        no_id = int(node.get("no", -1))
        missing_id = int(node.get("missing", yes_id))

        yes_child = child_by_id.get(yes_id, children[0] if len(children) > 0 else None)
        no_child = child_by_id.get(no_id, children[1] if len(children) > 1 else None)
        missing_child = child_by_id.get(missing_id, yes_child)

        if np.isnan(value):
            next_child = missing_child
            op_taken = "missing"
        elif value < threshold:
            next_child = yes_child
            op_taken = "<"
        else:
            next_child = no_child
            op_taken = ">="

        steps.append(
            {
                "feature_name": split_name,
                "threshold": threshold,
                "sample_value": value,
                "op_taken": op_taken,
            }
        )

        if next_child is None:
            break
        node = next_child

    leaf_value = float(node.get("leaf", 0.0)) if isinstance(node, dict) else 0.0
    return leaf_value, steps


def _distribute_leaf_by_path(
    leaf_value: float,
    steps: List[Dict[str, Any]],
    feature_to_idx: Dict[str, int],
    depth_emphasis: float,
) -> Dict[int, float]:
    """Allocate one leaf value over split features on the chosen path.

    Later path positions get higher weight when depth_emphasis > 0.
    """
    indexed_steps: List[Tuple[int, int]] = []
    for pos, step in enumerate(steps):
        idx = feature_to_idx.get(str(step.get("feature_name", "")), -1)
        if idx >= 0:
            indexed_steps.append((idx, pos))

    if not indexed_steps:
        return {}

    raw_weights = np.asarray(
        [(pos + 1) ** float(depth_emphasis) for _, pos in indexed_steps],
        dtype=float,
    )
    weight_sum = float(np.sum(raw_weights))
    if weight_sum <= 0.0:
        raw_weights = np.ones_like(raw_weights)
        weight_sum = float(np.sum(raw_weights))

    alloc: Dict[int, float] = {}
    for (feature_idx, _), w in zip(indexed_steps, raw_weights):
        alloc[feature_idx] = float(alloc.get(feature_idx, 0.0) + (leaf_value * float(w) / weight_sum))
    return alloc


def _compute_sample_weighted_leaf_contributions(
    booster: Any,
    sample_row: pd.Series,
    feature_names: List[str],
    depth_emphasis: float,
    max_trees: int,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    """Compute weighted-leaf contributions for one sample across trees."""
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    tree_dump = booster.get_dump(dump_format="json")
    if not tree_dump:
        return {}, []

    n_use = min(max_trees, len(tree_dump)) if max_trees > 0 else len(tree_dump)
    total: Dict[int, float] = {}
    per_tree: List[Dict[str, Any]] = []

    for tree_idx, tree_json in enumerate(tree_dump[:n_use]):
        tree_obj = json.loads(tree_json)
        leaf_value, steps = _traverse_tree_for_sample(tree_obj, sample_row)
        alloc = _distribute_leaf_by_path(
            leaf_value=leaf_value,
            steps=steps,
            feature_to_idx=feature_to_idx,
            depth_emphasis=depth_emphasis,
        )

        for feature_idx, contribution in alloc.items():
            total[feature_idx] = float(total.get(feature_idx, 0.0) + float(contribution))

        per_tree.append(
            {
                "tree_idx": int(tree_idx),
                "leaf_value": float(leaf_value),
                "path_length": int(len(steps)),
                "alloc": alloc,
                "steps": steps,
            }
        )

    return total, per_tree


def _to_readable_name_map(feature_names: List[str]) -> Dict[int, str]:
    """Map feature indices to human-readable names using shared substitution."""
    map_df = pd.DataFrame({"feature_index": list(feature_names)})
    map_df = substitute_feature_names(
        map_df,
        feature_index_col="feature_index",
        feature_names=feature_names,
    )
    return {i: str(name) for i, name in enumerate(map_df["feature_index"].astype(str).tolist())}


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute robust Pearson correlation for two vectors."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _format_step(step: Dict[str, Any], readable_name: str) -> str:
    """Format one split step into a readable path token."""
    op_taken = str(step.get("op_taken", ""))
    threshold = float(step.get("threshold", 0.0))
    if op_taken == "missing":
        return f"{readable_name} is missing"
    return f"{readable_name} {op_taken} {threshold:.4f}"


def _build_path_signature(
    steps: List[Dict[str, Any]],
    readable_by_raw: Dict[str, str],
) -> str:
    """Build a canonical path signature for grouping identical routes."""
    tokens: List[str] = []
    for step in steps:
        raw_name = str(step.get("feature_name", ""))
        readable_name = readable_by_raw.get(raw_name, raw_name)
        tokens.append(_format_step(step, readable_name))
    return " AND ".join(tokens)


def _aggregate_path_frequencies(
    tree_records: List[Dict[str, Any]],
    readable_by_raw: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate repeated decision paths and summarize their leaf values."""
    buckets: Dict[str, Dict[str, Any]] = {}

    for rec in tree_records:
        steps = list(rec.get("steps", []))
        signature = _build_path_signature(steps, readable_by_raw)
        leaf_value = float(rec.get("leaf_value", 0.0))
        tree_idx = int(rec.get("tree_idx", -1))

        if signature not in buckets:
            buckets[signature] = {
                "path_signature": signature,
                "frequency": 0,
                "leaf_values": [],
                "tree_indices": [],
                "path_length": int(len(steps)),
            }

        row = buckets[signature]
        row["frequency"] += 1
        row["leaf_values"].append(leaf_value)
        row["tree_indices"].append(tree_idx)

    rows: List[Dict[str, Any]] = []
    for payload in buckets.values():
        leaf_arr = np.asarray(payload["leaf_values"], dtype=float)
        rows.append(
            {
                "path_signature": str(payload["path_signature"]),
                "frequency": int(payload["frequency"]),
                "path_length": int(payload["path_length"]),
                "leaf_mean": float(np.mean(leaf_arr)) if leaf_arr.size else 0.0,
                "leaf_sum": float(np.sum(leaf_arr)) if leaf_arr.size else 0.0,
                "leaf_abs_sum": float(np.sum(np.abs(leaf_arr))) if leaf_arr.size else 0.0,
                "leaf_min": float(np.min(leaf_arr)) if leaf_arr.size else 0.0,
                "leaf_max": float(np.max(leaf_arr)) if leaf_arr.size else 0.0,
                "positive_leaf_rate": float(np.mean(leaf_arr > 0)) if leaf_arr.size else 0.0,
                "tree_indices": payload["tree_indices"],
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    return out_df.sort_values(
        ["frequency", "leaf_abs_sum"],
        ascending=[False, False],
    )


def render_sample_viewer_page() -> None:
    """Render sample-level weighted-leaf and SHAP attribution explorer."""
    st.header("Sample-Level Attribution Explorer")
    st.caption(
        "Inspect one sample at a time and compare SHAP with tree-path weighted leaf "
        "contributions to see why feature impact changes with this sample's values."
    )

    if "summary_traj" not in st.session_state or st.session_state.summary_traj is None:
        st.info("No trained model in session. Run Setup first.")
        return

    summary_traj = st.session_state.summary_traj
    booster = _get_booster(summary_traj)
    if booster is None:
        st.warning("Could not access model booster from session state.")
        return

    X_train = summary_traj.classification.train_test.get("X_train")
    if X_train is None or len(X_train) == 0:
        st.warning("Training features are unavailable in session state.")
        return
    X_df = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    if X_df.empty:
        st.warning("Training features are empty.")
        return

    y_train_raw = summary_traj.classification.train_test.get("Y_train")
    y_arr = np.asarray(y_train_raw).reshape(len(y_train_raw), -1)[:, 0] if y_train_raw is not None else None

    shap_history = list(getattr(summary_traj.explain, "shap_values_history", []))
    if not shap_history:
        st.warning("No SHAP history found in session. Run Setup first.")
        return

    shap_history = sorted(shap_history, key=lambda item: int(item.get("iteration", 0)))
    shap_matrix = _normalize_shap_matrix(shap_history[-1]["shap_values"])

    n_rows = min(int(X_df.shape[0]), int(shap_matrix.shape[0]))
    n_features = min(int(X_df.shape[1]), int(shap_matrix.shape[1]))
    if n_rows <= 0 or n_features <= 0:
        st.warning("Could not align SHAP and feature matrix for sample analysis.")
        return

    X_df = X_df.iloc[:n_rows, :n_features].copy()
    shap_matrix = shap_matrix[:n_rows, :n_features]
    feature_names = [str(c) for c in X_df.columns]

    readable_map = _to_readable_name_map(feature_names)
    readable_by_raw = {
        raw: str(readable_map.get(i, raw))
        for i, raw in enumerate(feature_names)
    }

    st.subheader("Sample Selection")
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_sample_idx = st.number_input(
            "Sample index",
            min_value=0,
            max_value=max(n_rows - 1, 0),
            value=0,
            step=1,
            key="sample_viewer_selected_idx",
        )
    with c2:
        depth_emphasis = st.slider(
            "Depth emphasis for leaf allocation",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="sample_viewer_depth_emphasis",
            help="Higher values assign more of each leaf value to later splits on the sample path.",
        )
    with c3:
        max_trees = st.number_input(
            "Trees to include",
            min_value=1,
            max_value=max(1, int(len(booster.get_dump()))),
            value=max(1, int(len(booster.get_dump()))),
            step=1,
            key="sample_viewer_max_trees",
        )

    sample_idx = int(selected_sample_idx)
    sample_row = X_df.iloc[sample_idx]
    shap_row = np.asarray(shap_matrix[sample_idx, :], dtype=float)

    with st.spinner("Computing weighted leaf contributions for selected sample..."):
        weighted_contribs, tree_records = _compute_sample_weighted_leaf_contributions(
            booster=booster,
            sample_row=sample_row,
            feature_names=feature_names,
            depth_emphasis=float(depth_emphasis),
            max_trees=int(max_trees),
        )

    tree_vector = np.zeros(n_features, dtype=float)
    for idx, val in weighted_contribs.items():
        if 0 <= int(idx) < n_features:
            tree_vector[int(idx)] = float(val)

    sample_pred_proba = None
    model = getattr(summary_traj, "model", None)
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if estimator is not None and hasattr(estimator, "predict_proba"):
        pred = np.asarray(estimator.predict_proba(sample_row.to_frame().T), dtype=float)
        if pred.ndim == 2 and pred.shape[0] > 0 and pred.shape[1] > 0:
            sample_pred_proba = float(pred[0, -1])

    total_leaf_sum = float(sum(float(r.get("leaf_value", 0.0)) for r in tree_records))
    shap_l1 = float(np.sum(np.abs(shap_row)))
    tree_l1 = float(np.sum(np.abs(tree_vector)))
    shap_tree_corr = _safe_corr(shap_row, tree_vector)

    st.subheader("Sample Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sample", sample_idx)
    m2.metric("Prediction p(y=1)", f"{sample_pred_proba:.4f}" if sample_pred_proba is not None else "N/A")
    if y_arr is not None and sample_idx < len(y_arr):
        m3.metric("True label", int(y_arr[sample_idx]))
    else:
        m3.metric("True label", "N/A")
    m4.metric("Leaf sum (margin)", f"{total_leaf_sum:.4f}")
    m5.metric("corr(SHAP, weighted-leaf)", f"{shap_tree_corr:.3f}")

    s1, s2 = st.columns(2)
    s1.metric("SHAP L1 norm", f"{shap_l1:.4f}")
    s2.metric("Weighted-leaf L1 norm", f"{tree_l1:.4f}")

    rows: List[Dict[str, Any]] = []
    for idx in range(n_features):
        shap_val = float(shap_row[idx])
        tree_val = float(tree_vector[idx])
        rows.append(
            {
                "feature_idx": int(idx),
                "feature": str(readable_map.get(idx, feature_names[idx])),
                "sample_value": float(sample_row.iloc[idx]) if pd.notna(sample_row.iloc[idx]) else np.nan,
                "shap": shap_val,
                "weighted_leaf": tree_val,
                "abs_shap": abs(shap_val),
                "abs_weighted_leaf": abs(tree_val),
                "same_direction": bool(np.sign(shap_val) == np.sign(tree_val)) if (shap_val != 0.0 and tree_val != 0.0) else False,
            }
        )

    contrib_df = pd.DataFrame(rows).sort_values("abs_weighted_leaf", ascending=False)

    st.subheader("Feature-Level Contributions For This Sample")
    top_k = st.slider(
        "Top features to display",
        min_value=5,
        max_value=min(40, n_features),
        value=min(15, n_features),
        step=1,
        key="sample_viewer_top_k",
    )
    view_df = contrib_df.head(int(top_k)).copy()
    st.dataframe(view_df, width="content")

    if not view_df.empty:
        labels = view_df["feature"].astype(str).tolist()[::-1]
        shap_vals = view_df["shap"].astype(float).to_numpy()[::-1]
        tree_vals = view_df["weighted_leaf"].astype(float).to_numpy()[::-1]

        fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=130)
        y_pos = np.arange(len(labels))
        h = 0.38
        ax.barh(y_pos - h / 2, shap_vals, height=h, label="SHAP", alpha=0.85)
        ax.barh(y_pos + h / 2, tree_vals, height=h, label="Weighted leaf", alpha=0.85)
        ax.axvline(x=0.0, color="#888888", lw=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Contribution value")
        ax.set_title("Per-feature contributions for selected sample")
        ax.legend(loc="best")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, width="content")

    st.subheader("Most Frequent Decision Paths For This Sample")
    paths_df = _aggregate_path_frequencies(tree_records, readable_by_raw)
    if paths_df.empty:
        st.info("No decision paths were collected for this sample.")
    else:
        top_paths_k = st.slider(
            "Top frequent paths to display",
            min_value=3,
            max_value=min(30, int(paths_df.shape[0])),
            value=min(10, int(paths_df.shape[0])),
            step=1,
            key="sample_viewer_top_paths_k",
        )
        show_df = paths_df.head(int(top_paths_k)).copy()
        st.dataframe(
            show_df[
                [
                    "frequency",
                    "path_length",
                    "leaf_mean",
                    "leaf_sum",
                    "leaf_abs_sum",
                    "leaf_min",
                    "leaf_max",
                    "positive_leaf_rate",
                    "path_signature",
                ]
            ],
            width="content",
        )

        selected_path_sig = st.selectbox(
            "Inspect one frequent path",
            options=show_df["path_signature"].astype(str).tolist(),
            key="sample_viewer_selected_path_signature",
        )
        selected_path_df = paths_df[paths_df["path_signature"] == selected_path_sig]
        if not selected_path_df.empty:
            selected_path = selected_path_df.iloc[0]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Frequency", int(selected_path.get("frequency", 0)))
            p2.metric("Mean leaf", f"{float(selected_path.get('leaf_mean', 0.0)):.4f}")
            p3.metric("Leaf sum", f"{float(selected_path.get('leaf_sum', 0.0)):.4f}")
            p4.metric("Positive leaf rate", f"{float(selected_path.get('positive_leaf_rate', 0.0)):.2f}")
            st.caption(
                "Trees using this path: "
                + ", ".join(str(int(i)) for i in selected_path.get("tree_indices", []))
            )

    st.subheader("Per-Feature Tree Path Details")
    feature_options = view_df["feature"].astype(str).tolist() if not view_df.empty else contrib_df["feature"].astype(str).tolist()
    if feature_options:
        selected_feature_name = st.selectbox(
            "Inspect one feature across trees",
            options=feature_options,
            key="sample_viewer_feature_detail",
        )
        selected_idx = int(
            contrib_df.loc[contrib_df["feature"] == selected_feature_name, "feature_idx"].iloc[0]
        )

        detail_rows: List[Dict[str, Any]] = []
        for rec in tree_records:
            alloc = rec.get("alloc", {})
            feature_contrib = float(alloc.get(selected_idx, 0.0))
            if abs(feature_contrib) <= 0.0:
                continue

            first_hit = None
            for step in rec.get("steps", []):
                if str(step.get("feature_name", "")) == feature_names[selected_idx]:
                    first_hit = step
                    break

            detail_rows.append(
                {
                    "tree_idx": int(rec.get("tree_idx", 0)),
                    "feature_contribution": feature_contrib,
                    "leaf_value": float(rec.get("leaf_value", 0.0)),
                    "path_length": int(rec.get("path_length", 0)),
                    "sample_value_at_split": (
                        float(first_hit.get("sample_value")) if first_hit is not None else np.nan
                    ),
                    "split_op_taken": str(first_hit.get("op_taken")) if first_hit is not None else "N/A",
                    "split_threshold": (
                        float(first_hit.get("threshold")) if first_hit is not None else np.nan
                    ),
                }
            )

        details_df = pd.DataFrame(detail_rows).sort_values(
            "feature_contribution",
            key=lambda s: np.abs(s.astype(float)),
            ascending=False,
        )

        if details_df.empty:
            st.info("The selected feature did not appear on this sample's evaluated tree paths.")
        else:
            st.dataframe(details_df.head(40), width="content")

    with st.expander("Raw sample values", expanded=False):
        raw_df = pd.DataFrame(
            {
                "feature": [str(readable_map.get(i, name)) for i, name in enumerate(feature_names)],
                "value": [sample_row.iloc[i] for i in range(n_features)],
            }
        )
        st.dataframe(raw_df, width="content")
