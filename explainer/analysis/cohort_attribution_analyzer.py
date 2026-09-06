"""Sub-population / cohort attribution from decision-path contexts.

Builds human-readable cohorts from repeated ``scope -> decider`` path contexts:
for a given decider feature, which scope conditions repeatedly narrow the sample
before that feature performs the final split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .trajectory_utils import round_float


class CohortAttributionAnalyzer:
    """Extract, aggregate, and score cohort attribution contexts."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = [str(x) for x in feature_names]
        self._feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_path_rules(self, tree_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract root-to-leaf path rules from raw XGBoost JSON trees."""
        rules: List[Dict[str, Any]] = []
        for entry in tree_entries:
            tree_raw = entry.get("tree")
            if tree_raw is None:
                continue
            tree_obj = json.loads(tree_raw) if isinstance(tree_raw, str) else tree_raw
            iteration = int(entry.get("iteration", 0))
            self._walk_tree_to_rules(tree_obj, iteration, [], rules)
        return rules

    def assign_samples_to_rules(
        self,
        X: pd.DataFrame,
        rule_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Attach sample assignments/support to each rule."""
        if X.empty:
            return []

        assigned: List[Dict[str, Any]] = []
        x_values = X.to_numpy()
        n_samples = x_values.shape[0]

        for rule in rule_records:
            conds = rule.get("conditions", [])
            if not conds:
                continue
            mask = np.ones(n_samples, dtype=bool)
            for cond in conds:
                feature_idx = int(cond.get("feature_idx", -1))
                if feature_idx < 0 or feature_idx >= x_values.shape[1]:
                    mask = np.zeros(n_samples, dtype=bool)
                    break
                threshold = float(cond.get("threshold", 0.0))
                col = x_values[:, feature_idx]
                if cond.get("op") == "<":
                    mask &= col < threshold
                else:
                    mask &= col >= threshold

            sample_ids = np.flatnonzero(mask).astype(int).tolist()
            if not sample_ids:
                continue

            row = dict(rule)
            row["sample_ids"] = sample_ids
            row["support_count"] = int(len(sample_ids))
            row["support_rate"] = round_float(len(sample_ids) / max(n_samples, 1))
            assigned.append(row)

        return assigned

    def build_cohorts(
        self,
        assigned_rules: List[Dict[str, Any]],
        *,
        min_support: int = 50,
        min_paths: int = 5,
    ) -> List[Dict[str, Any]]:
        """Aggregate assigned rules into cohorts by decider + scope signature."""
        groups: Dict[Tuple[int, str], Dict[str, Any]] = {}
        decider_rule_counts: Dict[int, int] = defaultdict(int)
        global_abs_leaf_baseline = float(
            np.mean([abs(float(r.get("leaf_value", 0.0))) for r in assigned_rules])
        ) if assigned_rules else 0.0

        for rule in assigned_rules:
            decider_idx = int(rule.get("decider_feature_idx", -1))
            if decider_idx < 0:
                continue
            decider_rule_counts[decider_idx] += 1
            scope_sig = str(rule.get("scope_signature", ""))
            key = (decider_idx, scope_sig)
            if key not in groups:
                groups[key] = {
                    "decider_feature_idx": decider_idx,
                    "decider_feature_name": str(
                        rule.get("decider_feature_name", self._feature_name(decider_idx))
                    ),
                    "scope_signature": scope_sig,
                    "scope_conditions": list(rule.get("scope_conditions", [])),
                    "path_occurrences": 0,
                    "support_ids": set(),
                    "leaf_values": [],
                    "top_paths": [],
                }

            grp = groups[key]
            grp["path_occurrences"] += 1
            grp["support_ids"].update(int(i) for i in rule.get("sample_ids", []))
            grp["leaf_values"].append(float(rule.get("leaf_value", 0.0)))
            grp["top_paths"].append(
                {
                    "iteration": int(rule.get("iteration", 0)),
                    "support_count": int(rule.get("support_count", 0)),
                    "leaf_value": round_float(float(rule.get("leaf_value", 0.0))),
                    "path_conditions": list(rule.get("conditions", [])),
                }
            )

        cohorts: List[Dict[str, Any]] = []
        for grp in groups.values():
            support_count = int(len(grp["support_ids"]))
            if support_count < min_support:
                continue
            if int(grp["path_occurrences"]) < min_paths:
                continue

            decider_idx = int(grp["decider_feature_idx"])
            decider_total = max(int(decider_rule_counts.get(decider_idx, 1)), 1)
            late_decider_rate_local = float(grp["path_occurrences"]) / decider_total
            mean_leaf = float(np.mean(grp["leaf_values"])) if grp["leaf_values"] else 0.0
            decision_impact_local_raw = late_decider_rate_local * abs(mean_leaf)
            baseline = max(global_abs_leaf_baseline, 1e-9)
            impact_magnitude_ratio_local = abs(mean_leaf) / baseline
            # Interpretable scale: frequency as local decider × relative leaf-impact magnitude.
            decision_impact_local = late_decider_rate_local * impact_magnitude_ratio_local

            top_paths = sorted(
                grp["top_paths"],
                key=lambda r: (-int(r["support_count"]), -abs(float(r["leaf_value"]))),
            )[:5]

            cohorts.append(
                {
                    "decider_feature_idx": decider_idx,
                    "decider_feature_name": str(grp["decider_feature_name"]),
                    "scope_signature": str(grp["scope_signature"]),
                    "scope_conditions": list(grp["scope_conditions"]),
                    "support_ids": set(grp["support_ids"]),
                    "support_count": support_count,
                    "path_occurrences": int(grp["path_occurrences"]),
                    "mean_leaf_value": mean_leaf,
                    "impact_magnitude_ratio_local": impact_magnitude_ratio_local,
                    "decision_impact_local_raw": decision_impact_local_raw,
                    "late_decider_rate_local": late_decider_rate_local,
                    "decision_impact_local": decision_impact_local,
                    "top_paths": top_paths,
                }
            )

        return cohorts

    def enrich_cohorts_with_metrics(
        self,
        cohorts: List[Dict[str, Any]],
        *,
        y_true: Sequence[Any],
        y_proba: Optional[Sequence[float]] = None,
        final_shap_values: Optional[Any] = None,
        top_local_shap: int = 8,
    ) -> List[Dict[str, Any]]:
        """Attach cohort-level risk/confidence and optional local SHAP summaries."""
        y_arr = _coerce_target(y_true)
        proba_arr = np.asarray(y_proba, dtype=float) if y_proba is not None else None
        shap_arr = _coerce_shap(final_shap_values) if final_shap_values is not None else None

        enriched: List[Dict[str, Any]] = []
        n_total = max(len(y_arr), 1)
        for cohort in cohorts:
            ids = sorted(int(i) for i in cohort.get("support_ids", set()) if int(i) < len(y_arr))
            if not ids:
                continue
            y_slice = y_arr[ids]
            support_count = len(ids)
            support_rate = float(support_count) / n_total
            risk_rate = float(np.mean(y_slice)) if y_slice.size else 0.0

            mean_conf = None
            conf_std = None
            if proba_arr is not None and len(proba_arr) >= max(ids) + 1:
                p_slice = proba_arr[ids]
                mean_conf = float(np.mean(p_slice))
                conf_std = float(np.std(p_slice))

            local_shap_summary: List[Dict[str, Any]] = []
            if shap_arr is not None and shap_arr.shape[0] >= max(ids) + 1:
                local_shap_summary = _local_shap_summary(
                    shap_arr[ids, :],
                    self.feature_names,
                    top_k=top_local_shap,
                )

            row = dict(cohort)
            row["support_rate"] = round_float(support_rate)
            row["risk_rate"] = round_float(risk_rate)
            row["mean_confidence"] = round_float(mean_conf) if mean_conf is not None else None
            row["confidence_std"] = round_float(conf_std) if conf_std is not None else None
            row["sample_id_preview"] = ids[:20]
            row["local_shap_summary"] = local_shap_summary
            # keep internal set out of final schema; leave until to_json_schema
            enriched.append(row)

        return enriched

    def to_json_schema(
        self,
        cohorts: List[Dict[str, Any]],
        *,
        top_k: int = 20,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize cohorts to ``cohort_attribution_v1`` schema."""
        ranked = sorted(
            cohorts,
            key=lambda c: (
                -float(c.get("support_count", 0)) * float(c.get("decision_impact_local", 0.0)),
                -float(c.get("support_count", 0)),
            ),
        )[: max(int(top_k), 1)]

        exported: List[Dict[str, Any]] = []
        for idx, cohort in enumerate(ranked, start=1):
            exported.append(
                {
                    "cohort_id": f"C{idx:03d}",
                    "decider_feature_idx": int(cohort["decider_feature_idx"]),
                    "decider_feature_name": str(cohort["decider_feature_name"]),
                    "scope_signature": str(cohort["scope_signature"]),
                    "scope_conditions": list(cohort.get("scope_conditions", [])),
                    "support_count": int(cohort.get("support_count", 0)),
                    "support_rate": round_float(float(cohort.get("support_rate", 0.0))),
                    "path_occurrences": int(cohort.get("path_occurrences", 0)),
                    "risk_rate": round_float(float(cohort.get("risk_rate", 0.0))),
                    "mean_confidence": (
                        round_float(float(cohort["mean_confidence"]))
                        if cohort.get("mean_confidence") is not None
                        else None
                    ),
                    "confidence_std": (
                        round_float(float(cohort["confidence_std"]))
                        if cohort.get("confidence_std") is not None
                        else None
                    ),
                    "decision_impact_local": round_float(float(cohort.get("decision_impact_local", 0.0))),
                    "decision_impact_local_raw": round_float(float(cohort.get("decision_impact_local_raw", 0.0))),
                    "impact_magnitude_ratio_local": round_float(float(cohort.get("impact_magnitude_ratio_local", 0.0))),
                    "late_decider_rate_local": round_float(float(cohort.get("late_decider_rate_local", 0.0))),
                    "mean_leaf_value": round_float(float(cohort.get("mean_leaf_value", 0.0))),
                    "sample_id_preview": list(cohort.get("sample_id_preview", [])),
                    "top_paths": list(cohort.get("top_paths", [])),
                    "local_shap_summary": list(cohort.get("local_shap_summary", [])),
                }
            )

        return {
            "schema": "cohort_attribution_v1",
            "metadata": {
                "cohort_count": len(exported),
                **(metadata or {}),
            },
            "cohorts": exported,
        }

    # ------------------------------------------------------------------
    # Internal tree walk helpers
    # ------------------------------------------------------------------

    def _walk_tree_to_rules(
        self,
        node: Any,
        iteration: int,
        path_conditions: List[Dict[str, Any]],
        out_rules: List[Dict[str, Any]],
    ) -> None:
        if not isinstance(node, dict):
            return

        if "leaf" in node:
            if not path_conditions:
                return
            decider = path_conditions[-1]
            scope_conditions = path_conditions[:-1]
            out_rules.append(
                {
                    "iteration": int(iteration),
                    "leaf_value": float(node.get("leaf", 0.0)),
                    "conditions": list(path_conditions),
                    "scope_conditions": list(scope_conditions),
                    "decider_feature_idx": int(decider.get("feature_idx", -1)),
                    "decider_feature_name": str(decider.get("feature_name", "")),
                    "scope_signature": _scope_signature(scope_conditions),
                }
            )
            return

        split_name = str(node.get("split", ""))
        feature_idx = int(self._feature_to_idx.get(split_name, -1))
        threshold = float(node.get("split_condition", 0.0))
        feature_name = split_name if split_name else self._feature_name(feature_idx)

        children = node.get("children", [])
        if not isinstance(children, list) or not children:
            return
        child_by_id = {
            int(child.get("nodeid", -1)): child
            for child in children
            if isinstance(child, dict)
        }
        yes_id = int(node.get("yes", -1))
        no_id = int(node.get("no", -1))
        yes_child = child_by_id.get(yes_id, children[0] if len(children) > 0 else None)
        no_child = child_by_id.get(no_id, children[1] if len(children) > 1 else None)

        if yes_child is not None:
            yes_cond = {
                "feature_idx": feature_idx,
                "feature_name": feature_name,
                "op": "<",
                "threshold": round_float(threshold),
                "depth": len(path_conditions),
            }
            self._walk_tree_to_rules(
                yes_child,
                iteration,
                path_conditions + [yes_cond],
                out_rules,
            )

        if no_child is not None:
            no_cond = {
                "feature_idx": feature_idx,
                "feature_name": feature_name,
                "op": ">=",
                "threshold": round_float(threshold),
                "depth": len(path_conditions),
            }
            self._walk_tree_to_rules(
                no_child,
                iteration,
                path_conditions + [no_cond],
                out_rules,
            )

    def _feature_name(self, idx: int) -> str:
        if 0 <= idx < len(self.feature_names):
            return str(self.feature_names[idx])
        return str(idx)


def _scope_signature(scope_conditions: Iterable[Dict[str, Any]]) -> str:
    """Normalize scope conditions into a compact deterministic signature."""
    parts = []
    for cond in scope_conditions:
        name = str(cond.get("feature_name", cond.get("feature_idx", "")))
        op = str(cond.get("op", ""))
        thr = round_float(float(cond.get("threshold", 0.0)))
        parts.append(f"{name} {op} {thr}")
    return " AND ".join(parts)


def _coerce_target(y: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 0:
        return arr.reshape(1).astype(float)
    if arr.ndim == 1:
        return arr.astype(float)
    return arr.reshape(arr.shape[0], -1)[:, 0].astype(float)


def _coerce_shap(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return np.asarray(arr[:, :, -1])
    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


def _local_shap_summary(
    shap_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    if shap_matrix.size == 0:
        return []
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][: max(int(top_k), 1)]
    rows: List[Dict[str, Any]] = []
    for idx in top_idx:
        fi = int(idx)
        rows.append(
            {
                "feature_idx": fi,
                "feature_name": str(feature_names[fi]) if fi < len(feature_names) else str(fi),
                "mean_abs_shap": round_float(float(mean_abs[fi])),
            }
        )
    return rows
