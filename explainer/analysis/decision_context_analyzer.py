"""Analyzes how features are positioned within XGBoost decision-tree paths.

For every root-to-leaf path across all tree iterations, this module tracks:
- Whether each feature acts as an *early gatekeeper* (top third of the path)
  or a *late decider* (bottom third, just before the leaf).
- Which features precede it in the path (*scope features*).
- Which features appear immediately before it near the leaf (*near-leaf partners*).
- The leaf values associated with each role.

These signals are combined into a ``decision_impact_score`` that surfaces
features with outsized late-stage influence, and identifies
*OUTLIER SPECIALIST* candidates from the behavioral quadrant taxonomy.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .behavioral_quadrants import BehavioralQuadrantAnalyzer
from .decision_graph import build_decision_narrowing_graph
from .trajectory_utils import round_float
from .tree_walker import collect_split_path_records


class DecisionContextAnalyzer:
    """Analyses feature co-occurrence contexts along tree decision paths."""

    def __init__(
        self,
        feature_names: List[str],
        quadrant_analyzer: BehavioralQuadrantAnalyzer,
    ) -> None:
        self.feature_names = feature_names
        self._quadrant_analyzer = quadrant_analyzer

    def analyze(
        self,
        tree_entries: List[Dict[str, Any]],
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Analyze feature co-occurrence contexts along tree decision paths.

        Tracks where each feature appears in root-to-leaf paths, how often it
        acts as a late decider, and which preceding features define the scope
        before it splits.

        Args:
            tree_entries: List of iteration dicts, each with a ``'tree'`` key
                          containing a raw XGBoost JSON tree (string or dict).
            trajectory_metrics: Optional output of
                                 ``TreeAnalyzer.compute_trajectory_metrics()``.
                                 When provided, each feature profile is enriched
                                 with its behavioral quadrant category.
            top_k: Maximum number of scope / partner features to retain per feature.

        Returns:
            Dictionary with:
            - ``tree_count``: number of trees processed
            - ``iteration_threshold_for_late_training``: median iteration index
            - ``global_mean_abs_leaf_value``: baseline for impact ratio computation
            - ``feature_profiles``: list of per-feature context profiles, sorted by
              ``decision_impact_score`` descending
            - ``outlier_specialist_candidates``: subset flagged as OUTLIER SPECIALISTs
              with high late-decider rate and strong impact
            - ``top_scope_to_decider_pairs``: global ranking of (scope → decider)
              co-occurrence counts
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not tree_entries:
            return {
                "tree_count": 0,
                "feature_profiles": [],
                "outlier_specialist_candidates": [],
            }

        import json

        feature_to_index = {name: i for i, name in enumerate(self.feature_names)}
        per_feature: Dict[int, Any] = defaultdict(
            lambda: {
                "path_count": 0,
                "late_decider_count": 0,
                "early_presence_count": 0,
                "iteration_hits": [],
                "late_iteration_hits": [],
                "leaf_values_all_paths": [],
                "leaf_values_late_decider": [],
                "scope_counter": Counter(),
                "near_leaf_partner_counter": Counter(),
            }
        )
        scope_pair_counter: Counter = Counter()
        all_iterations: List[int] = []
        all_leaf_values: List[float] = []

        for entry in tree_entries:
            tree_raw = entry.get("tree")
            if tree_raw is None:
                continue

            tree_obj = json.loads(tree_raw) if isinstance(tree_raw, str) else tree_raw
            iteration = int(entry.get("iteration", 0))
            all_iterations.append(iteration)

            path_records: List[Dict[str, Any]] = []
            collect_split_path_records(
                node=tree_obj,
                depth=0,
                current_path=[],
                path_records=path_records,
                feature_to_index=feature_to_index,
            )

            for record in path_records:
                path = list(record.get("path", []))
                if not path:
                    continue

                leaf_value = float(record.get("leaf_value", 0.0))
                all_leaf_values.append(leaf_value)
                path_features = [item[0] for item in path]

                for feature_idx in sorted(set(path_features)):
                    positions = [i for i, fi in enumerate(path_features) if fi == feature_idx]
                    first_pos = positions[0]
                    last_pos = positions[-1]
                    path_len = len(path_features)
                    first_ratio = (first_pos + 1) / max(path_len, 1)
                    last_ratio = (last_pos + 1) / max(path_len, 1)

                    stats = per_feature[feature_idx]
                    stats["path_count"] += 1
                    stats["iteration_hits"].append(iteration)
                    stats["leaf_values_all_paths"].append(leaf_value)

                    if first_ratio <= 1.0 / 3.0:
                        stats["early_presence_count"] += 1

                    if last_ratio >= 2.0 / 3.0:
                        stats["late_decider_count"] += 1
                        stats["late_iteration_hits"].append(iteration)
                        stats["leaf_values_late_decider"].append(leaf_value)

                    for scope_idx in sorted(set(path_features[:last_pos])):
                        stats["scope_counter"][scope_idx] += 1
                        scope_pair_counter[(scope_idx, feature_idx)] += 1

                    near_leaf_partners = path_features[max(0, last_pos - 2): last_pos]
                    for partner_idx in sorted(set(near_leaf_partners)):
                        stats["near_leaf_partner_counter"][partner_idx] += 1

        if not per_feature:
            return {
                "tree_count": int(len(tree_entries)),
                "feature_profiles": [],
                "outlier_specialist_candidates": [],
            }

        threshold_iter = float(np.median(all_iterations)) if all_iterations else 0.0
        global_mean_abs_leaf = float(np.mean(np.abs(all_leaf_values))) if all_leaf_values else 0.0

        # Build quadrant category map if trajectory data is available.
        category_map: Dict[int, str] = {}
        if trajectory_metrics:
            categories = self._quadrant_analyzer.categorize_features(trajectory_metrics)
            category_map = {
                int(item["feature_idx"]): str(item["quadrant_category"])
                for item in categories.get("features", [])
            }

        feature_profiles = []
        for feature_idx, stats in per_feature.items():
            path_count = int(stats["path_count"])
            if path_count <= 0:
                continue

            profile = self._build_feature_profile(
                feature_idx=feature_idx,
                stats=stats,
                path_count=path_count,
                threshold_iter=threshold_iter,
                global_mean_abs_leaf=global_mean_abs_leaf,
                category_map=category_map,
                top_k=top_k,
            )
            feature_profiles.append(profile)

        feature_profiles.sort(
            key=lambda x: (-x["decision_impact_score"], -x["late_decider_rate"], -x["path_count"], x["feature_idx"])
        )

        outlier_candidates = [
            {
                "feature_idx": int(p["feature_idx"]),
                "feature_name": str(p["feature_name"]),
                "path_count": int(p["path_count"]),
                "late_decider_rate": round_float(float(p["late_decider_rate"])),
                "late_in_late_training_rate": round_float(float(p["late_in_late_training_rate"])),
                "decision_impact_score": round_float(float(p["decision_impact_score"])),
                "mean_abs_leaf_when_late_decider": round_float(float(p["mean_abs_leaf_when_late_decider"])),
                "scope_entropy": round_float(float(p["scope_entropy"])),
                "top_scope_features": p["top_scope_features"][: min(5, top_k)],
            }
            for p in feature_profiles
            if p["outlier_specialist_candidate"]
        ]

        top_scope_pairs = [
            {
                "scope_feature_idx": int(scope_idx),
                "decider_feature_idx": int(decider_idx),
                "cooccurrence_count": int(count),
            }
            for (scope_idx, decider_idx), count in sorted(
                scope_pair_counter.items(), key=lambda kv: (-kv[1], kv[0])
            )[: max(10, top_k)]
        ]

        return {
            "tree_count": int(len(tree_entries)),
            "iteration_threshold_for_late_training": round_float(float(threshold_iter)),
            "global_mean_abs_leaf_value": round_float(float(global_mean_abs_leaf)),
            "feature_profiles": [_round_profile(p) for p in feature_profiles],
            "outlier_specialist_candidates": outlier_candidates,
            "top_scope_to_decider_pairs": top_scope_pairs,
        }

    # ------------------------------------------------------------------
    # Decision-narrowing graph
    # ------------------------------------------------------------------

    def build_decision_narrowing_graph(
        self,
        tree_entries: List[Dict[str, Any]],
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        *,
        decision_context: Optional[Dict[str, Any]] = None,
        top_deciders: int = 15,
        max_scope_per_decider: int = 5,
        min_cooccurrence: int = 1,
        max_edges: int = 60,
    ) -> Dict[str, Any]:
        """Build a directed decision-narrowing graph.

        Nodes are features (annotated with late-decider behaviour), and edges
        ``scope_feature -> decider_feature`` capture how earlier features narrow
        the decision scope before a late decider splits.

        Args:
            tree_entries: Iteration dicts with raw XGBoost JSON trees. Ignored
                when ``decision_context`` is supplied.
            trajectory_metrics: Optional trajectory metrics used to enrich the
                underlying decision-context analysis with quadrant categories.
            decision_context: Pre-computed output of :meth:`analyze`. When
                provided, the tree walk is skipped and this is reused directly.
            top_deciders: Number of top deciders (by decision impact) to link.
            max_scope_per_decider: Max incoming scope edges kept per decider.
            min_cooccurrence: Minimum co-occurrence count for an edge.
            max_edges: Global cap on the number of edges.

        Returns:
            A JSON-serializable decision-narrowing graph dict.
        """
        if decision_context is None:
            decision_context = self.analyze(
                tree_entries,
                trajectory_metrics=trajectory_metrics,
                top_k=max(top_deciders, max_scope_per_decider),
            )

        return build_decision_narrowing_graph(
            decision_context,
            top_deciders=top_deciders,
            max_scope_per_decider=max_scope_per_decider,
            min_cooccurrence=min_cooccurrence,
            max_edges=max_edges,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_profile(
        self,
        feature_idx: int,
        stats: Dict[str, Any],
        path_count: int,
        threshold_iter: float,
        global_mean_abs_leaf: float,
        category_map: Dict[int, str],
        top_k: int,
    ) -> Dict[str, Any]:
        late_decider_rate = stats["late_decider_count"] / path_count
        early_presence_rate = stats["early_presence_count"] / path_count

        late_hits = np.asarray(stats["late_iteration_hits"], dtype=float)
        late_in_early_rate = float(np.mean(late_hits <= threshold_iter)) if late_hits.size else 0.0
        late_in_late_rate = float(np.mean(late_hits > threshold_iter)) if late_hits.size else 0.0

        leaf_all = np.asarray(stats["leaf_values_all_paths"], dtype=float)
        leaf_late = np.asarray(stats["leaf_values_late_decider"], dtype=float)

        mean_leaf = float(np.mean(leaf_all)) if leaf_all.size else 0.0
        mean_abs_leaf = float(np.mean(np.abs(leaf_all))) if leaf_all.size else 0.0
        mean_leaf_late = float(np.mean(leaf_late)) if leaf_late.size else 0.0
        mean_abs_leaf_late = float(np.mean(np.abs(leaf_late))) if leaf_late.size else 0.0
        pos_rate_late = float(np.mean(leaf_late > 0)) if leaf_late.size else 0.0
        neg_rate_late = float(np.mean(leaf_late < 0)) if leaf_late.size else 0.0

        baseline = max(global_mean_abs_leaf, 1e-9)
        impact_ratio = mean_abs_leaf_late / baseline
        directional_shift = abs(mean_leaf_late - mean_leaf)
        decision_impact_score = late_decider_rate * impact_ratio * (1.0 + directional_shift)

        top_scope = sorted(stats["scope_counter"].items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        top_partners = sorted(stats["near_leaf_partner_counter"].items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

        scope_total = float(sum(stats["scope_counter"].values()))
        scope_entropy = 0.0
        if scope_total > 0:
            for _, cnt in stats["scope_counter"].items():
                p = cnt / scope_total
                scope_entropy -= p * float(np.log2(max(p, 1e-12)))

        category = category_map.get(feature_idx)
        outlier_specialist_candidate = bool(
            category == "OUTLIER SPECIALIST"
            and late_decider_rate >= 0.4
            and len(top_scope) > 0
            and decision_impact_score >= 0.5
        )

        feature_name = (
            self.feature_names[feature_idx]
            if feature_idx < len(self.feature_names)
            else str(feature_idx)
        )

        return {
            "feature_idx": int(feature_idx),
            "feature_name": feature_name,
            "quadrant_category": category,
            "path_count": path_count,
            "late_decider_rate": late_decider_rate,
            "early_presence_rate": early_presence_rate,
            "late_in_early_training_rate": late_in_early_rate,
            "late_in_late_training_rate": late_in_late_rate,
            "mean_leaf_value_when_present": mean_leaf,
            "mean_abs_leaf_when_present": mean_abs_leaf,
            "mean_leaf_value_when_late_decider": mean_leaf_late,
            "mean_abs_leaf_when_late_decider": mean_abs_leaf_late,
            "positive_leaf_rate_when_late_decider": pos_rate_late,
            "negative_leaf_rate_when_late_decider": neg_rate_late,
            "impact_magnitude_ratio": impact_ratio,
            "decision_impact_score": decision_impact_score,
            "scope_entropy": scope_entropy,
            "top_scope_features": [
                {
                    "feature_idx": int(si),
                    "cooccurrence_count": int(cnt),
                    "cooccurrence_rate": float(cnt / path_count),
                }
                for si, cnt in top_scope
            ],
            "top_near_leaf_partners": [
                {
                    "feature_idx": int(pi),
                    "cooccurrence_count": int(cnt),
                    "cooccurrence_rate": float(cnt / path_count),
                }
                for pi, cnt in top_partners
            ],
            "outlier_specialist_candidate": outlier_specialist_candidate,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_FLOAT_FIELDS = (
    "late_decider_rate", "early_presence_rate",
    "late_in_early_training_rate", "late_in_late_training_rate",
    "mean_leaf_value_when_present", "mean_abs_leaf_when_present",
    "mean_leaf_value_when_late_decider", "mean_abs_leaf_when_late_decider",
    "positive_leaf_rate_when_late_decider", "negative_leaf_rate_when_late_decider",
    "impact_magnitude_ratio", "decision_impact_score", "scope_entropy",
)


def _round_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *profile* with all float fields rounded."""
    out = dict(profile)
    for field in _FLOAT_FIELDS:
        if field in out:
            out[field] = round_float(float(out[field]))
    return out
