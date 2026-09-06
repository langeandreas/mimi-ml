"""Tree structure analysis and summarization.

``TreeAnalyzer`` is the single public entry-point for all tree-based analysis.
It owns the two data-extraction methods (``summarize_tree_step`` and
``compute_trajectory_metrics``) and delegates every higher-level operation to
a set of focused helper classes:

    ┌──────────────────────────┐
    │       TreeAnalyzer       │  ← public API (backward-compatible)
    └──────┬───────────────────┘
           │ delegates to
    ┌──────▼──────────────────────────────────────────┐
    │ BehavioralQuadrantAnalyzer  (behavioral_quadrants.py)  │
    │ DecisionContextAnalyzer     (decision_context_analyzer.py) │
    │ ShapBehaviorCorrelator      (shap_behavior_correlator.py)  │
    │ plot_behavioral_signatures  (behavioral_plotter.py)        │
    └────────────────────────────────────────────────────────────┘

Tree traversal primitives live in tree_walker.py and are shared between
this module and DecisionContextAnalyzer.
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .behavioral_plotter import plot_behavioral_signatures
from .behavioral_quadrants import BehavioralQuadrantAnalyzer
from .cohort_attribution_analyzer import CohortAttributionAnalyzer
from .decision_context_analyzer import DecisionContextAnalyzer
from .shap_behavior_correlator import ShapBehaviorCorrelator
from .trajectory_utils import round_float, TOP_K_TREE_FEATURES, TOP_K_INTERACTIONS
from .tree_walker import walk_tree


class TreeAnalyzer:
    """Analyses tree structures and generates tree-based summaries.

    Instantiate once per classification object and call any combination of
    the public methods.  All methods that accept ``trajectory_metrics`` expect
    the dict returned by :meth:`compute_trajectory_metrics`.
    """

    def __init__(self, classification) -> None:
        self.classification = classification
        self.feature_names = [str(c) for c in classification.train_test["X_train"].columns]

        # Storage for trajectory analysis (retained for external inspection).
        self.feature_gains_trajectory = defaultdict(list)
        self.feature_cover_trajectory = defaultdict(list)
        self.feature_split_counts_trajectory = defaultdict(list)
        self.tree_summaries: List[Dict[str, Any]] = []

        # Delegate objects — not part of the public API.
        self._quadrant_analyzer = BehavioralQuadrantAnalyzer(self.feature_names)
        self._context_analyzer = DecisionContextAnalyzer(self.feature_names, self._quadrant_analyzer)
        self._cohort_analyzer = CohortAttributionAnalyzer(self.feature_names)
        self._shap_correlator = ShapBehaviorCorrelator(self.feature_names)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def summarize_tree_step(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Summarise a single boosting iteration's tree.

        Performs a depth-first traversal to collect:
        - total split and leaf counts
        - maximum tree depth
        - top-K most-used split features with their average split depth
        - top-K feature co-occurrence pairs (appear together on the same
          root-to-leaf path)
        - leaf value statistics (mean / std / min / max)

        Args:
            entry: Dict with at least ``'iteration'`` and ``'tree'`` keys.
                   ``'tree'`` may be a JSON string or an already-parsed dict.

        Returns:
            Summary dict, or *None* when no tree is present in *entry*.
        """
        tree_raw = entry.get("tree")
        if tree_raw is None:
            return None

        tree_obj = json.loads(tree_raw) if isinstance(tree_raw, str) else tree_raw

        split_counter: Counter = Counter()
        split_depth_sum: Counter = Counter()
        pair_counter: Counter = Counter()
        leaf_values: List[float] = []
        feature_to_index = {name: i for i, name in enumerate(self.feature_names)}

        max_depth = walk_tree(
            tree_obj, 0, [],
            split_counter, split_depth_sum, pair_counter,
            leaf_values, feature_to_index,
        )

        top_features = sorted(split_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_TREE_FEATURES]
        top_pairs = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_INTERACTIONS]

        if leaf_values:
            leaf_stats = {
                "mean": round_float(float(np.mean(leaf_values))),
                "std": round_float(float(np.std(leaf_values))),
                "min": round_float(float(np.min(leaf_values))),
                "max": round_float(float(np.max(leaf_values))),
            }
        else:
            leaf_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "iteration": int(entry["iteration"]),
            "split_count": int(sum(split_counter.values())),
            "leaf_count": int(len(leaf_values)),
            "max_depth": int(max_depth),
            "top_split_features": [
                {
                    "feature_index": int(fi),
                    "split_count": int(cnt),
                    "avg_split_depth": round_float(split_depth_sum[fi] / cnt),
                }
                for fi, cnt in top_features
            ],
            "top_feature_interactions": [
                {
                    "left_feature_index": int(a),
                    "right_feature_index": int(b),
                    "cooccurrence_count": int(cnt),
                }
                for (a, b), cnt in top_pairs
            ],
            "leaf_value_stats": leaf_stats,
        }

    def compute_trajectory_metrics(self, booster) -> Dict[int, Dict[str, Any]]:
        """Compute per-feature trajectory metrics from an XGBoost booster.

        Iterates every tree in *booster* and builds a time-series per feature:

        - **velocity** V_f(t): gain at iteration t
        - **acceleration** A_f(t): ΔV_f(t) = V_f(t) − V_f(t−1)
        - **FCI** (Feature Confusion Index): cover / n_splits at iteration t
        - **cumulative_gain**: running sum of gain

        Args:
            booster: Fitted XGBoost ``Booster`` object.

        Returns:
            Dict mapping feature index → metrics dict with keys
            ``iterations``, ``velocity``, ``acceleration``, ``fci``,
            ``cumulative_gain``, ``gains``, ``covers``, ``split_counts``,
            ``feature_name``.
        """
        df_trees = booster.trees_to_dataframe()
        feature_metrics: Dict[int, Any] = defaultdict(
            lambda: {"iterations": [], "gains": [], "covers": [], "split_counts": []}
        )

        for tree_idx in df_trees["Tree"].unique():
            tree_data = df_trees[df_trees["Tree"] == tree_idx]

            for feature_name in tree_data["Feature"].unique():
                if pd.isna(feature_name):
                    continue

                try:
                    feature_idx = self.feature_names.index(str(feature_name))
                except ValueError:
                    continue

                feature_data = tree_data[tree_data["Feature"] == feature_name]
                feature_metrics[feature_idx]["iterations"].append(tree_idx)
                feature_metrics[feature_idx]["gains"].append(float(feature_data["Gain"].sum()))
                feature_metrics[feature_idx]["covers"].append(float(feature_data["Cover"].sum()))
                feature_metrics[feature_idx]["split_counts"].append(int(len(feature_data)))

        trajectory: Dict[int, Dict[str, Any]] = {}
        for feature_idx, raw in feature_metrics.items():
            gains = np.array(raw["gains"])
            covers = np.array(raw["covers"])
            split_counts = np.array(raw["split_counts"])

            trajectory[feature_idx] = {
                "iterations": raw["iterations"],
                "velocity": gains.tolist(),
                "acceleration": np.diff(gains, prepend=0).tolist(),
                "fci": (covers / split_counts).tolist(),
                "cumulative_gain": np.cumsum(gains).tolist(),
                "gains": gains.tolist(),
                "covers": covers.tolist(),
                "split_counts": split_counts.tolist(),
                "feature_name": (
                    self.feature_names[feature_idx]
                    if feature_idx < len(self.feature_names)
                    else str(feature_idx)
                ),
            }

        # Add raw-Hessian trajectories (suggestion 7.1) when objective supports it.
        self._attach_raw_hessian_trajectories(booster, trajectory)
        return trajectory

    def _attach_raw_hessian_trajectories(
        self,
        booster: Any,
        trajectory: Dict[int, Dict[str, Any]],
    ) -> None:
        """Attach per-feature raw Hessian metrics by boosting iteration.

        For binary logistic objectives, Hessian per sample is:
            h_i = p_i * (1 - p_i)
        where p_i is the predicted probability at iteration t.

        Per iteration and feature, we accumulate Hessians on each split node
        where that feature is used and compute:
            hessian_mean_raw = hessian_sum_raw / hessian_count_raw
        """
        if not trajectory:
            return

        # Determine objective from booster config (booster.attr('objective') is
        # often None for sklearn wrappers).
        objective_name = ""
        try:
            cfg = json.loads(booster.save_config())
            objective_name = str(
                cfg.get("learner", {})
                .get("objective", {})
                .get("name", "")
            )
        except Exception:
            objective_name = ""
        if "logistic" not in objective_name:
            return

        X_train = self.classification.train_test["X_train"]
        if X_train is None or len(X_train) == 0:
            return

        x_values = np.asarray(X_train.to_numpy(), dtype=float)
        if x_values.ndim != 2 or x_values.shape[0] == 0:
            return

        feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        dtrain = xgb.DMatrix(x_values, feature_names=self.feature_names)
        tree_dump = booster.get_dump(dump_format="json")
        if not tree_dump:
            return

        hessian_by_feature: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {"iterations": [], "h_sum": [], "h_count": []}
        )

        for tree_idx, tree_json in enumerate(tree_dump):
            margin = booster.predict(
                dtrain,
                output_margin=True,
                iteration_range=(0, tree_idx + 1),
            )
            probs = 1.0 / (1.0 + np.exp(-np.asarray(margin, dtype=float)))
            hessian = probs * (1.0 - probs)

            tree_obj = json.loads(tree_json)
            feature_h_sum: Dict[int, float] = defaultdict(float)
            feature_h_count: Dict[int, int] = defaultdict(int)
            all_idx = np.arange(x_values.shape[0], dtype=int)
            self._accumulate_tree_raw_hessian(
                node=tree_obj,
                sample_indices=all_idx,
                x_values=x_values,
                hessian=hessian,
                feature_to_idx=feature_to_idx,
                feature_h_sum=feature_h_sum,
                feature_h_count=feature_h_count,
            )

            for feature_idx in feature_h_sum.keys():
                hessian_by_feature[feature_idx]["iterations"].append(int(tree_idx))
                hessian_by_feature[feature_idx]["h_sum"].append(float(feature_h_sum[feature_idx]))
                hessian_by_feature[feature_idx]["h_count"].append(int(feature_h_count[feature_idx]))

        for feature_idx, hvals in hessian_by_feature.items():
            if feature_idx not in trajectory:
                continue
            h_sum = np.asarray(hvals["h_sum"], dtype=float)
            h_count = np.asarray(hvals["h_count"], dtype=float)
            h_mean = np.divide(h_sum, np.maximum(h_count, 1.0))

            trajectory[feature_idx]["hessian_sum_raw"] = h_sum.tolist()
            trajectory[feature_idx]["hessian_count_raw"] = h_count.astype(int).tolist()
            trajectory[feature_idx]["hessian_mean_raw"] = h_mean.tolist()
            # HVI: Hessian Volatility Index (volatility over boosting rounds).
            trajectory[feature_idx]["hessian_volatility_index"] = float(np.std(h_mean))

    def _accumulate_tree_raw_hessian(
        self,
        *,
        node: Any,
        sample_indices: np.ndarray,
        x_values: np.ndarray,
        hessian: np.ndarray,
        feature_to_idx: Dict[str, int],
        feature_h_sum: Dict[int, float],
        feature_h_count: Dict[int, int],
    ) -> None:
        """Accumulate raw Hessian mass/count through split nodes recursively."""
        if sample_indices.size == 0 or not isinstance(node, dict):
            return
        if "leaf" in node:
            return

        split_name = str(node.get("split", ""))
        feature_idx = int(feature_to_idx.get(split_name, -1))
        threshold = float(node.get("split_condition", 0.0))
        children = node.get("children", [])
        if feature_idx < 0 or not isinstance(children, list) or not children:
            return

        # Raw Hessian mass flowing through this split.
        feature_h_sum[feature_idx] += float(np.sum(hessian[sample_indices]))
        feature_h_count[feature_idx] += int(sample_indices.size)

        col = x_values[sample_indices, feature_idx]
        is_nan = np.isnan(col)
        left_mask = (~is_nan) & (col < threshold)
        right_mask = (~is_nan) & (col >= threshold)

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

        left_idx = sample_indices[left_mask]
        right_idx = sample_indices[right_mask]
        missing_idx = sample_indices[is_nan]

        if missing_child is yes_child:
            left_idx = np.concatenate([left_idx, missing_idx])
        elif missing_child is no_child:
            right_idx = np.concatenate([right_idx, missing_idx])

        if yes_child is not None and left_idx.size:
            self._accumulate_tree_raw_hessian(
                node=yes_child,
                sample_indices=left_idx,
                x_values=x_values,
                hessian=hessian,
                feature_to_idx=feature_to_idx,
                feature_h_sum=feature_h_sum,
                feature_h_count=feature_h_count,
            )
        if no_child is not None and right_idx.size:
            self._accumulate_tree_raw_hessian(
                node=no_child,
                sample_indices=right_idx,
                x_values=x_values,
                hessian=hessian,
                feature_to_idx=feature_to_idx,
                feature_h_sum=feature_h_sum,
                feature_h_count=feature_h_count,
            )

    # ------------------------------------------------------------------
    # Delegation — behavioral quadrant analysis
    # ------------------------------------------------------------------

    def get_behavioral_signature(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        feature_idx: int,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Classify a single feature's behavioral signature.

        See :meth:`.BehavioralQuadrantAnalyzer.get_signature` for full docs.
        """
        return self._quadrant_analyzer.get_signature(
            trajectory_metrics,
            feature_idx,
            confusion_metric=confusion_metric,
        )

    def categorize_features_into_quadrants(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Categorize all features into the four behavioral quadrants.

        See :meth:`.BehavioralQuadrantAnalyzer.categorize_features` for full docs.
        """
        return self._quadrant_analyzer.categorize_features(
            trajectory_metrics,
            confusion_metric=confusion_metric,
        )

    # ------------------------------------------------------------------
    # Delegation — visualization
    # ------------------------------------------------------------------

    def plot_behavioral_signatures(
        self,
        trajectory_metrics: Dict,
        figsize: Tuple[int, int] = (18, 12),
        save_path: Optional[str] = None,
        selected_features: Optional[List[int]] = None,
        points_per_feature: Optional[int] = None,
        confusion_metric: str = "hessian",
    ):
        """Plot Feature Acceleration vs. FCI behavioral signatures (2×2 figure).

        See :func:`.behavioral_plotter.plot_behavioral_signatures` for full docs.
        """
        return plot_behavioral_signatures(
            trajectory_metrics,
            figsize=figsize,
            save_path=save_path,
            selected_features=selected_features,
            points_per_feature=points_per_feature,
            confusion_metric=confusion_metric,
        )

    # ------------------------------------------------------------------
    # Delegation — SHAP × behavior correlation
    # ------------------------------------------------------------------

    def correlate_final_shap_with_behavior(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        final_shap_values: Any,
        top_k: int = 15,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Link final-step SHAP variation with FCI / acceleration metrics.

        See :meth:`.ShapBehaviorCorrelator.correlate` for full docs.
        """
        return self._shap_correlator.correlate(
            trajectory_metrics,
            final_shap_values,
            top_k=top_k,
            confusion_metric=confusion_metric,
        )

    # ------------------------------------------------------------------
    # Delegation — decision-path context analysis
    # ------------------------------------------------------------------

    def analyze_feature_decision_contexts(
        self,
        tree_entries: List[Dict[str, Any]],
        trajectory_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Analyze feature co-occurrence contexts along tree decision paths.

        See :meth:`.DecisionContextAnalyzer.analyze` for full docs.
        """
        return self._context_analyzer.analyze(tree_entries, trajectory_metrics, top_k=top_k)

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
        """Build a directed decision-narrowing graph from decision-path context.

        See :meth:`.DecisionContextAnalyzer.build_decision_narrowing_graph`.
        """
        return self._context_analyzer.build_decision_narrowing_graph(
            tree_entries,
            trajectory_metrics=trajectory_metrics,
            decision_context=decision_context,
            top_deciders=top_deciders,
            max_scope_per_decider=max_scope_per_decider,
            min_cooccurrence=min_cooccurrence,
            max_edges=max_edges,
        )

    # ------------------------------------------------------------------
    # Delegation — cohort attribution analysis
    # ------------------------------------------------------------------

    def analyze_subpopulation_cohorts(
        self,
        tree_entries: List[Dict[str, Any]],
        *,
        model: Optional[Any] = None,
        final_shap_values: Optional[Any] = None,
        min_support: int = 50,
        min_paths: int = 5,
        top_k: int = 20,
        top_local_shap: int = 8,
    ) -> Dict[str, Any]:
        """Build cohort attribution payload from repeated scope->decider contexts."""
        X_train = self.classification.train_test["X_train"]
        y_train = self.classification.train_test["Y_train"]
        y_arr = np.asarray(y_train).reshape(len(y_train), -1)[:, 0]

        y_proba = None
        if model is not None:
            estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_train)[:, -1]

        rules = self._cohort_analyzer.extract_path_rules(tree_entries)
        assigned = self._cohort_analyzer.assign_samples_to_rules(X_train, rules)
        cohorts = self._cohort_analyzer.build_cohorts(
            assigned,
            min_support=min_support,
            min_paths=min_paths,
        )
        enriched = self._cohort_analyzer.enrich_cohorts_with_metrics(
            cohorts,
            y_true=y_arr,
            y_proba=y_proba,
            final_shap_values=final_shap_values,
            top_local_shap=top_local_shap,
        )
        return self._cohort_analyzer.to_json_schema(
            enriched,
            top_k=top_k,
            metadata={
                "min_support": int(min_support),
                "min_paths": int(min_paths),
                "top_k": int(top_k),
                "rules_extracted": int(len(rules)),
                "rules_with_support": int(len(assigned)),
                "sample_count": int(len(X_train)),
            },
        )
