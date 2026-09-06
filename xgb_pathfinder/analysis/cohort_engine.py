"""
Cohort engine for xgb-pathfinder.

Extracts sub-populations (cohorts) from decision tree paths and enriches
with metrics like support, risk rate, decision impact, and SHAP summaries.
"""

from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

from ..types import DecisionRule, CohortRecord
from ..config import DEFAULT_CONFIG
from .tree_engine import TreeEngine
from .utils import ensure_dataframe


class CohortEngine:
    """
    Extract and analyze cohorts (population segments) from decision trees.
    
    A cohort is defined by:
    - scope_signature: AND-ed conditions on early scope features
    - decider_feature: Primary decision feature
    - Enriched with: support, risk_rate, decision_impact, local SHAP summary
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target (for risk rate computation).
    feature_names : List[str]
        Human-readable feature names.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not provided).
    """
    
    def __init__(
        self,
        booster: xgb.Booster,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        config: Optional[Dict] = None,
    ):
        """Initialize cohort engine."""
        self.booster = booster
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.feature_names = feature_names
        self.config = config or DEFAULT_CONFIG
        self.tree_engine = TreeEngine(booster, feature_names)
        self._decision_rules = None  # Cache
        self._sample_rule_mapping = None  # Cache
    
    def extract_decision_rules(self) -> List[DecisionRule]:
        """
        Extract decision rules (root-to-leaf paths) from all trees.
        
        Returns
        -------
        List[DecisionRule]
            All decision rules from all trees in booster.
        """
        if self._decision_rules is not None:
            return self._decision_rules
        
        # Use TreeEngine to extract raw rules
        raw_rules = self.tree_engine.extract_decision_rules_all_trees()
        
        # Convert to DecisionRule TypedDict format
        decision_rules: List[DecisionRule] = []
        for raw_rule in raw_rules:
            rule: DecisionRule = {
                "rule_id": raw_rule["rule_id"],
                "conditions": raw_rule["conditions"],
                "leaf_value": raw_rule["leaf_value"],
                "tree_index": raw_rule["tree_index"],
                "support_count": None,  # Will be filled after assignment
            }
            decision_rules.append(rule)
        
        self._decision_rules = decision_rules
        return decision_rules
    
    def assign_samples_to_rules(self) -> pd.DataFrame:
        """
        Evaluate each sample against all decision rules and assign matches.
        
        Returns
        -------
        pd.DataFrame
            Columns: sample_index, assigned_rule_ids (List[str])
            One row per sample.
            
        Examples
        --------
        >>> assignments = engine.assign_samples_to_rules()
        >>> print(f"Sample 0 matches rules: {assignments.loc[0, 'assigned_rule_ids']}")
        """
        rules = self.extract_decision_rules()
        n_samples = len(self.X_train)
        
        # For each sample, find which rules it satisfies
        sample_assignments = []
        
        for sample_idx in range(n_samples):
            sample = self.X_train[sample_idx]
            matched_rules = []
            
            for rule in rules:
                if self._evaluate_rule(rule, sample):
                    matched_rules.append(rule["rule_id"])
            
            sample_assignments.append({
                "sample_index": sample_idx,
                "assigned_rule_ids": matched_rules,
            })
        
        # Update support counts in rules
        rule_support = defaultdict(int)
        for assignment in sample_assignments:
            for rule_id in assignment["assigned_rule_ids"]:
                rule_support[rule_id] += 1
        
        for rule in rules:
            rule["support_count"] = rule_support.get(rule["rule_id"], 0)
        
        self._sample_rule_mapping = pd.DataFrame(sample_assignments)
        return self._sample_rule_mapping
    
    def _evaluate_rule(self, rule: DecisionRule, sample: np.ndarray) -> bool:
        """
        Check if a sample satisfies all conditions in a rule.
        
        Parameters
        ----------
        rule : DecisionRule
            Decision rule with conditions.
        sample : np.ndarray
            Feature vector of sample.
            
        Returns
        -------
        bool
            True if sample satisfies all conditions.
        """
        for condition in rule["conditions"]:
            feat_idx = condition["feature_index"]
            threshold = condition["threshold"]
            operator = condition.get("operator", "<")
            
            feat_value = sample[feat_idx]
            
            if operator == "<":
                if not (feat_value < threshold):
                    return False
            elif operator == ">=":
                if not (feat_value >= threshold):
                    return False
        
        return True
    
    def build_cohorts(
        self,
        min_support: Optional[int] = None,
        min_paths: Optional[int] = None,
    ) -> List[CohortRecord]:
        """
        Aggregate similar rules into cohorts.
        
        A cohort is formed by grouping rules that share:
        - Similar scope_signature (early conditions)
        - Same decider_feature (primary late decision)
        
        Parameters
        ----------
        min_support : int, optional
            Minimum sample count per cohort. Uses config if not specified.
        min_paths : int, optional
            Minimum distinct paths per cohort. Uses config if not specified.
            
        Returns
        -------
        List[CohortRecord]
            Cohorts sorted by decision_impact (descending).
        """
        if min_support is None:
            min_support = self.config["min_cohort_support"]
        if min_paths is None:
            min_paths = self.config["min_paths_per_cohort"]
        
        rules = self.extract_decision_rules()
        assignments_df = self.assign_samples_to_rules()
        shap_values = None
        try:
            import shap
            shap_values = shap.TreeExplainer(self.booster).shap_values(self.X_train)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values)
        except (ImportError, ValueError, RuntimeError):
            pass
        global_leaf_scale = np.mean(
            [abs(rule["leaf_value"]) for rule in rules]
        ) if rules else 0.0
        
        # Group rules by (scope_signature, decider_feature)
        # For now: use all conditions as scope_signature, last feature as decider
        cohort_groups = defaultdict(list)
        
        for rule in rules:
            if rule["support_count"] is None or rule["support_count"] == 0:
                continue
            
            # Create scope signature from conditions (except last)
            if len(rule["conditions"]) > 0:
                decider_feature = rule["conditions"][-1]["feature_name"]
                scope_conditions = rule["conditions"][:-1]
            else:
                decider_feature = "root"
                scope_conditions = []
            
            # Create human-readable scope signature
            scope_sig = " AND ".join(
                [f"{cond['feature_name']} < {cond['threshold']:.2f}" for cond in scope_conditions]
            ) if scope_conditions else "all_samples"
            
            cohort_key = (scope_sig, decider_feature)
            cohort_groups[cohort_key].append(rule)
        
        # Build CohortRecord for each group
        cohorts = []
        cohort_id_counter = 0
        
        for (scope_sig, decider_feat), group_rules in cohort_groups.items():
            total_support = sum(r["support_count"] for r in group_rules)
            
            # Filter by support and path count
            if total_support < min_support or len(group_rules) < min_paths:
                continue
            
            # Collect samples in this cohort
            cohort_samples = set()
            for sample_idx in range(len(assignments_df)):
                for rule in group_rules:
                    if rule["rule_id"] in assignments_df.loc[sample_idx, "assigned_rule_ids"]:
                        cohort_samples.add(sample_idx)
                        break
            
            cohort_sample_list = list(cohort_samples)
            
            # Compute metrics
            cohort_y = self.y_train[cohort_sample_list]
            risk_rate = float(np.mean(cohort_y))
            
            # Get predictions for confidence
            cohort_pred_proba = self.booster.predict(
                xgb.DMatrix(self.X_train[cohort_sample_list])
            )
            mean_confidence = float(np.mean(cohort_pred_proba))
            
            # Decision impact (placeholder; will be enriched later)
            leaf_values = np.array([r["leaf_value"] for r in group_rules])
            decision_impact = float(np.mean(np.abs(leaf_values)))
            late_decider_rate = float(np.mean([
                bool(rule["conditions"])
                and (len(rule["conditions"]) - 1) >= 0.7 * len(rule["conditions"])
                for rule in group_rules
            ]))
            impact_magnitude_ratio = (
                float(np.mean(np.abs(leaf_values)) / global_leaf_scale)
                if global_leaf_scale > 0 else 0.0
            )
            top_features_by_shap = []
            if shap_values is not None and cohort_sample_list:
                local_importance = np.mean(
                    np.abs(shap_values[cohort_sample_list]), axis=0
                )
                top_indices = np.argsort(local_importance)[::-1][:5]
                top_features_by_shap = [
                    (self.feature_names[index], float(local_importance[index]))
                    for index in top_indices
                ]
            
            cohort_id = f"cohort_{cohort_id_counter}"
            cohort_id_counter += 1
            
            cohort: CohortRecord = {
                "cohort_id": cohort_id,
                "scope_signature": scope_sig,
                "decider_feature_name": decider_feat,
                "decider_feature_index": self._get_feature_index(decider_feat),
                "support_count": total_support,
                "risk_rate": risk_rate,
                "mean_confidence": mean_confidence,
                "late_decider_rate": late_decider_rate,
                "decision_impact": decision_impact,
                "impact_magnitude_ratio": impact_magnitude_ratio,
                "leaf_values": leaf_values.tolist(),
                "top_features_by_shap": top_features_by_shap,
                "contributing_rules": [r["rule_id"] for r in group_rules],
            }
            
            cohorts.append(cohort)
        
        # Sort by decision_impact
        cohorts.sort(key=lambda c: c["decision_impact"], reverse=True)
        
        return cohorts
    
    def _get_feature_index(self, feature_name: str) -> int:
        """Get index of feature by name."""
        if feature_name == "root":
            return -1
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            return -1
