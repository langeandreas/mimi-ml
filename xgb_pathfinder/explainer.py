"""
xgb-pathfinder: Generalized tree-based model explanation package.

Main entry point: ModelExplainer class for XGBoost tree analysis.
Computes trajectory metrics, extracts cohorts, and provides geographic visualization.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb

from .types import (
    PathfinderConfig, ModelExplainerState,
    FeatureTrajectory, FeatureProfile, CohortRecord, DecisionRule,
    RegionImpactMetric
)
from .config import DEFAULT_CONFIG
from .analysis.trajectory_engine import TrajectoryEngine
from .analysis.cohort_engine import CohortEngine
from .analysis.feature_analyzer import FeatureAnalyzer
from .geographic.impact_aggregator import ImpactAggregator
from .geographic.geo_plotter import GeoPlotter


class ModelExplainer:
    """
    Main entry point for XGBoost model explanation using xgb-pathfinder.
    
    Computes trajectory metrics (velocity, acceleration, FCI) across training iterations,
    extracts decision-based cohorts, and provides geographic visualization of impact metrics.
    
    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster model with training history.
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix used for SHAP computation.
    y_train : array-like, shape (n_samples,)
        Training target values.
    feature_names : List[str], optional
        Human-readable feature names. If None, uses ["feature_0", "feature_1", ...].
    config : PathfinderConfig, optional
        Configuration dict for thresholds, top-K values, etc.
        Uses DEFAULT_CONFIG if not provided.
        
    Attributes
    ----------
    booster : xgboost.Booster
        The fitted model.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    feature_names : List[str]
        Feature name mapping.
    config : PathfinderConfig
        Active configuration.
    _state : ModelExplainerState
        Internal computed state (lazy-loaded on first access).
    """
    
    def __init__(
        self,
        booster: xgb.Booster,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        config: Optional[PathfinderConfig] = None,
    ):
        """Initialize explainer with model and data."""
        self.booster = booster
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        else:
            if len(feature_names) != self.X_train.shape[1]:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) does not match "
                    f"number of features ({self.X_train.shape[1]})"
                )
            self.feature_names = list(feature_names)
        
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Lazy-loaded state
        self._state: Optional[ModelExplainerState] = None
        self._trajectory_engine = TrajectoryEngine(self.booster, self.feature_names, self.config)
        self._cohort_engine = CohortEngine(
            self.booster, self.X_train, self.y_train, self.feature_names, self.config
        )
        self._feature_analyzer = FeatureAnalyzer(
            self.booster, self.X_train, self.y_train, self.feature_names, self.config
        )
    
    # ========================================================================
    # Public API: Core Analysis
    # ========================================================================
    
    def compute_trajectory_metrics(self) -> Dict[str, FeatureTrajectory]:
        """
        Compute per-feature trajectory metrics across training iterations.
        
        Returns
        -------
        Dict[str, FeatureTrajectory]
            Maps feature name -> trajectory (velocity, acceleration, FCI, etc. per iteration).
            
        Examples
        --------
        >>> trajectories = explainer.compute_trajectory_metrics()
        >>> for feature_name, traj in trajectories.items():
        ...     print(f"{feature_name}: peak velocity = {traj['peak_velocity']}")
        """
        self._ensure_state_computed(["trajectory_metrics"])
        return self._state["trajectory_metrics"]
    
    def extract_cohorts(
        self,
        min_support: Optional[int] = None,
        include_feature_profiles: bool = False,
    ) -> List[CohortRecord]:
        """
        Extract cohorts (population segments) from decision tree paths.
        
        A cohort is defined by:
        - scope_signature: AND-ed conditions on early scope features
        - decider_feature: Primary decision feature
        - Enriched with metrics: support, risk rate, decision impact, SHAP summary
        
        Parameters
        ----------
        min_support : int, optional
            Minimum sample count per cohort. Uses config value if not specified.
        include_feature_profiles : bool
            If True, compute feature profiles for samples in each cohort (slower).
            
        Returns
        -------
        List[CohortRecord]
            Sorted by decision_impact (highest first).
            
        Examples
        --------
        >>> cohorts = explainer.extract_cohorts(min_support=20)
        >>> for cohort in cohorts[:5]:
        ...     print(f"Cohort {cohort['cohort_id']}: {cohort['support_count']} samples, "
        ...           f"risk={cohort['risk_rate']:.2%}, impact={cohort['decision_impact']:.3f}")
        """
        cohorts = self._cohort_engine.build_cohorts(min_support=min_support)
        self._ensure_state_computed()
        self._state["cohorts"] = cohorts
        self._state["decision_rules"] = self._cohort_engine.extract_decision_rules()
        if include_feature_profiles:
            self._state["feature_profiles"] = self.get_feature_profiles()
        return cohorts
    
    def get_feature_profiles(self) -> Dict[str, FeatureProfile]:
        """
        Get comprehensive per-feature profiles.
        
        Combines trajectory metrics, SHAP decision impact, feature interactions,
        and path statistics into a single profile per feature.
        
        Returns
        -------
        Dict[str, FeatureProfile]
            Maps feature name -> profile.
            
        Examples
        --------
        >>> profiles = explainer.get_feature_profiles()
        >>> for feature_name, profile in profiles.items():
        ...     print(f"{feature_name}: behavior={profile['behavior_signature']}, "
        ...           f"peak_impact={profile['decision_impact']['total_swing']:.3f}")
        """
        self._ensure_state_computed(["feature_profiles"])
        return self._state["feature_profiles"]
    
    def get_segments(
        self,
        rules: Optional[List[DecisionRule]] = None,
    ) -> pd.DataFrame:
        """
        Assign samples to cohorts/segments based on decision rules.
        
        Parameters
        ----------
        rules : List[DecisionRule], optional
            Decision rules to evaluate. If None, uses rules from extract_cohorts().
            
        Returns
        -------
        pd.DataFrame
            One row per sample, columns: sample_index, assigned_cohort_ids (list), 
            primary_cohort_id, confidence.
            
        Examples
        --------
        >>> segments = explainer.get_segments()
        >>> cohort_counts = segments['primary_cohort_id'].value_counts()
        """
        assignments = self._cohort_engine.assign_samples_to_rules().copy()
        self._ensure_state_computed()
        rule_ids = {rule["rule_id"] for rule in (rules or self._state["decision_rules"])}
        assignments["assigned_rule_ids"] = assignments["assigned_rule_ids"].apply(
            lambda ids: [rule_id for rule_id in ids if rule_id in rule_ids]
        )
        assignments["primary_rule_id"] = assignments["assigned_rule_ids"].apply(
            lambda ids: ids[0] if ids else None
        )
        assignments["confidence"] = self.booster.predict(self.X_train)
        return assignments
    
    # ========================================================================
    # Public API: Geographic & Impact
    # ========================================================================
    
    def aggregate_by_region(
        self,
        geodata_path: str,
        sample_region_mapping: Union[pd.DataFrame, Dict[int, str]],
        admin_level: Optional[int] = None,
        impact_metric: str = "magnitude_x_frequency",
    ) -> Dict[str, RegionImpactMetric]:
        """
        Aggregate cohort impact by geographic region.
        
        Ranks regions by positive prediction impact (not just frequency).
        
        Parameters
        ----------
        geodata_path : str
            Path to GeoJSON or Shapefile with polygon geometries.
        sample_region_mapping : pd.DataFrame or Dict
            Maps sample index or ID to region name.
            If DataFrame: index must be sample index, values must be region names.
            If Dict: keys are sample index, values are region names.
        admin_level : int, optional
            Administrative hierarchy level (1, 2, etc.). Uses config if not specified.
        impact_metric : str
            How to compute impact: "magnitude" (mean), "frequency" (count),
            or "magnitude_x_frequency" (product).
            
        Returns
        -------
        Dict[str, RegionImpactMetric]
            Maps region name -> impact metrics (sorted by total_impact descending).
            
        Examples
        --------
        >>> mapping = pd.read_csv("sample_to_region.csv", index_col=0)
        >>> regions = explainer.aggregate_by_region(
        ...     geodata_path="admin_boundaries.geojson",
        ...     sample_region_mapping=mapping,
        ... )
        >>> for region_name, metrics in regions.items():
        ...     print(f"{region_name}: impact={metrics['total_impact']:.2f}, "
        ...           f"samples={metrics['total_support']}")
        """
        cohorts = self.extract_cohorts()
        segments = self.get_segments()
        cohort_by_rule = {
            rule_id: cohort["cohort_id"]
            for cohort in cohorts
            for rule_id in cohort["contributing_rules"]
        }
        sample_to_cohorts = {
            int(row.sample_index): [
                cohort_by_rule[rule_id]
                for rule_id in row.assigned_rule_ids
                if rule_id in cohort_by_rule
            ]
            for row in segments.itertuples(index=False)
        }
        aggregation = ImpactAggregator(cohorts, self.config).aggregate_by_region(
            sample_region_mapping,
            sample_to_cohort_mapping=sample_to_cohorts,
            impact_metric=impact_metric,
        )
        self._ensure_state_computed()
        self._state["geographic_aggregation"] = aggregation
        return aggregation
    
    def plot_geographic_impact(
        self,
        geodata_path: str,
        sample_region_mapping: Union[pd.DataFrame, Dict[int, str]],
        output_path: Optional[str] = None,
        impact_metric: str = "magnitude_x_frequency",
        **plot_kwargs,
    ) -> Any:
        """
        Create choropleth map of regions colored by cohort impact.
        
        Parameters
        ----------
        geodata_path : str
            Path to GeoJSON or Shapefile.
        sample_region_mapping : pd.DataFrame or Dict
            Sample to region mapping.
        output_path : str, optional
            Save map to file. If None, returns matplotlib/folium object.
        impact_metric : str
            Impact computation method (see aggregate_by_region).
        **plot_kwargs
            Additional kwargs to GeoPlotter (color_scale, figsize, etc.)
            
        Returns
        -------
        Any
            Matplotlib Figure or Folium Map depending on visualization backend.
            
        Examples
        --------
        >>> fig = explainer.plot_geographic_impact(
        ...     geodata_path="admin_boundaries.geojson",
        ...     sample_region_mapping=mapping,
        ...     output_path="impact_map.html",
        ... )
        """
        aggregation = self.aggregate_by_region(
            geodata_path, sample_region_mapping, impact_metric=impact_metric
        )
        plotter = GeoPlotter(aggregation)
        plotter.load_geodata(geodata_path)
        backend = plot_kwargs.pop("backend", "matplotlib")
        color_by = plot_kwargs.pop("color_by", "total_impact")
        if backend == "folium":
            return plotter.plot_choropleth_folium(
                output_path=output_path, color_by=color_by, **plot_kwargs
            )
        return plotter.plot_choropleth_matplotlib(
            output_path=output_path, color_by=color_by, **plot_kwargs
        )
    
    # ========================================================================
    # Public API: Export & Serialization
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export all computed metrics as a nested dictionary.
        
        Suitable for serialization to JSON or further processing.
        
        Returns
        -------
        Dict[str, Any]
            Keys: "trajectory_metrics", "feature_profiles", "cohorts", "decision_rules".
            
        Examples
        --------
        >>> data = explainer.to_dict()
        >>> import json
        >>> with open("results.json", "w") as f:
        ...     json.dump(data, f, indent=2)
        """
        self._ensure_state_computed()
        payload = {
            "config": self.config,
            "feature_names": self.feature_names,
            "trajectory_metrics": self._state["trajectory_metrics"],
            "feature_profiles": self._state["feature_profiles"],
            "decision_rules": self._state["decision_rules"],
            "cohorts": self._state["cohorts"],
        }
        if self._state.get("geographic_aggregation") is not None:
            payload["geographic_aggregation"] = self._state["geographic_aggregation"]
        return self._json_safe(payload)
    
    def to_json(self, output_path: str, indent: Optional[int] = None) -> None:
        """
        Export all metrics to JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to output JSON file.
        indent : int, optional
            JSON indentation. Uses config if not specified.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output_file:
            json.dump(
                self.to_dict(),
                output_file,
                indent=self.config["json_indent"] if indent is None else indent,
            )
    
    def to_csv(self, output_dir: str) -> None:
        """
        Export metrics to separate CSV files in a directory.
        
        Creates: trajectory_metrics.csv, cohorts.csv, feature_profiles.csv, etc.
        
        Parameters
        ----------
        output_dir : str
            Directory to write CSV files.
        """
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        trajectory_rows = []
        for feature_name, trajectory in payload["trajectory_metrics"].items():
            for metric in trajectory["metrics"]:
                trajectory_rows.append({"feature_name": feature_name, **metric})
        pd.DataFrame(trajectory_rows).to_csv(target / "trajectory_metrics.csv", index=False)
        pd.DataFrame(payload["cohorts"]).to_csv(target / "cohorts.csv", index=False)
        pd.DataFrame(payload["decision_rules"]).to_csv(target / "decision_rules.csv", index=False)
        profile_rows = []
        for feature_name, profile in payload["feature_profiles"].items():
            row = {
                "feature_name": feature_name,
                "behavior_signature": profile["behavior_signature"],
            }
            row.update(profile["decision_impact"])
            row.update(profile["path_statistics"])
            profile_rows.append(row)
        pd.DataFrame(profile_rows).to_csv(target / "feature_profiles.csv", index=False)
    
    # ========================================================================
    # Private: State Management
    # ========================================================================
    
    def _ensure_state_computed(self, components: Optional[List[str]] = None) -> None:
        """
        Ensure state components are computed (lazy loading).
        
        Parameters
        ----------
        components : List[str], optional
            Which state components to compute. If None, computes all on-demand.
            Options: "trajectory_metrics", "feature_profiles", "cohorts", etc.
        """
        if self._state is None:
            self._state = {}
        requested = set(components or [])
        if not requested:
            requested = {"trajectory_metrics", "feature_profiles", "cohorts", "decision_rules"}
        if "trajectory_metrics" in requested and "trajectory_metrics" not in self._state:
            self._state["trajectory_metrics"] = self._trajectory_engine.compute_trajectories()
        if "feature_profiles" in requested and "feature_profiles" not in self._state:
            self._state["feature_profiles"] = self._feature_analyzer.build_profiles()
        if "cohorts" in requested and "cohorts" not in self._state:
            self._state["cohorts"] = self._cohort_engine.build_cohorts()
        if "decision_rules" in requested and "decision_rules" not in self._state:
            self._state["decision_rules"] = self._cohort_engine.extract_decision_rules()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert numpy values and containers into JSON-compatible values."""
        if isinstance(value, dict):
            return {str(key): ModelExplainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ModelExplainer._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return ModelExplainer._json_safe(value.tolist())
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    
    def _clear_cache(self) -> None:
        """Clear cached state to free memory."""
        self._state = None
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_config(self) -> PathfinderConfig:
        """Return current configuration."""
        return self.config.copy()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters. Use with caution (clears cache)."""
        self.config.update(kwargs)
        self._clear_cache()
    
    def __repr__(self) -> str:
        """String representation."""
        n_samples, n_features = self.X_train.shape
        return (
            f"ModelExplainer(n_samples={n_samples}, n_features={n_features}, "
            f"n_trees={len(self.booster.get_dump())})"
        )
