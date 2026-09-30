"""
Impact aggregator for xgb-pathfinder.

Aggregates the magnitude and representation of cohort decision paths by region.

These descriptive metrics do not measure causal dependence on cohorts. Cohorts
are post-hoc groups of tree paths, not features supplied to the model.
"""

from collections import Counter
from typing import Dict, Union, Optional, List
import numpy as np
import pandas as pd

from ..types import RegionImpactMetric
from ..config import DEFAULT_CONFIG


REGIONAL_IMPACT_METRICS = {
    "mean_member_impact": {
        "label": "Mean cohort path magnitude",
        "description": (
            "Mean absolute XGBoost leaf value across sample-cohort memberships in the region. "
            "Higher values indicate stronger-magnitude decision paths, not causal dependence."
        ),
    },
    "cumulative_member_impact": {
        "label": "Cumulative cohort path magnitude",
        "description": (
            "Sum of absolute cohort path magnitudes across sample-cohort memberships. "
            "This increases with both path magnitude and the number of memberships."
        ),
    },
    "distinct_cohort_count": {
        "label": "Distinct cohort count",
        "description": "Number of distinct extracted cohorts represented in the region.",
    },
}

_METRIC_ALIASES = {
    "magnitude": "mean_member_impact",
    "magnitude_x_frequency": "cumulative_member_impact",
    "frequency": "distinct_cohort_count",
}


class ImpactAggregator:
    """
    Aggregate cohort impact by geographic region.
    
    Parameters
    ----------
    cohorts : pd.DataFrame
        Extracted cohorts with one row per cohort.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not specified).
    """
    
    def __init__(
        self,
        cohorts: pd.DataFrame,
        config: Optional[Dict] = None,
    ):
        """Initialize impact aggregator."""
        self.cohorts = cohorts
        self.config = config or DEFAULT_CONFIG
    
    def aggregate_by_region(
        self,
        sample_region_mapping: Union[pd.DataFrame, Dict],
        sample_to_cohort_mapping: Optional[Dict[int, List[str]]] = None,
        impact_metric: str = "mean_member_impact",
    ) -> Dict[str, RegionImpactMetric]:
        """
        Aggregate cohort path magnitude and representation by region.
        
        Parameters
        ----------
        sample_region_mapping : pd.DataFrame or Dict
            Maps sample index to region name.
            If DataFrame: index=sample index, values=region names.
            If Dict: keys=sample index, values=region names.
        sample_to_cohort_mapping : Dict[int, List[str]], optional
            Maps sample index to list of cohort IDs.
            If None, must be provided elsewhere or computed.
        impact_metric : str
            ``mean_member_impact`` averages cohort path magnitude over memberships;
            ``cumulative_member_impact`` sums it; ``distinct_cohort_count`` counts cohorts.
            Legacy names ``magnitude``, ``magnitude_x_frequency``, and ``frequency``
            remain supported as aliases.
            
        Returns
        -------
        Dict[str, RegionImpactMetric]
            Maps region name -> impact metrics, sorted by total_impact descending.
            
        Examples
        --------
        >>> mapping = pd.read_csv("sample_to_region.csv", index_col=0)
        >>> regions = aggregator.aggregate_by_region(mapping)
        >>> for region, metrics in regions.items():
        ...     print(f"{region}: impact={metrics['total_impact']:.2f}")
        """
        impact_metric = _METRIC_ALIASES.get(impact_metric, impact_metric)
        if impact_metric not in REGIONAL_IMPACT_METRICS:
            raise ValueError(
                f"Unknown impact_metric {impact_metric!r}; expected one of "
                f"{sorted(REGIONAL_IMPACT_METRICS)}"
            )
        metric_definition = REGIONAL_IMPACT_METRICS[impact_metric]

        # Convert mapping to dict if DataFrame
        if isinstance(sample_region_mapping, pd.DataFrame):
            admin_level = int(self.config.get("admin_level", 2))
            region_column = next(
                (
                    column
                    for column in (f"adm{admin_level}Name", "district", "region", "region_name")
                    if column in sample_region_mapping.columns
                ),
                None,
            )
            id_column = next(
                (
                    column
                    for column in ("household_id", "sample_id", "hhid")
                    if column in sample_region_mapping.columns
                ),
                None,
            )
            if region_column is None:
                raise ValueError(
                    "sample_region_mapping must contain a region column such as "
                    f"adm{admin_level}Name, district, region, or region_name"
                )
            if id_column is None:
                region_dict = {
                    str(sample_id): region
                    for sample_id, region in sample_region_mapping[region_column].items()
                }
            else:
                region_dict = {
                    str(sample_id): region
                    for sample_id, region in sample_region_mapping.set_index(id_column)[region_column].items()
                }
        else:
            region_dict = {
                str(sample_id): region
                for sample_id, region in dict(sample_region_mapping).items()
            }
        
        # Initialize region aggregates
        region_data = {}  # region_name -> {cohorts: [], impacts: [], support: []}
        
        for region_name in set(region_dict.values()):
            region_data[region_name] = {
                "cohorts": [],
                "impact_values": [],
                "support_counts": [],
                "sample_ids": set(),
            }
        
        # Aggregate cohorts by region
        # For simplicity, assign each cohort to regions where it has samples
        cohort_dict = self.cohorts.set_index("cohort_id").to_dict(orient="index")
        
        if sample_to_cohort_mapping:
            for sample_idx, cohort_ids in sample_to_cohort_mapping.items():
                mapping_key = str(sample_idx)
                if mapping_key not in region_dict:
                    continue
                
                region_name = region_dict[mapping_key]
                region_data[region_name]["sample_ids"].add(mapping_key)
                for cohort_id in cohort_ids:
                    if cohort_id in cohort_dict:
                        cohort = cohort_dict[cohort_id]
                        region_data[region_name]["cohorts"].append(cohort_id)
                        region_data[region_name]["impact_values"].append(cohort["decision_impact"])
                        region_data[region_name]["support_counts"].append(cohort["support_count"])
        else:
            # Fallback: distribute cohorts evenly (approximate)
            for region_name in region_data.keys():
                for cohort_id, cohort in cohort_dict.items():
                    region_data[region_name]["cohorts"].append(cohort_id)
                    region_data[region_name]["impact_values"].append(cohort["decision_impact"])
        
        # Compute region metrics
        results = {}
        
        for region_name, data in region_data.items():
            if not data["cohorts"]:
                continue
            
            cohort_set = set(data["cohorts"])  # Unique cohorts
            impact_values = np.array(data["impact_values"])
            membership_count = len(impact_values)
            sample_count = len(data["sample_ids"])
            mean_member_impact = float(np.mean(impact_values))
            cumulative_member_impact = float(np.sum(impact_values))

            if impact_metric == "mean_member_impact":
                total_impact = mean_member_impact
            elif impact_metric == "distinct_cohort_count":
                total_impact = float(len(cohort_set))
            else:
                total_impact = cumulative_member_impact

            unique_impacts = [cohort_dict[cohort_id]["decision_impact"] for cohort_id in cohort_set]
            avg_impact_per_cohort = float(np.mean(unique_impacts))
            avg_impact_per_sample = cumulative_member_impact / max(1, sample_count)
            regional_counts = Counter(data["cohorts"])
            
            # Get top cohorts in region
            top_cohorts_in_region = sorted(
                [(cid, regional_counts[cid], cohort_dict[cid]["decision_impact"])
                 for cid in cohort_set if cid in cohort_dict],
                key=lambda x: x[1] * x[2],
                reverse=True,
            )[:5]
            
            metric: RegionImpactMetric = {
                "region_name": region_name,
                "region_id": None,  # Optional: could add from mapping
                "impact_metric": impact_metric,
                "impact_label": metric_definition["label"],
                "impact_description": metric_definition["description"],
                "cohort_count": len(cohort_set),
                "total_impact": total_impact,
                "mean_member_impact": mean_member_impact,
                "cumulative_member_impact": cumulative_member_impact,
                "membership_count": membership_count,
                "sample_count": sample_count,
                "avg_impact_per_cohort": avg_impact_per_cohort,
                "avg_impact_per_sample": avg_impact_per_sample,
                "total_support": sample_count,
                "top_cohorts": [
                    {
                        "cohort_id": cid,
                        "regional_membership_count": regional_count,
                        "cohort_path_magnitude": impact,
                    }
                    for cid, regional_count, impact in top_cohorts_in_region
                ],
            }
            
            results[region_name] = metric
        
        # Sort by total_impact descending
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["total_impact"],
            reverse=True,
        )
        
        return dict(sorted_results)
    
    def get_impact_ranking(
        self,
        sample_region_mapping: Union[pd.DataFrame, Dict],
        impact_metric: str = "mean_member_impact",
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Get ranked list of regions by impact.
        
        Parameters
        ----------
        sample_region_mapping : pd.DataFrame or Dict
            Sample to region mapping.
        impact_metric : str
            Impact computation method.
        top_k : int, optional
            Return only top K regions.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (region_name, total_impact) sorted descending.
        """
        aggregation = self.aggregate_by_region(
            sample_region_mapping,
            impact_metric=impact_metric,
        )
        
        ranking = [
            (region_name, metrics["total_impact"])
            for region_name, metrics in aggregation.items()
        ]
        
        if top_k is not None:
            ranking = ranking[:top_k]
        
        return ranking
