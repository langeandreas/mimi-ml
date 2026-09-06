"""
Impact aggregator for xgb-pathfinder.

Aggregates cohort-level metrics by geographic region, computing:
- Total impact per region
- Average impact per cohort
- Sample counts and distribution
- Regional rankings
"""

from typing import Dict, Union, Optional, List
import numpy as np
import pandas as pd

from ..types import CohortRecord, RegionImpactMetric
from ..config import DEFAULT_CONFIG


class ImpactAggregator:
    """
    Aggregate cohort impact by geographic region.
    
    Parameters
    ----------
    cohorts : List[CohortRecord]
        List of extracted cohorts with metrics.
    config : Dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not specified).
    """
    
    def __init__(
        self,
        cohorts: List[CohortRecord],
        config: Optional[Dict] = None,
    ):
        """Initialize impact aggregator."""
        self.cohorts = cohorts
        self.config = config or DEFAULT_CONFIG
    
    def aggregate_by_region(
        self,
        sample_region_mapping: Union[pd.DataFrame, Dict],
        sample_to_cohort_mapping: Optional[Dict[int, List[str]]] = None,
        impact_metric: str = "magnitude_x_frequency",
    ) -> Dict[str, RegionImpactMetric]:
        """
        Aggregate cohort impact by region.
        
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
            How to compute impact: "magnitude", "frequency", or "magnitude_x_frequency".
            
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
        # Convert mapping to dict if DataFrame
        if isinstance(sample_region_mapping, pd.DataFrame):
            region_dict = sample_region_mapping.iloc[:, 0].to_dict()
        else:
            region_dict = dict(sample_region_mapping)
        
        # Initialize region aggregates
        region_data = {}  # region_name -> {cohorts: [], impacts: [], support: []}
        
        for region_name in set(region_dict.values()):
            region_data[region_name] = {
                "cohorts": [],
                "impact_values": [],
                "support_counts": [],
            }
        
        # Aggregate cohorts by region
        # For simplicity, assign each cohort to regions where it has samples
        cohort_dict = {c["cohort_id"]: c for c in self.cohorts}
        
        if sample_to_cohort_mapping:
            for sample_idx, cohort_ids in sample_to_cohort_mapping.items():
                if sample_idx not in region_dict:
                    continue
                
                region_name = region_dict[sample_idx]
                for cohort_id in cohort_ids:
                    if cohort_id in cohort_dict:
                        cohort = cohort_dict[cohort_id]
                        region_data[region_name]["cohorts"].append(cohort_id)
                        region_data[region_name]["impact_values"].append(cohort["decision_impact"])
                        region_data[region_name]["support_counts"].append(cohort["support_count"])
        else:
            # Fallback: distribute cohorts evenly (approximate)
            for region_name in region_data.keys():
                for cohort in self.cohorts:
                    region_data[region_name]["cohorts"].append(cohort["cohort_id"])
                    region_data[region_name]["impact_values"].append(cohort["decision_impact"])
        
        # Compute region metrics
        results = {}
        
        for region_name, data in region_data.items():
            if not data["cohorts"]:
                continue
            
            cohort_set = set(data["cohorts"])  # Unique cohorts
            impact_values = np.array(data["impact_values"])
            support_counts = np.array(data["support_counts"])
            
            # Compute impact based on metric
            if impact_metric == "magnitude":
                total_impact = float(np.mean(impact_values))
            elif impact_metric == "frequency":
                total_impact = float(len(cohort_set))
            else:  # magnitude_x_frequency
                total_impact = float(np.sum(impact_values) / max(1, len(support_counts)))
            
            avg_impact_per_cohort = float(np.mean(impact_values)) if len(impact_values) > 0 else 0.0
            total_support = int(np.sum(support_counts))
            avg_impact_per_sample = float(total_impact / max(1, total_support))
            
            # Get top cohorts in region
            top_cohorts_in_region = sorted(
                [(cid, cohort_dict[cid]["support_count"], cohort_dict[cid]["decision_impact"])
                 for cid in cohort_set if cid in cohort_dict],
                key=lambda x: x[2],
                reverse=True,
            )[:5]
            
            metric: RegionImpactMetric = {
                "region_name": region_name,
                "region_id": None,  # Optional: could add from mapping
                "cohort_count": len(cohort_set),
                "total_impact": total_impact,
                "avg_impact_per_cohort": avg_impact_per_cohort,
                "avg_impact_per_sample": avg_impact_per_sample,
                "total_support": total_support,
                "top_cohorts": [
                    {
                        "cohort_id": cid,
                        "support": support,
                        "impact": impact,
                    }
                    for cid, support, impact in top_cohorts_in_region
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
        impact_metric: str = "magnitude_x_frequency",
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
