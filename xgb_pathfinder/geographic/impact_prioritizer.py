"""
Impact prioritizer for xgb-pathfinder.

Ranks regions/cohorts by positive prediction impact rather than raw frequency.
Impact combines: prediction magnitude × support count (frequency).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from ..types import CohortRecord, RegionImpactMetric


class ImpactPrioritizer:
    """
    Rank regions and cohorts by impact (not frequency).
    
    Impact = (prediction magnitude × mean confidence) × support
    
    Parameters
    ----------
    cohorts : List[CohortRecord]
        List of cohorts with metrics.
    """
    
    def __init__(self, cohorts: List[CohortRecord]):
        """Initialize prioritizer."""
        self.cohorts = cohorts
    
    def compute_impact_score(
        self,
        cohort: CohortRecord,
        weight_magnitude: float = 0.5,
        weight_confidence: float = 0.3,
        weight_support: float = 0.2,
    ) -> float:
        """
        Compute composite impact score for a cohort.
        
        Parameters
        ----------
        cohort : CohortRecord
            Cohort to score.
        weight_magnitude : float
            Weight for decision_impact metric (0-1).
        weight_confidence : float
            Weight for mean_confidence metric (0-1).
        weight_support : float
            Weight for support_count metric (0-1).
            
        Returns
        -------
        float
            Normalized impact score (0-1).
        """
        # Normalize metrics to [0, 1]
        norm_magnitude = min(1.0, cohort["decision_impact"] / 0.5)  # Assume max ~0.5
        norm_confidence = cohort["mean_confidence"]  # Already 0-1
        norm_support = min(1.0, cohort["support_count"] / 1000)  # Assume max ~1000
        
        # Weighted combination
        score = (
            weight_magnitude * norm_magnitude +
            weight_confidence * norm_confidence +
            weight_support * norm_support
        )
        
        return float(score)
    
    def rank_cohorts_by_impact(
        self,
        weight_magnitude: float = 0.5,
        weight_confidence: float = 0.3,
        weight_support: float = 0.2,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rank all cohorts by impact score.
        
        Parameters
        ----------
        weight_magnitude : float
            Weight for decision_impact.
        weight_confidence : float
            Weight for mean_confidence.
        weight_support : float
            Weight for support_count.
        top_k : int, optional
            Return only top K cohorts.
            
        Returns
        -------
        List[Tuple[str, float, Dict]]
            List of (cohort_id, impact_score, cohort_metrics) sorted by impact descending.
        """
        ranked = []
        
        for cohort in self.cohorts:
            score = self.compute_impact_score(
                cohort,
                weight_magnitude=weight_magnitude,
                weight_confidence=weight_confidence,
                weight_support=weight_support,
            )
            ranked.append((cohort["cohort_id"], score, cohort))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def rank_regions_by_impact(
        self,
        region_aggregation: Dict[str, RegionImpactMetric],
        weight_magnitude: float = 0.6,
        weight_frequency: float = 0.4,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank regions by impact (prediction magnitude × frequency).
        
        Parameters
        ----------
        region_aggregation : Dict[str, RegionImpactMetric]
            Region metrics from ImpactAggregator.
        weight_magnitude : float
            Weight for average impact per cohort (0-1).
        weight_frequency : float
            Weight for cohort count/support (0-1).
        top_k : int, optional
            Return only top K regions.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (region_name, impact_score) sorted descending.
        """
        ranking = []
        
        # Normalize metrics
        all_impacts = [m["avg_impact_per_cohort"] for m in region_aggregation.values()]
        all_supports = [m["total_support"] for m in region_aggregation.values()]
        
        max_impact = max(all_impacts) if all_impacts else 1.0
        max_support = max(all_supports) if all_supports else 1.0
        
        for region_name, metrics in region_aggregation.items():
            # Normalize to [0, 1]
            norm_impact = metrics["avg_impact_per_cohort"] / max(max_impact, 0.01)
            norm_support = metrics["total_support"] / max(max_support, 1)
            
            # Weighted combination
            score = weight_magnitude * norm_impact + weight_frequency * norm_support
            
            ranking.append((region_name, float(score)))
        
        # Sort by score descending
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranking = ranking[:top_k]
        
        return ranking
    
    def identify_high_impact_regions(
        self,
        region_aggregation: Dict[str, RegionImpactMetric],
        threshold: float = 0.7,  # Top 30%
    ) -> List[str]:
        """
        Identify regions with above-threshold impact.
        
        Parameters
        ----------
        region_aggregation : Dict[str, RegionImpactMetric]
            Region metrics.
        threshold : float
            Percentile threshold (0-1). Default 0.7 = top 30%.
            
        Returns
        -------
        List[str]
            Region names with impact >= threshold.
        """
        ranking = self.rank_regions_by_impact(region_aggregation)
        
        if not ranking:
            return []
        
        # Compute threshold value
        scores = [score for _, score in ranking]
        threshold_value = np.percentile(scores, threshold * 100)
        
        # Return regions above threshold
        return [
            region_name
            for region_name, score in ranking
            if score >= threshold_value
        ]
