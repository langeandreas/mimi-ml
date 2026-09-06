"""
Geographic visualization and impact aggregation for xgb-pathfinder.

Modules:
--------
- impact_aggregator: Aggregate cohorts by region
- impact_prioritizer: Rank regions by positive prediction impact
- geo_plotter: Choropleth visualization
- mappings: Sample→region mapping and admin hierarchy utilities
"""

from .impact_aggregator import ImpactAggregator
from .impact_prioritizer import ImpactPrioritizer
from .geo_plotter import GeoPlotter
from .mappings import AdminHierarchyMapper, SampleToRegionMapper

__all__ = [
	"ImpactAggregator",
	"ImpactPrioritizer",
	"GeoPlotter",
	"AdminHierarchyMapper",
	"SampleToRegionMapper",
]
