"""
Geographic visualization and impact aggregation for xgb-pathfinder.

Modules:
--------
- impact_aggregator: Aggregate cohorts by region
- impact_prioritizer: Rank regions with a configurable heuristic path score
- geo_plotter: Choropleth visualization
- mappings: Sample→region mapping and admin hierarchy utilities
"""

from .impact_aggregator import ImpactAggregator
from .impact_prioritizer import ImpactPrioritizer
from .geo_plotter import GeoPlotter
from .mappings import AdminHierarchyMapper, SampleToRegionMapper
from .visualizations import (
	load_admin_geodata,
	load_household_admin_mapping,
	build_cohort_membership,
	aggregate_cohorts_by_region,
	visualise_cohort_map,
	visualise_region_decision_paths,
	plot_cohort_geography,
)

__all__ = [
	"ImpactAggregator",
	"ImpactPrioritizer",
	"GeoPlotter",
	"AdminHierarchyMapper",
	"SampleToRegionMapper",
	"load_admin_geodata",
	"load_household_admin_mapping",
	"build_cohort_membership",
	"aggregate_cohorts_by_region",
	"visualise_cohort_map",
	"visualise_region_decision_paths",
	"plot_cohort_geography",
]
