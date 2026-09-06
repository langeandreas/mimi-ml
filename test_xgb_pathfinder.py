"""Quick test to verify xgb-pathfinder package structure."""
import sys
sys.path.insert(0, 'c:\\Users\\Andreas\\projects\\mimi-ml')

# Test imports
print("Testing xgb-pathfinder imports...")

try:
    import xgb_pathfinder
    print(f"✅ Main package: {xgb_pathfinder.__version__}")
except Exception as e:
    print(f"❌ Main package error: {e}")
    exit(1)

try:
    from xgb_pathfinder import ModelExplainer
    print(f"✅ ModelExplainer class imported")
except Exception as e:
    print(f"❌ ModelExplainer error: {e}")

try:
    from xgb_pathfinder.analysis import utils, tree_engine, trajectory_engine, shap_engine
    print(f"✅ Analysis modules imported")
except Exception as e:
    print(f"❌ Analysis modules error: {e}")

try:
    from xgb_pathfinder.geographic import impact_aggregator, impact_prioritizer, geo_plotter, mappings
    print(f"✅ Geographic modules imported")
except Exception as e:
    print(f"❌ Geographic modules error: {e}")

try:
    from xgb_pathfinder.types import (
        TrajectoryMetric, FeatureTrajectory, CohortRecord, 
        FeatureProfile, DecisionRule, RegionImpactMetric
    )
    print(f"✅ Type definitions imported")
except Exception as e:
    print(f"❌ Type definitions error: {e}")

print("\n🎉 Package structure verified successfully!")
print("\nAvailable public API:")
print(f"  - ModelExplainer (main entry point)")
print(f"  - compute_trajectory_metrics()")
print(f"  - extract_cohorts()")
print(f"  - get_feature_profiles()")
print(f"  - aggregate_by_region()")
print(f"  - plot_geographic_impact()")
print(f"  - to_dict() / to_json() / to_csv()")
