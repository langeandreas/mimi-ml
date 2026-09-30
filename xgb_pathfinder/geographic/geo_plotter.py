"""
Geographic plotter for xgb-pathfinder.

Creates choropleth visualizations of regions colored by cohort impact.
Uses geopandas + matplotlib for static maps or folium for interactive maps.
"""

from typing import Dict, Optional, Union, Any, List
from textwrap import fill
import numpy as np
import pandas as pd

from ..types import RegionImpactMetric


_REGION_VALUE_METADATA = {
    "mean_member_impact": (
        "Mean cohort path magnitude",
        "Mean absolute XGBoost leaf value across sample-cohort memberships in the region.",
    ),
    "cumulative_member_impact": (
        "Cumulative cohort path magnitude",
        "Sum of cohort path magnitudes across memberships; sensitive to regional sample size.",
    ),
    "cohort_count": ("Distinct cohort count", "Number of distinct cohorts represented in the region."),
    "sample_count": ("Represented sample count", "Number of mapped model samples represented in the region."),
    "membership_count": (
        "Sample-cohort membership count",
        "Number of overlapping sample-to-cohort assignments represented in the region.",
    ),
    "avg_impact_per_cohort": (
        "Mean magnitude per distinct cohort",
        "Mean path magnitude across the distinct cohorts represented in the region.",
    ),
    "avg_impact_per_sample": (
        "Cumulative path magnitude per sample",
        "Cumulative cohort path magnitude divided by represented regional samples.",
    ),
}


def _value_metadata(metric: Dict[str, Any], color_by: str) -> tuple[str, str]:
    if color_by == "total_impact":
        return (
            metric.get("impact_label", "Regional cohort metric"),
            metric.get("impact_description", ""),
        )
    return _REGION_VALUE_METADATA.get(
        color_by,
        (color_by.replace("_", " ").title(), ""),
    )


class GeoPlotter:
    """
    Visualize regional summaries of cohort decision-path magnitude.
    
    Parameters
    ----------
    region_aggregation : Dict[str, RegionImpactMetric]
        Region metrics from ImpactAggregator.
    geodata : Optional[Any]
        GeoDataFrame with polygons (lazy-loaded if None).
    """
    
    def __init__(
        self,
        region_aggregation: Dict[str, RegionImpactMetric],
        geodata: Optional[Any] = None,
    ):
        """Initialize plotter."""
        self.region_aggregation = region_aggregation
        self.geodata = geodata
    
    def load_geodata(self, geodata_path: str, admin_level: Optional[int] = None) -> Any:
        """
        Load geographic data from file.
        
        Parameters
        ----------
        geodata_path : str
            Path to GeoJSON or Shapefile.
            
        Returns
        -------
        GeoDataFrame
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for geographic visualization. "
                "Install with: pip install geopandas"
            )
        
        if geodata_path.lower().endswith(".csv"):
            from .visualizations import load_admin_geodata

            level = admin_level or 2
            gdf = load_admin_geodata(geodata_path, admin_level=f"adm{level}")
        else:
            gdf = gpd.read_file(geodata_path)
        
        self.geodata = gdf
        return gdf
    
    def plot_choropleth_matplotlib(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        color_by: str = "total_impact",
        figsize: tuple = (12, 10),
        cmap: str = "YlOrRd",
        edgecolor: str = "black",
        linewidth: float = 0.5,
        **kwargs,
    ) -> Any:
        """
        Create static choropleth map using matplotlib/geopandas.
        
        Parameters
        ----------
        output_path : str, optional
            Save to file. If None, returns figure.
        title : str, optional
            Map title. By default, names the selected regional metric.
        color_by : str
            Which metric to color by: "total_impact", "avg_impact_per_cohort", 
            "total_support", "cohort_count".
        figsize : tuple
            Figure size (width, height).
        cmap : str
            Matplotlib colormap name.
        edgecolor : str
            Polygon boundary color.
        linewidth : float
            Polygon boundary width.
        **kwargs
            Additional kwargs to gdf.plot() (edgecolor, linewidth, etc.)
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.geodata is None:
            raise ValueError("Geodata not loaded. Call load_geodata() first.")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")
        
        # Prepare data
        gdf = self.geodata.copy()
        
        # Merge region data
        region_names = list(self.region_aggregation.keys())
        metric_values = [self.region_aggregation[rn][color_by] for rn in region_names]
        metric = next(iter(self.region_aggregation.values()), {})
        metric_label, metric_description = _value_metadata(metric, color_by)
        if title is None:
            title = f"{metric_label} by Region"
        
        # Assume geodata has a 'name' or first string column with region names
        region_col = "Name" if "Name" in gdf.columns else next(
            (col for col in gdf.columns if col != "geometry" and gdf[col].dtype == object),
            gdf.columns[0],
        )
        
        # Create mapping
        value_dict = dict(zip(region_names, metric_values))
        gdf["impact_value"] = gdf[region_col].map(value_dict)
        gdf["impact_value"] = gdf["impact_value"].fillna(0.0)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        legend_kwargs = kwargs.pop("legend_kwds", {})
        legend_kwargs.setdefault("label", metric_label)
        gdf.plot(
            column="impact_value",
            ax=ax,
            legend=True,
            legend_kwds=legend_kwargs,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if metric_description:
            fig.text(0.5, 0.02, fill(metric_description, width=100), ha="center", fontsize=9)
            fig.subplots_adjust(bottom=0.12)
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_choropleth_folium(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        color_by: str = "total_impact",
        zoom_start: int = 5,
        fill_color: str = "YlOrRd",
        tiles: str = "OpenStreetMap",
    ) -> Any:
        """
        Create interactive choropleth map using folium.
        
        Parameters
        ----------
        output_path : str, optional
            Save to HTML file. If None, returns map object.
        title : str, optional
            Map title. By default, names the selected regional metric.
        color_by : str
            Which metric to color by.
        zoom_start : int
            Initial zoom level.
        fill_color : str
            Folium/ColorBrewer sequential color scheme.
        tiles : str
            Folium base-tile provider.
            
        Returns
        -------
        folium.Map
        """
        if self.geodata is None:
            raise ValueError("Geodata not loaded. Call load_geodata() first.")
        
        try:
            import folium
            import json
        except ImportError:
            raise ImportError(
                "folium is required for interactive maps. "
                "Install with: pip install folium"
            )
        
        # Convert geodata to GeoJSON
        geojson_data = json.loads(self.geodata.to_json())
        
        # Extract impact values
        metric = next(iter(self.region_aggregation.values()), {})
        metric_label, metric_description = _value_metadata(metric, color_by)
        if title is None:
            title = f"{metric_label} by Region"
        region_col = "Name" if "Name" in self.geodata.columns else next(
            (col for col in self.geodata.columns if col != "geometry" and self.geodata[col].dtype == object),
            self.geodata.columns[0],
        )
        
        value_dict = {}
        for feat in geojson_data["features"]:
            region_name = feat["properties"].get(region_col)
            if region_name in self.region_aggregation:
                value_dict[region_name] = self.region_aggregation[region_name][color_by]
            else:
                value_dict[region_name] = 0.0
        
        # Compute map center
        centroid = self.geodata.geometry.centroid
        center_lat = centroid.y.mean()
        center_lon = centroid.x.mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tiles,
        )
        
        # Add choropleth
        folium.Choropleth(
            geo_data=geojson_data,
            name=title,
            data=pd.DataFrame(
                list(value_dict.items()),
                columns=[region_col, color_by],
            ),
            columns=[region_col, color_by],
            key_on=f"feature.properties.{region_col}",
            fill_color=fill_color,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=metric_label,
        ).add_to(m)
        
        # Add title
        title_html = (
            f'<div align="center"><h3 style="font-size:16px"><b>{title}</b></h3>'
            f'<p style="font-size:12px">{metric_description}</p></div>'
        )
        m.get_root().html.add_child(folium.Element(title_html))
        
        if output_path:
            m.save(output_path)
        
        return m
    
    def create_impact_summary_table(
        self,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Create summary table of top regions by impact.
        
        Parameters
        ----------
        top_k : int
            Number of top regions to include.
            
        Returns
        -------
        pd.DataFrame
            Summary table with region name, metrics, and rankings.
        """
        # Sort regions by total_impact
        sorted_regions = sorted(
            self.region_aggregation.items(),
            key=lambda x: x[1]["total_impact"],
            reverse=True,
        )
        
        rows = []
        for rank, (region_name, metrics) in enumerate(sorted_regions[:top_k], 1):
            rows.append({
                "Rank": rank,
                "Region": region_name,
                "Total Impact": f"{metrics['total_impact']:.3f}",
                "Avg Impact/Cohort": f"{metrics['avg_impact_per_cohort']:.3f}",
                "Total Support": metrics["total_support"],
                "Cohort Count": metrics["cohort_count"],
                "Impact/Sample": f"{metrics['avg_impact_per_sample']:.4f}",
            })
        
        return pd.DataFrame(rows)
