"""
Geographic plotter for xgb-pathfinder.

Creates choropleth visualizations of regions colored by cohort impact.
Uses geopandas + matplotlib for static maps or folium for interactive maps.
"""

from typing import Dict, Optional, Union, Any, List
import numpy as np
import pandas as pd

from ..types import RegionImpactMetric


class GeoPlotter:
    """
    Create geographic visualizations of cohort impact by region.
    
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
    
    def load_geodata(self, geodata_path: str) -> Any:
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
        
        if geodata_path.endswith(".json") or geodata_path.endswith(".geojson"):
            gdf = gpd.read_file(geodata_path)
        else:
            gdf = gpd.read_file(geodata_path)
        
        self.geodata = gdf
        return gdf
    
    def plot_choropleth_matplotlib(
        self,
        output_path: Optional[str] = None,
        title: str = "Cohort Impact by Region",
        color_by: str = "total_impact",
        figsize: tuple = (12, 10),
        **kwargs,
    ) -> Any:
        """
        Create static choropleth map using matplotlib/geopandas.
        
        Parameters
        ----------
        output_path : str, optional
            Save to file. If None, returns figure.
        title : str
            Map title.
        color_by : str
            Which metric to color by: "total_impact", "avg_impact_per_cohort", 
            "total_support", "cohort_count".
        figsize : tuple
            Figure size (width, height).
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
        
        # Assume geodata has a 'name' or first string column with region names
        region_col = next(
            (col for col in gdf.columns if col != "geometry" and gdf[col].dtype == object),
            gdf.columns[0],
        )
        
        # Create mapping
        value_dict = dict(zip(region_names, metric_values))
        gdf["impact_value"] = gdf[region_col].map(value_dict)
        gdf["impact_value"] = gdf["impact_value"].fillna(0.0)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(
            column="impact_value",
            ax=ax,
            legend=True,
            cmap="YlOrRd",
            edgecolor="black",
            linewidth=0.5,
            **kwargs,
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(
                vmin=gdf["impact_value"].min(),
                vmax=gdf["impact_value"].max(),
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Impact Score")
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_choropleth_folium(
        self,
        output_path: Optional[str] = None,
        title: str = "Cohort Impact by Region",
        color_by: str = "total_impact",
        zoom_start: int = 5,
    ) -> Any:
        """
        Create interactive choropleth map using folium.
        
        Parameters
        ----------
        output_path : str, optional
            Save to HTML file. If None, returns map object.
        title : str
            Map title.
        color_by : str
            Which metric to color by.
        zoom_start : int
            Initial zoom level.
            
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
        region_col = next(
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
            tiles="OpenStreetMap",
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
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"Impact ({color_by})",
        ).add_to(m)
        
        # Add title
        title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
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
