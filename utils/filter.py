"""
Functions used in streamflow data retrieval process.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def filter_drb_sites(x, sdir = '../Pywr-DRB/DRB_spatial/DRB_shapefiles'):
    """Filters USGS gauge data to remove gauges outside the DRB boundary.

    Args:
        x (pd.DataFrame): A dataframe with gauges including columns "long" and "lat" with location data. 
        sdir (str, optional) The location of the folder containing the DRB shapefile.
    Returns:
        pd.DataFrame: Dataframe containing gauge data, for gauges within the DRB boundary
    """
    crs = 4386

    drb_boarder = gpd.read_file(f'{sdir}/drb_bnd_polygon.shp')
    drb_boarder = drb_boarder.to_crs(crs)
    x_all = gpd.GeoDataFrame(x, geometry = gpd.points_from_xy(x.long, x.lat, crs = crs))
    x_filtered = gpd.clip(x_all, drb_boarder)
    return x_filtered
