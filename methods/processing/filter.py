"""
Functions used in streamflow data retrieval process.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

import pynhd
from pynhd import NLDI


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


def filter_usgs_nwis_query_by_type(query_result, 
                              site_tp_cd=['ST'],
                              parm_cd='00060'):
    """
    Filters USGS gauge metadata based on site_tp_cd and parm_cd.
    
    Info about site_tp_cd: https://maps.waterdata.usgs.gov/mapper/help/sitetype.html
    Info about parm_cd: https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
    
    Args:
        query_result (pd.DataFrame): The result of a query to the USGS NWIS database from pygeohydro.
        site_tp_cd (str, optional): The site type classification. Defaults to 'ST'.

    Returns:
        pd.DataFrame: The filtered dataframe containing gauge metadata.
    """
    
    # Filter non-streamflow stations
    query_result = query_result.query(f"site_tp_cd in ({site_tp_cd})") 
    query_result = query_result[query_result.parm_cd == parm_cd] 
    query_result = query_result.reset_index(drop = True)
    return query_result



def get_sites_by_basin(basin):
    """
    Returns gauge IDs which are located in either the upper, mid, or lower basin. 
    Upper basin is upstream of Montague.
    Mid basin is between Montague and Trenton.
    Lower basin is below Trenton.
    
    Args:
        basin (str): The basin of interest. Must be one of 'upper', 'mid', or 'lower'.
    
    Returns:
        list: A list containing gauge numbers for the specified basin.    
    """
    
    basin_options=['upper', 'mid', 'lower']
    assert(basin in basin_options), f'basin must be one of {basin_options}'
    
    # Select the lower gauge point corresponding to basin
    upper_gauge_number= '01438500'
    mid_gauge_number= '01463500'
    lower_gauge_number= '01482100' # Furthest downstream gauge in mainstem
    
    # Search for upstream stations for each basin
    nldi= NLDI() 
    
    # Upper basin
    upper_basin_data= nldi.navigate_byid(fsource="nwissite",
                                              fid=f"USGS-{upper_gauge_number}",
                                              navigation="upstreamTributaries",
                                              source="nwissite",
                                              distance=1000)
    upper_stations = upper_basin_data.identifier.str.split("-").str[1].unique()
    
    # Mid basin
    mid_basin_data= nldi.navigate_byid(fsource="nwissite",
                                                fid=f"USGS-{mid_gauge_number}",
                                                navigation="upstreamTributaries",
                                                source="nwissite",
                                                distance=1000)
    upper_mid_stations = mid_basin_data.identifier.str.split("-").str[1].unique()
    
    # Remove upper basin stations from mid basin stations
    mid_stations = np.setdiff1d(upper_mid_stations, upper_stations)
    
    # Lower basin
    lower_basin_data= nldi.navigate_byid(fsource="nwissite",
                                                fid=f"USGS-{lower_gauge_number}",
                                                navigation="upstreamTributaries",
                                                source="nwissite",
                                                distance=1000)
    all_stations = lower_basin_data.identifier.str.split("-").str[1].unique()
    
    # Remove mid basin stations from lower basin stations
    lower_stations = np.setdiff1d(all_stations, upper_mid_stations)
    
    # Return the appropriate basin
    if basin == 'upper':
        return upper_stations
    elif basin == 'mid':
        return mid_stations
    elif basin == 'lower':
        return lower_stations
    else:
        return None
    
