"""
Contains scripts for helping to handle catchment and subcatchment processing. 
"""

import numpy as np
import pandas as pd
import pynhd as nhd
import geopandas as gpd

from methods.utils.constants import GEO_CRS
from methods.utils.directories import DATA_DIR



def get_basin_catchment_area(feature_id, feature_source='comid'):
    """Returns the catchment area of a comid.
    
    Args:
        feature_id (str): A comid or USGS number as a string.
        feature_source (str): 'comid' or 'usgs'. Defaults to 'comid'.
    Returns:
        float: The catchment area of the basin
    """
    cartesian_crs = 3857
    nldi = nhd.NLDI()
    basin_data = nldi.get_basins(fsource=feature_source, feature_ids=feature_id)
    area = basin_data.to_crs(cartesian_crs).geometry.area.values/10**6
    return area


def load_station_catchments(crs = GEO_CRS,
                            boundary='drb',
                            marginal = False):
    # catchment geometry
    geom_file = f'{boundary}_station_catchments.shp' if not marginal else f'{boundary}_station_marginal_catchments.shp'
    geom_file = f'{DATA_DIR}/NHD/' + geom_file
    
    station_catchments = gpd.read_file(geom_file,
                                       index_col=0, crs=GEO_CRS)
    station_catchments.index = station_catchments['index']
    station_catchments.drop('index', axis=1, inplace=True)

    # Find duplicate sites and keep biggest polygon
    station_catchments['area'] = station_catchments.area
    station_catchments = station_catchments.sort_values(by='area', ascending=False)
    station_catchments = station_catchments[~station_catchments.index.duplicated(keep='first')]
    return station_catchments











def calculate_marginal_catchment_flow(Q_df, 
                                      station_id,
                                      upstream_gauges):
    """
    Calculates the marginal flow at a station by removing the flow from upstream gauges.
    
    Args: 
        Q_df (pd.DataFrame): Dataframe of streamflows with columns as gauge station IDs
        station_id (str): Gauge station ID of interest
        upstream_gauges (list): List of gauge station IDs upstream of station_id
    Returns:
        marginal_flow (pd.Series): Series of marginal flow for station_id
    """
    
    # Check if station_id is in upstream_gauges
    if station_id in upstream_gauges:
        upstream_gauges.remove(station_id)
    
    # Calculate marginal flow
    marginal_flow = Q_df[station_id]
    for gauge in upstream_gauges:
        marginal_flow -= Q_df[gauge]
        
    # check for nans
    if marginal_flow.isna().sum() > 0:
        print(f'Warning: {station_id} has nans')
    return marginal_flow


def get_lag_days(Qa, Qb, 
                 method='cross-correlation', 
                 l_min=0, l_max=4):
    """
    Methods for approximating integer day lag between two time series.
    """
    
    for method in ['cross-correlation', 'min-diff']:
        if method == 'cross-correlation':
            # Cross-correlation
            xcorr = np.correlate(Qa, Qb, mode='full')
            lags = np.arange(-len(Qa) + 1, len(Qa))
            lag = lags[np.argmax(xcorr)]
            return lag
        elif method == 'min-diff':
            # Min-difference
            diffs = np.zeros(l_max - l_min)
            for l in range(l_min, l_max):
                diffs[l] = np.sum(np.abs(Qa - np.roll(Qb, l)))
            lag = np.argmin(diffs)
            return lag
        else:
            raise ValueError(f'Invalid method: {method}')


