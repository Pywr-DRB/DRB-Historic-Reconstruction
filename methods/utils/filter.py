"""
Functions used to filter datasets.
"""
import pandas as pd
import geopandas as gpd
from methods.utils.directories import PYWRDRB_DIR
from methods.utils.constants import GEO_CRS
from methods.retrieval.NID import load_drb_boundary



def filter_drb_sites(x,
                     crs = GEO_CRS):
    """Filters USGS gauge data to remove gauges outside the DRB boundary.

    Args:
        x (pd.DataFrame): A dataframe with gauges including columns "long" and "lat" with location data. 
        sdir (str, optional) The location of the folder containing the DRB shapefile.
    Returns:
        pd.DataFrame: Dataframe containing gauge data, for gauges within the DRB boundary
    """

    drb_boarder = load_drb_boundary(crs)
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


def filter_gage_data_by_record_length(gage_data, MIN_YEARS):
    
    # Check that start_date and end_date columns exist
    if 'start_date' not in gage_data.columns or 'end_date' not in gage_data.columns:
        raise ValueError("gage_data must have columns 'start_date' and 'end_date'")
    
    ### Filter data 
    ## Drop sites with less than 10 years of data
    gage_data['years'] = (pd.to_datetime(gage_data['end_date']) - pd.to_datetime(gage_data['begin_date'])).dt.days / 365.25
    gage_data = gage_data[gage_data['years'] > MIN_YEARS]
    return gage_data