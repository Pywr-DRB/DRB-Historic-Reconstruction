import geopandas as gpd

from methods.spatial.catchments import CatchmentManager

from config import PYWRDRB_NODE_CATCHMENT_FILE, USGS_GAGE_CATCHMENT_FILE


def calculate_catchment_spatial_coverage(
    target_catchment,
    gage_metadata, 
    gage_catchments):
    """
    For a given target catchment, calculate the spatial coverage of gages in the target catchment.
    
    Parameters
    ----------
    target_catchment : Polygon
        The geometry of the target catchment for which to calculate spatial coverage.
    station_metadata : DataFrame
        The metadata of the gages, including the long and lat of the gages.
        
    Returns
    -------
    spatial_coverage : float
        The percentage (0-100) of the target catchment that is covered by gages.
    """
    
    # Calculate the area of the target catchment
    target_catchment_area = target_catchment.area
    
    # Get gage catchments that are in the target catchment
    # using station metadata (with long, lat columns)
    gage_metadata['geometry'] = gpd.points_from_xy(gage_metadata['long'], gage_metadata['lat'])
    gage_metadata = gpd.GeoDataFrame(gage_metadata, geometry='geometry')
    
    # Filter gage catchments to only include those in the target catchment
    gage_metadata = gage_metadata[gage_metadata.within(target_catchment)]
    catchment_gage_catchments = gage_catchments[gage_catchments.index.isin(gage_metadata.index)]
    
    
    # Get area of gage catchments (they may overlap)
    catchment_total_gaged_area = catchment_gage_catchments.unary_union
    if catchment_total_gaged_area is None:
        catchment_total_gaged_area = 0.0
    else:
        catchment_total_gaged_area = catchment_total_gaged_area.area
    
    # Calculate the spatial coverage
    spatial_coverage = catchment_total_gaged_area / target_catchment_area * 100
    
    return spatial_coverage
    
    
    
def plot_pywrdrb_catchment_gage_coverage():
    
    # load
    catchment_manager = CatchmentManager()
    gage_catchments = catchment_manager.load_catchments(USGS_GAGE_CATCHMENT_FILE)
    pywrdrb_catchments = catchment_manager.load_catchments(PYWRDRB_NODE_CATCHMENT_FILE)
    
    # calculate spatial coverage
    pywrdrb_catchments['spatial_coverage'] = 
    