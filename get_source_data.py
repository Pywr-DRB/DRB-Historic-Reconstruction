"""
This script is used to retreive data for use in the pseudo-observed marginal flow study.

The data is retreived from the following sources:
1. Query gauges from NWIS
2. Get 

"""


import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeohydro as gh
from pygeohydro import NWIS
import pynhd as pynhd
import matplotlib.pyplot as plt

from methods.utils.directories import DATA_DIR, OUTPUT_DIR, FIGURE_DIR
from methods.utils.filter import filter_usgs_nwis_query_by_type, filter_drb_sites
from methods.retrieval.NLDI import add_comid_metadata_to_gage_data
from methods.spatial.upstream import get_upstream_gauges_for_id_list, update_upstream_gauge_file, get_immediate_upstream_sites
from methods.spatial.downstream import get_downstream_gauges_for_id_list, update_downstream_gauge_file, get_immediate_downstream_sites
from methods.spatial.catchments import get_basin_geometry
from methods.utils.contants import GEO_CRS
from methods.utils.nid import get_nid_data, get_dam_storages
from methods.utils.nid import load_drb_mainstem
from methods.utils.nid import get_dam_data_within_catchment


# Filenames for upstream gauge jsons
from methods.spatial.upstream import comid_upstream_gauge_file, station_upstream_gauge_file
from methods.spatial.downstream import comid_downstream_gauge_file, station_downstream_gauge_file


## Direct outputs to a txt file
import sys
sys.stdout = open(f'{OUTPUT_DIR}/get_source_data.txt', 'wt')

####################################################################################################
### Specifications ###
####################################################################################################
dates = ('1900-01-01', '2023-12-31')
bbox = (-77.8, 37.5, -74.0, 44.0)
filter_drb = True
boundary = 'drb' if filter_drb else 'regional'

RERUN = False
MIN_YEARS = 20
MIN_ALLOWABLE_STORAGE = 2000 # Storage in acre-feet which is allowed in a catchment 


if __name__ == "__main__":
    ####################################################################################################
    ### Query all available gauges from NWIS ###
    ####################################################################################################

    nwis = NWIS()

    # Send a query_request for all gage info in the bbox
    query_request = {"bBox": ",".join(f"{b:.06f}" for b in bbox),
                        "hasDataTypeCd": "dv",
                        "outputDataTypeCd": "dv"}

    query_result = nwis.get_info(query_request, 
                                expanded= True)

    # Filter non-streamflow stations
    query_result = filter_usgs_nwis_query_by_type(query_result)

    # list of stations
    stations = list(set(query_result.site_no.tolist()))

    ### Location data (long,lat)
    gage_data = query_result[['site_no', 'dec_long_va', 'dec_lat_va', 'begin_date', 'end_date']]
    gage_data.columns = ['site_no', 'long', 'lat', 'begin_date', 'end_date']
    gage_data.index = gage_data['site_no']
    gage_data= gage_data.drop('site_no', axis=1)

    ## Remove sites outside the DRB boundary
    if filter_drb:
        gage_data = filter_drb_sites(gage_data)
    gage_data = gage_data[~gage_data.index.duplicated(keep = 'first')]

    print(f"Gage data gathered, {len(stations)} USGS streamflow gauges initially found.")
    
    ####################################################################################################
    ### Filter data ###
    ####################################################################################################

    ## Drop sites with less than 10 years of data
    gage_data['years'] = (pd.to_datetime(gage_data['end_date']) - pd.to_datetime(gage_data['begin_date'])).dt.days / 365.25
    gage_data = gage_data[gage_data['years'] > MIN_YEARS]

    ## Update stations list
    stations = list(gage_data.index.unique())
    print(f"Filtered to {len(stations)} USGS streamflow gauges with more than {MIN_YEARS} years of data.")

    ####################################################################################################
    ### Get COMID numbers for each gauge ###
    ####################################################################################################
    print("Gathering COMID numbers for gauges.")
    gage_data_with_comid = add_comid_metadata_to_gage_data(gage_data)

    # Drop duplicate rows
    gage_data_with_comid = gage_data_with_comid.drop_duplicates(subset=['comid'])
    gage_data_with_comid = gage_data_with_comid.dropna(how='any', axis=0)
    print(f"COMID numbers gathered for {gage_data_with_comid.shape[0]} of {gage_data.shape[0]} USGS streamflow gauges.")

    ## Set up dict to translate between comid and site_no
    # comid_to_station is dict with 1-1 mapping of comid to station
    comid_to_station= {}
    for i, row in gage_data_with_comid.iterrows():
        comid_to_station[row['comid']] = i
        
    station_to_comid = {v: k for k, v in comid_to_station.items()}
    
    ####################################################################################################
    ### Get list of gauges upstream of each station ###
    ####################################################################################################

    ## Get upstream gauges for each station
    station_upstream_gauges = get_upstream_gauges_for_id_list(gage_data_with_comid.index.unique(),
                                                            fsource='nwissite',
                                                            file=station_upstream_gauge_file,
                                                            overwrite=False,
                                                            restrict_to_list=True)
    print(f'Got upstream gauges for {len(station_upstream_gauges)} stations.')
    
    # update station upstream gauges file
    update_upstream_gauge_file(station_upstream_gauges,
                                file=station_upstream_gauge_file,
                                overwrite=False)


    ## Get a list of all gauges and their upstream gauges
    all_main_stations = list(station_upstream_gauges.keys())

    # unpack upstream gauge lists
    all_upstream_stations = itertools.chain(*station_upstream_gauges.values())
    all_upstream_stations = list(set(all_upstream_stations))

    # combine into one list
    all_stations = all_main_stations + all_upstream_stations
    all_stations = list(set(all_stations))
    print(f"Got {len(all_stations)} unique stations total.")
    
    ####################################################################################################
    ### Get catchment geometries ###
    #################################################################################################### 

    init = False
    for i, row in gage_data_with_comid.iterrows():
        comid = row['comid']
        station_id = row.name
        basin = get_basin_geometry(station_id, fsource='nwissite')
        
        if basin is None:
            print(f"Failed to get catchment geometry for {station_id}.")
        else:
            basin = basin.to_crs(GEO_CRS)
            if not init:
                station_catchments = gpd.GeoDataFrame(geometry=[basin.geometry[0]], 
                                                    index=[station_id], crs=GEO_CRS)
                init = True
            else:
                station_catchments.loc[station_id] = basin.geometry[0]

    station_catchments = station_catchments
    station_catchments = station_catchments.explode(index_parts=False)
    station_catchments.to_file(f'{DATA_DIR}/NHD/{boundary}_station_catchments.shp')

    print(f"Saved catchment geometries for {len(station_catchments)} stations to {DATA_DIR}/NHD/{boundary}_station_catchments.shp.")

    ### Get marginal catchment geometries for each station
    station_marginal_catchments = station_catchments.copy()
    station_marginal_catchments['geometry'] = None

    for id in station_marginal_catchments.index:
        if id not in station_upstream_gauges.keys():
            continue
        immediate_upstream = get_immediate_upstream_sites(station_upstream_gauges, id)
        if len(immediate_upstream) > 0:
            catchments = station_catchments.loc[immediate_upstream]
            agg_upstream_catchment = catchments.unary_union
            
            # Get difference between total and agg_upstream_catchment geometry
            station_catchment_geom = station_catchments.loc[id, 'geometry']
            station_marginal_catchment_geometry = station_catchment_geom.difference(agg_upstream_catchment)
            
            
            # if multypolygon, take largest
            if station_marginal_catchment_geometry.geom_type == 'MultiPolygon':
                station_marginal_catchment_geometry_fragments = gpd.GeoDataFrame({'geometry':[station_marginal_catchment_geometry]}).explode(index_parts=False)
                station_marginal_catchment_geometry_fragments.reset_index(drop=True, inplace=True)
                argmax_area = station_marginal_catchment_geometry_fragments.area.argmax()
                station_marginal_catchment_geometry = station_marginal_catchment_geometry_fragments.loc[argmax_area, 'geometry']
                
            station_marginal_catchments.loc[id, 'geometry'] = station_marginal_catchment_geometry
            
            # Get the ratio of marginal to total catchment area
            station_marginal_catchments.loc[id, 'marginal_ratio'] = station_marginal_catchments.loc[id, 'geometry'].area / station_catchments.loc[id, 'geometry'].area
            
        else:
            station_marginal_catchments.loc[id, 'geometry'] = station_catchments.loc[id, 'geometry']
            station_marginal_catchments.loc[id, 'marginal_ratio'] = 1.0
            
    # station_marginal_catchments.explode(index_parts=False, inplace=True)
    station_marginal_catchments.to_file(f'{DATA_DIR}/NHD/{boundary}_station_marginal_catchments.shp')

    ####################################################################################################
    ### Get dam data from National Inventory of Dams (NID) ###
    #################################################################################################### 

    ## Get NID metadata for all dams NY, NJ, PA, DE
    drb_dams_df = get_nid_data(MIN_STORAGE=MIN_ALLOWABLE_STORAGE, 
                            filter_drb=False,
                            filter_bbox=bbox)
    dam_storage = get_dam_storages(drb_dams_df)
    drb_dams_df['storage'] = dam_storage
    drb_dams_df.to_file(f'{DATA_DIR}/NID/{boundary}_dam_metadata.shp')

    ## Find dams within each catchment
    catchment_dams_df = get_dam_data_within_catchment(station_catchments, drb_dams_df)
    marginal_catchment_dams_df = get_dam_data_within_catchment(station_marginal_catchments, drb_dams_df)

    # Save dataframe to csv
    catchment_dams_df.to_csv(f'{DATA_DIR}/NID/{boundary}_catchment_dam_summary.csv', index=True)
    marginal_catchment_dams_df.to_csv(f'{DATA_DIR}/NID/{boundary}_marginal_catchment_dam_summary.csv', index=True)
    
    ####################################################################################################    
    ### Classify catchments as managed or unmanaged ###
    ####################################################################################################

    ## Catchments with limited storage in the entire upstream catchment 
    unmanaged_catchments = catchment_dams_df[catchment_dams_df['total_storage'] < MIN_ALLOWABLE_STORAGE].index.tolist()
    unmanaged_catchments_gauge_data = gage_data_with_comid.loc[unmanaged_catchments]
    unmanaged_catchments_gauge_data.to_csv(f'{DATA_DIR}/USGS/{boundary}_unmanaged_usgs_metadata.csv', index=True)

    ## Catchments with limited storage in the marginal catchment
    unmanaged_marginal_catchments = marginal_catchment_dams_df[marginal_catchment_dams_df['total_storage'] < MIN_ALLOWABLE_STORAGE].index.tolist()
    unmanaged_marginal_catchment_gauge_data = gage_data_with_comid.loc[unmanaged_marginal_catchments]
    unmanaged_marginal_catchment_gauge_data.to_csv(f'{DATA_DIR}/USGS/{boundary}_unmanaged_marginal_usgs_metadata.csv', index=True)

    print(f'{unmanaged_catchments_gauge_data.shape[0]} unmanaged catchments and {unmanaged_marginal_catchment_gauge_data.shape[0]} unmanaged marginal catchments found.')
    
    ####################################################################################################
    ### Get streamflow timeseries from NWIS
    ####################################################################################################
    
    ## List of all stations which made it this far
    all_stations = marginal_catchment_dams_df.index.unique()

    ## Use NWIS to get timeseries data
    nwis = NWIS()
    Q_all = nwis.get_streamflow(all_stations, dates)
    Q_all.index = pd.to_datetime(Q_all.index.date)
    Q_all.columns = [usgs_id.split('-')[1] for usgs_id in Q_all.columns]

    # Subset just unmanaged catchments
    Q_unmanaged_total = Q_all.loc[:, unmanaged_catchments]

    # Subset just unmanaged based on marginal catchment
    Q_unmanaged_marginal = Q_all.loc[:, unmanaged_marginal_catchments]

    # Export
    Q_all.to_csv(f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_cms.csv', 
                                sep=',')

    Q_unmanaged_total.to_csv(f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_unmanaged_cms.csv',
                                sep=',')

    Q_unmanaged_marginal.to_csv(f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_marginal_unmanaged_cms.csv',
                                sep=',')
    
    print(f'Got streamflow timeseries at all sites.')
    print('DONE!')