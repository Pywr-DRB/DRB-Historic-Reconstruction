"""
This script is used to retreive data for use in the pseudo-observed marginal flow study.


"""

import itertools
import pandas as pd
import geopandas as gpd
from pygeohydro import NWIS
import pynhd as pynhd

from methods.utils.directories import DATA_DIR, OUTPUT_DIR
from methods.utils.filter import filter_usgs_nwis_query_by_type, filter_drb_sites
from methods.retrieval.NLDI import add_comid_metadata_to_gage_data
from methods.spatial.upstream import get_upstream_gauges_for_id_list, update_upstream_gauge_file
from methods.spatial.catchments import get_basin_geometry
from methods.utils.constants import GEO_CRS
from methods.utils.nid import get_nid_data, get_dam_storages
from methods.utils.nid import get_dam_data_within_catchment
from methods.generator.pywr_drb_node_data import nhm_site_matches, obs_pub_site_matches

# Filenames for upstream gauge jsons
from methods.spatial.upstream import station_upstream_gauge_file


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

GET_DAYMET_DATA = False
GET_NLDI_DATA = True

# list of known reservoir inflow sites at pywrdrb nodes
pywrdrb_obs_sites = [s for node, sites in obs_pub_site_matches.items() for s in sites if sites is not None]


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
    stations += pywrdrb_obs_sites

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
    
    ### Get comid numbers for each gauge
    print("Gathering COMID numbers for gauges.")
    gage_data_with_comid = add_comid_metadata_to_gage_data(gage_data)

    gage_data_with_comid = gage_data_with_comid.drop_duplicates(subset=['comid'])
    gage_data_with_comid = gage_data_with_comid.dropna(how='any', axis=0)
    print(f"COMID numbers gathered for {gage_data_with_comid.shape[0]} of {gage_data.shape[0]} USGS streamflow gauges.")

    gage_data_with_comid.to_csv(f'{DATA_DIR}/USGS/{boundary}_usgs_metadata.csv', index=True)

    
    ### Repeat for PywrDRB node prediction locations 
    # Load locations (node name is index)
    pywrdrb_node_locations = pd.read_csv(f'{DATA_DIR}/prediction_locations.csv', index_col=0)
    
    # get comids
    print("Gathering COMID numbers for PywrDRB node locations.")
    pywrdrb_node_metadata = pywrdrb_node_locations.copy()
    pywrdrb_node_metadata_with_comid = add_comid_metadata_to_gage_data(pywrdrb_node_metadata)
    pywrdrb_node_metadata_with_comid.to_csv(f'{DATA_DIR}/USGS/pywrdrb_node_metadata.csv', index=True)

    
    ### Set up dict to translate between comid and site_no
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

    station_catchment = None
    for i, row in gage_data_with_comid.iterrows():
        comid = row['comid']
        station_id = row.name
        basin = get_basin_geometry(station_id, fsource='nwissite')
        
        if basin is None:
            print(f"Failed to get catchment geometry for {station_id}.")
        else:
            basin = basin.to_crs(GEO_CRS)
            if station_catchments is None:
                station_catchments = gpd.GeoDataFrame(geometry=[basin.geometry[0]], 
                                                    index=[station_id], crs=GEO_CRS)
            else:
                station_catchments.loc[station_id] = basin.geometry[0]

    # Save to file
    station_catchments = station_catchments.explode(index_parts=False)
    station_catchments.to_file(f'{DATA_DIR}/NHD/{boundary}_station_catchments.shp')
    print(f"Saved catchment geometries for {len(station_catchments)} stations to {DATA_DIR}/NHD/{boundary}_station_catchments.shp.")


    ### Repeat for PywrDRB nodes
    
    pywrdrb_node_catchments = None
    for cid in pywrdrb_node_metadata_with_comid['comid']:
        try:
            basin = get_basin_geometry(cid, fsource='comid')
            if basin is None:
                print(f"Failed to get catchment geometry for PywrDRB node with COMID: {cid}.")
            else:
                basin = basin.to_crs(GEO_CRS)
                station_catchments.loc[cid] = basin.geometry[0]
                if pywrdrb_node_catchments is None:
                    pywrdrb_node_catchments = gpd.GeoDataFrame(geometry=[basin.geometry[0]], 
                                                        index=[cid], crs=GEO_CRS)
                else:
                    pywrdrb_node_catchments.loc[cid] = basin.geometry[0]
        except Exception as e:
            print(f"Failed to get catchment geometry for PywrDRB node with COMID: {cid}: {e}")

    # Save PywrDRB node catchments to file
    pywrdrb_node_catchments = pywrdrb_node_catchments.explode(index_parts=False)
    pywrdrb_node_catchments.to_file(f'{DATA_DIR}/NHD/{boundary}_pywrdrb_node_catchments.shp')
    print(f'Save PywrDRB node catchments to {DATA_DIR}/NHD/{boundary}_pywrdrb_node_catchments.shp.')


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

    # Save dataframe to csv
    catchment_dams_df.to_csv(f'{DATA_DIR}/NID/{boundary}_catchment_dam_summary.csv', index=True)
    
    ####################################################################################################    
    ### Classify catchments as managed or unmanaged ###
    ####################################################################################################

    ## Catchments with limited storage in the entire upstream catchment 
    unmanaged_catchment_stations = catchment_dams_df[catchment_dams_df['total_storage'] < MIN_ALLOWABLE_STORAGE].index.tolist()
    unmanaged_catchments_gauge_data = gage_data_with_comid.loc[unmanaged_catchment_stations]
    unmanaged_catchments_gauge_data.to_csv(f'{DATA_DIR}/USGS/{boundary}_unmanaged_usgs_metadata.csv', index=True)

    # take a subset of the catchments that are unmanaged
    unmanaged_catchments = station_catchments.loc[unmanaged_catchment_stations]

    ####################################################################################################
    ### DayMet data ###
    ####################################################################################################
    
    if GET_DAYMET_DATA:
        import concurrent.futures
        from methods.retrieval.DayMet import get_catchment_dayment
        
        daymet_start_date = '1980-01-01'
        daymet_end_date = '2020-12-31'
        daymet_timescale = 'daily'
        
        daymet_vars = ['prcp', 'tmax', 'tmin', 'pet']
        daymet_vars = [f'{v}_{agg}' for v in daymet_vars for agg in ['sum', 'mean']]
        
        N_CORE = 10
        
        # This is slow; run using multithreading
        # ONLY GETTING DATA FOR UNMANAGED CATCHMENTS and PywrDRB sites
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORE) as executor:
            
            # Prepare a list of jobs
            futures = [executor.submit(get_catchment_dayment, catchment, index, daymet_start_date, daymet_end_date, 
                                       daymet_timescale, daymet_vars) for index, catchment in unmanaged_catchments.iterrows()]

            # initialize dataframes to combine results
            daymet_dfs = {v: pd.DataFrame(columns=unmanaged_catchments.index) for v in daymet_vars}
            
            for future in concurrent.futures.as_completed(futures):
                catchment_dayment_data = future.result()

                # store in local data
                if catchment_dayment_data is not None:
                    for v in daymet_vars:
                        daymet_dfs[v][catchment_dayment_data['index']] = catchment_dayment_data[v]
                
                # get dates index only once
                if 'daymet_dates' not in locals() and catchment_dayment_data is not None:
                    daymet_dates = catchment_dayment_data['date']
        
        ## Export each
        for v in daymet_vars:
            daymet_dfs[v].to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_{v}_{daymet_timescale}.csv')
         
                    
        ## Repeat for PywrDRB node catchments
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORE) as executor:
            
            futures = [executor.submit(get_catchment_dayment, catchment, index, daymet_start_date, daymet_end_date,
                                       daymet_timescale, daymet_vars) for index, catchment in pywrdrb_node_catchments.iterrows()]
            
            # initialize dataframes to combine results
            daymet_dfs = {v: pd.DataFrame(columns=pywrdrb_node_catchments.index) for v in daymet_vars}
            
            for future in concurrent.futures.as_completed(futures):
                catchment_dayment_data = future.result()

                # store in local data
                if catchment_dayment_data is not None:
                    for v in daymet_vars:
                        daymet_dfs[v][catchment_dayment_data['index']] = catchment_dayment_data[v]
                
                # get dates index only once
                if 'daymet_dates' not in locals() and catchment_dayment_data is not None:
                    daymet_dates = catchment_dayment_data['date']
        
        # Export each
        for v in daymet_vars:
            daymet_dfs[v].to_csv(f'{DATA_DIR}/Daymet/pywrdrb_node_{v}_{daymet_timescale}.csv')


    ####################################################################################################
    ### NLDI catchment characteristic data ###
    ####################################################################################################
    
    if GET_NLDI_DATA:
        # Initialize the NLDI database
        nldi = pynhd.NLDI()

        nldi_comid_sites = unmanaged_catchments_gauge_data['comid']

        ## Use the station IDs to retrieve basin information
        tot_chars = nldi.getcharacteristic_byid(nldi_comid_sites, fsource = 'comid', 
                                                char_type= "tot")
        local_chars = nldi.getcharacteristic_byid(nldi_comid_sites, fsource = 'comid',
                                                    char_type= "local")

        cat_chars = pd.concat([tot_chars, local_chars], axis=1)
        cat_chars['comid'] = cat_chars.index
        cat_chars.to_csv(f'{DATA_DIR}/NLDI/{boundary}_usgs_nldi_catchment_characteristics.csv', index=True)
        print(f'Found characteristics for {cat_chars.shape} of {gage_data.shape} basins.')


        ### Repeat for PywrDRB nodes
        pywrdrb_node_comids = pywrdrb_node_metadata_with_comid['comid']
        tot_chars = nldi.getcharacteristic_byid(pywrdrb_node_comids, fsource = 'comid', 
                                                char_type= "tot")
        local_chars = nldi.getcharacteristic_byid(pywrdrb_node_comids, fsource = 'comid',
                                                    char_type= "local")
        
        cat_chars = pd.concat([tot_chars, local_chars], axis=1)
        cat_chars['comid'] = cat_chars.index
        cat_chars.to_csv(f'{DATA_DIR}/NLDI/pywrdrb_node_nldi_catchment_characteristics.csv', index=True)
        print(f'Found characteristics for {cat_chars.shape} of {pywrdrb_node_metadata_with_comid.shape} PywrDRB nodes.')


    ####################################################################################################
    ### Get streamflow timeseries from NWIS
    ####################################################################################################
    
    ## List of unmanaged stations which made it this far
    all_stations = station_catchments.index.unique()
    
    unmanaged_catchment_stations = unmanaged_catchments.index.to_list() + pywrdrb_obs_sites
    unmanaged_catchment_stations = list(set(unmanaged_catchment_stations))

    ## Use NWIS to get timeseries data
    nwis = NWIS()
    Q_all = nwis.get_streamflow(all_stations, dates)
    Q_all.index = pd.to_datetime(Q_all.index.date)
    Q_all.columns = [usgs_id.split('-')[1] for usgs_id in Q_all.columns]

    # Subset just unmanaged catchments
    Q_unmanaged_total = Q_all.loc[:, unmanaged_catchment_stations]

    # Export
    Q_all.to_csv(f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_cms.csv', 
                                sep=',')

    Q_unmanaged_total.to_csv(f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_unmanaged_cms.csv',
                                sep=',')
    
    print(f'Got streamflow timeseries at all sites.')
    print('DONE!')
    
    
    
    
    ####################################################################################################
    ### OLD: marginal catchment retrieval code; wasn't super useful, but keep for now. ###
    ####################################################################################################
    
    ### Get marginal catchment geometries for each station
    # station_marginal_catchments = station_catchments.copy()
    # station_marginal_catchments['geometry'] = None

    # for id in station_marginal_catchments.index:
    #     if id not in station_upstream_gauges.keys():
    #         continue
    #     immediate_upstream = get_immediate_upstream_sites(station_upstream_gauges, id)
    #     if len(immediate_upstream) > 0:
    #         catchments = station_catchments.loc[immediate_upstream]
    #         agg_upstream_catchment = catchments.unary_union
            
    #         # Get difference between total and agg_upstream_catchment geometry
    #         station_catchment_geom = station_catchments.loc[id, 'geometry']
    #         station_marginal_catchment_geometry = station_catchment_geom.difference(agg_upstream_catchment)
            
            
    #         # if multypolygon, take largest
    #         if station_marginal_catchment_geometry.geom_type == 'MultiPolygon':
    #             station_marginal_catchment_geometry_fragments = gpd.GeoDataFrame({'geometry':[station_marginal_catchment_geometry]}).explode(index_parts=False)
    #             station_marginal_catchment_geometry_fragments.reset_index(drop=True, inplace=True)
    #             argmax_area = station_marginal_catchment_geometry_fragments.area.argmax()
    #             station_marginal_catchment_geometry = station_marginal_catchment_geometry_fragments.loc[argmax_area, 'geometry']
                
    #         station_marginal_catchments.loc[id, 'geometry'] = station_marginal_catchment_geometry
            
    #         # Get the ratio of marginal to total catchment area
    #         station_marginal_catchments.loc[id, 'marginal_ratio'] = station_marginal_catchments.loc[id, 'geometry'].area / station_catchments.loc[id, 'geometry'].area
            
    #     else:
    #         station_marginal_catchments.loc[id, 'geometry'] = station_catchments.loc[id, 'geometry']
    #         station_marginal_catchments.loc[id, 'marginal_ratio'] = 1.0
            
    # station_marginal_catchments.explode(index_parts=False, inplace=True)
    # station_marginal_catchments.to_file(f'{DATA_DIR}/NHD/{boundary}_station_marginal_catchments.shp')
    
    ## Catchments with limited storage in the marginal catchment
    # unmanaged_marginal_catchments = marginal_catchment_dams_df[marginal_catchment_dams_df['total_storage'] < MIN_ALLOWABLE_STORAGE].index.tolist()
    # unmanaged_marginal_catchment_gauge_data = gage_data_with_comid.loc[unmanaged_marginal_catchments]
    # unmanaged_marginal_catchment_gauge_data.to_csv(f'{DATA_DIR}/USGS/{boundary}_unmanaged_marginal_usgs_metadata.csv', index=True)

    # print(f'{unmanaged_catchments_gauge_data.shape[0]} unmanaged catchments and {unmanaged_marginal_catchment_gauge_data.shape[0]} unmanaged marginal catchments found.')
