"""
DESCRIPTION:
This script retrieves all necessary data 
for the historic reconstruction of streamflow in the Delaware River Basin (DRB).

Data is stored in /data/ directory.

The data includes, for both USGS gauge stations and PywrDRB nodes:
- station metadata
- catchment geometries
- upstream gauges
- catchment dam data from the National Inventory of Dams (NID)
- Daymet precipitation data
- NLDI catchment characteristics
- NWIS streamflow data
"""

from pathlib import Path

import logging
import pandas as pd
import geopandas as gpd
import pynhd as pynhd

from mpi4py import MPI

from methods.retrieval.DayMet import DayMetManager, DayMetProcessor, DayMetRetriever
from methods.retrieval.DayMet import process_daymet_data_for_catchments
from methods.retrieval.NWIS import RegionalGageDataRetriever
from methods.spatial.upstream import UpstreamGaugeManager
from methods.spatial.catchments import CatchmentManager
from methods.retrieval.NID import NIDManager

from methods.utils.directories import DATA_DIR, OUTPUT_DIR, PYWRDRB_DIR, FIG_DIR


from config import DATES, BBOX
from config import GEO_CRS
from config import MIN_YEARS
from config import FILTER_DRB, BOUNDARY, FILTER_DRB

# Filenames for data outputs
from config import USGS_GAGE_CATCHMENT_FILE, PYWRDRB_NODE_CATCHMENT_FILE
from config import STATION_UPSTREAM_GAGE_FILE, COMID_UPSTREAM_GAGE_FILE, IMMEDIATE_UPSTREAM_GAGE_FILE
from config import USGS_GAGE_METADATA_FILE, PYWRDRB_NODE_METADATA_FILE
from config import PYWRDRB_NODE_METADATA_GEO_FILE
from config import PREDICTION_LOCATIONS_FILE
from config import NID_METADATA_FILE, NID_SUMMARY_FILE
from config import USGS_NLDI_CHARACTERISTICS_FILE, PYWRDRB_NLDI_CHARACTERISTICS_FILE
from config import ALL_USGS_DAILY_FLOW_FILE, PYWRDRB_USGS_DAILY_FLOW_FILE
from config import ALL_SITE_METADATA_FILE

from pywrdrb_node_data import obs_site_matches


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)


####################################################################################################
### Specifications ###
####################################################################################################

REQUERY_GAUGE_META = False
GET_UPSTREAM_STATIONS = False
GET_STATION_CATCHMENTS = True
GET_NID_DATA = False
GET_DAYMET_DATA = True
GET_NLDI_DATA = False
GET_NWIS_DATA = False



if __name__ == "__main__":
    
    ####################################################################################################
    ### Query all available gauges from NWIS ###
    ####################################################################################################

    if rank == 0 and REQUERY_GAUGE_META:
        
        ## Retrieve station data from NWIS
        retriever = RegionalGageDataRetriever()
        gage_data = retriever.get_stations_in_bbox(BBOX, return_gage_data = True)        

        # Get comid numbers for each gauge using NLDI
        print("Gathering COMID numbers for all USGS gauges...")
        gage_data_with_comid = retriever.add_comid_metadata_to_gage_data(return_gage_data=True)
        gage_data_with_comid.to_csv(USGS_GAGE_METADATA_FILE, index=True)
        
        # NHM IDs for "points of interest" (POIs) which are bath USGS stations and NHM segments
        nhm_poi_ids = pd.read_csv(f'{DATA_DIR}/NHMv10/meta/nhm_poi_ids.csv', index_col=0)
        nhm_poi_ids.drop(['long', 'lat', 'reachcode'], axis=1, inplace=True)

        # Merge metadata to include USGS ID, COMID, and NHM ID
        gauge_metadata = gage_data_with_comid.reset_index()
        all_site_metadata = pd.merge(nhm_poi_ids, gauge_metadata, on='comid', how='outer')
        all_site_metadata.to_csv(ALL_SITE_METADATA_FILE)


        ## Repeat for PywrDRB node prediction locations 
        # Load locations (node name is index)
        pywrdrb_node_locations = pd.read_csv(PREDICTION_LOCATIONS_FILE, index_col=0)

        # get comids
        print("Gathering COMID numbers for PywrDRB node locations...")
        pywrdrb_node_metadata = pywrdrb_node_locations.copy()
        retriever = RegionalGageDataRetriever()
        retriever.gage_data = pywrdrb_node_metadata
        pywrdrb_node_metadata_with_comid = retriever.add_comid_metadata_to_gage_data(return_gage_data=True,
                                                                                     drop_duplicates=False,
                                                                                     drop_na=False)
        
        pywrdrb_node_metadata_with_comid.to_csv(PYWRDRB_NODE_METADATA_FILE, 
                                                index=True)

        pywrdrb_gdf = gpd.GeoDataFrame(
                        pywrdrb_node_metadata_with_comid, 
                        geometry=gpd.points_from_xy(pywrdrb_node_metadata_with_comid['long'], pywrdrb_node_metadata_with_comid['lat']),
                        crs=GEO_CRS  
                    )

        pywrdrb_gdf.to_file(PYWRDRB_NODE_METADATA_GEO_FILE)

    elif rank == 0 and not REQUERY_GAUGE_META:
        gage_data_with_comid = pd.read_csv(USGS_GAGE_METADATA_FILE, 
                                           index_col=0, 
                                           dtype={'site_no':str, 'comid':str})

        pywrdrb_node_metadata_with_comid = pd.read_csv(PYWRDRB_NODE_METADATA_FILE, 
                                                       index_col=0,
                                                       dtype={'comid':str})
    else:
        gage_data_with_comid = None
        pywrdrb_node_metadata_with_comid = None
    
    # share data with all processes
    gage_data_with_comid = comm.bcast(gage_data_with_comid, root=0)
    pywrdrb_node_metadata_with_comid = comm.bcast(pywrdrb_node_metadata_with_comid, root=0)
        
    
    ####################################################################################################
    ### Get list of gauges upstream of each station ###
    ####################################################################################################

    if rank == 0 and GET_UPSTREAM_STATIONS:
        print('Identifying upstream gauges for each gauge...')
        
        station_list = gage_data_with_comid.index.unique()
        
        ## Get upstream gauges for each station
        upstream_manager = UpstreamGaugeManager(data_dir=DATA_DIR)
        station_upstream_gauges_dict = upstream_manager.get_gauges_for_multiple_sites(station_list,
                                                                                 fsource='nwissite',
                                                                                 overwrite=True,
                                                                                 restrict_to_list=True)
        
        upstream_manager.update_gauge_file(station_upstream_gauges_dict,
                                           file=STATION_UPSTREAM_GAGE_FILE,
                                           overwrite=True)
        
        immediate_upstream_gauges_dict = upstream_manager.find_immediate_upstream_sites(station_upstream_gauges_dict)
        upstream_manager.update_gauge_file(immediate_upstream_gauges_dict,
                                             file=IMMEDIATE_UPSTREAM_GAGE_FILE,
                                             overwrite=True)
        
        print(f'Got upstream gauges for {len(station_upstream_gauges_dict)} stations.')
        print(f'Identified immediate upstream gauges for {len(immediate_upstream_gauges_dict)} stations.')
        

    ####################################################################################################
    ### Get catchment geometries ###
    #################################################################################################### 

    if rank == 0 and GET_STATION_CATCHMENTS:

        station_to_comid = {}
        comid_to_station = {}
        for station, row in gage_data_with_comid.iterrows():
            station_to_comid[station] = row['comid']
            comid_to_station[row['comid']] = station

        station_list = gage_data_with_comid.index.unique().tolist()
        station_comid_list = gage_data_with_comid['comid'].unique().tolist()
        
        print(f'Getting catchment geometries for {len(station_list)} USGS stations...')
        

        ## Get catchment geometries for each station
        catchment_manager = CatchmentManager()
        station_catchments = catchment_manager.get_catchments_from_list(station_comid_list, 
                                                                        fsource='comid')
        # re-index using station id instead of comid
        station_catchments['site_no'] = [comid_to_station[comid] for comid in station_catchments.index]
        station_catchments = station_catchments.set_index('site_no', drop=True)

        catchment_manager.save_catchments(station_catchments, 
                                          USGS_GAGE_CATCHMENT_FILE)
        print(f"Saved catchment geometries for {len(station_catchments)} stations to {USGS_GAGE_CATCHMENT_FILE}")
        
        
        # Repeat for PywrDRB nodes
        catchment_manager = CatchmentManager()
        pywrdrb_comid_list = pywrdrb_node_metadata_with_comid['comid'].values.tolist()
        pywrdrb_node_catchments = catchment_manager.get_catchments_from_list(pywrdrb_comid_list,
                                                                             fsource='comid',
                                                                             keep_largest=False) # keep duplicates; same catchments for different pywrdrb nodes
        catchment_manager.save_catchments(pywrdrb_node_catchments,
                                          PYWRDRB_NODE_CATCHMENT_FILE)

        print(f"Saved catchment geometries for {len(pywrdrb_node_catchments)} PywrDRB nodes to {PYWRDRB_NODE_CATCHMENT_FILE}")

    else:
        catchment_manager = CatchmentManager()
        station_catchments = catchment_manager.load_catchments(USGS_GAGE_CATCHMENT_FILE)
        pywrdrb_node_catchments = catchment_manager.load_catchments(PYWRDRB_NODE_CATCHMENT_FILE)
        
        if rank==0:
            print(f"Loaded catchment geometries for {len(station_catchments)} stations")
            print(f"Loaded catchment geometries for {len(pywrdrb_node_catchments)} PywrDRB nodes")
        


    ####################################################################################################
    ### Get dam data from National Inventory of Dams (NID) ###
    #################################################################################################### 
    if rank == 0 and GET_NID_DATA:
        print('Getting dam data across the basin from NID...')
        nid_manager = NIDManager()

        ## Get NID metadata for all dams in NY, NJ, PA, DE
        drb_dams_gdf = nid_manager.get_nid_data(min_storage=5000,
                                               filter_drb=FILTER_DRB, 
                                               filter_bbox=None)
        dam_storage = nid_manager.add_dam_storages_to_gdf(drb_dams_gdf)
        
        nid_manager.save(drb_dams_gdf, NID_METADATA_FILE)
        

        ## Find dams within each catchment
        catchment_dams_df = nid_manager.get_dams_within_catchments(station_catchments, 
                                                                      drb_dams_gdf)
        nid_manager.save(catchment_dams_df, 
                         NID_SUMMARY_FILE,
                         index=False)
    else:
        # Load NID data
        nid_manager = NIDManager()
        catchment_dams_df = nid_manager.load(NID_SUMMARY_FILE, index_col=0)


    ####################################################################################################
    ### DayMet data ###
    ####################################################################################################


    if rank == 0 and GET_DAYMET_DATA:
        
        DAYMET_DATA_DIR = Path(DATA_DIR) / 'Daymet'
        
        # for timescale in ['monthly', 'annual']:                

        #     # Initialize classes
        #     retriever = DayMetRetriever(timescale=timescale)
        #     manager = DayMetManager(DAYMET_DATA_DIR)
        #     processor = DayMetProcessor()
                
            
    
        #     # Get initial dataset
        #     ds_raw = retriever.get_daymet_in_bbox(BBOX)
        #     ds_prcp = ds_raw[['prcp']]
            
        #     # Save raw data
        #     manager.save_netcdf(ds_prcp, f"drb_daymet_{timescale}_prcp.nc")
            
        #     print(f'Saved raw {timescale} prcp data as netCDF')

        #     # close ds_raw
        #     del ds_raw
        #     del ds_prcp


        
        for catchment_type in ['gauge', 'pywrdrb']:
            for timescale in ['monthly', 'annual']:
        
                print(f'Processing DayMet data for {catchment_type} locations at {timescale} timescale...')
                
                # Initialize classes
                retriever = DayMetRetriever(timescale=timescale)
                manager = DayMetManager(DAYMET_DATA_DIR)
                processor = DayMetProcessor()
                
                
                if catchment_type == 'gauge':
                    catchments = station_catchments.copy()
                elif catchment_type == 'pywrdrb':
                    catchments = pywrdrb_node_catchments.copy()
                    catchments = catchments.loc[:, ['geometry']]
                
                ds = manager.load_netcdf(f"drb_daymet_{timescale}_prcp.nc")
    
                    
                process_daymet_data_for_catchments(catchment_type,
                                                   ds,
                                                   catchments,
                                                   timescale,
                                                   processor=processor,
                                                   manager=manager,
                                                   plot_results=True,
                                                   figdir=f'{FIG_DIR}/daymet_prcp/')
                
        
            
    ####################################################################################################
    ### NLDI catchment characteristic data ###
    ####################################################################################################

    if GET_NLDI_DATA:
        if rank == 0:
            print(f'Getting NLDI catchment characteristics for {len(gage_data_with_comid)} sites.')
    
        # Assuming each process will handle its portion of catchments
        if rank == 0:
            station_ids = list(station_catchments.index.unique())
            station_comids = list(gage_data_with_comid.loc[station_ids, 'comid'].unique())
                
            # Split indices into nearly equal parts for each process
            chunks = [station_comids[i::size] for i in range(size)]
        else:
            chunks = None


        # Scatter chunks to each process
        comm.barrier()
        catchment_chunks = comm.scatter(chunks, root=0)
        
        # Initialize the NLDI database
        nldi = pynhd.NLDI()
        rank_tot_chars = nldi.getcharacteristic_byid(catchment_chunks, 
                                                     fsource = 'comid',
                                                     char_type= "tot")
        rank_local_chars = nldi.getcharacteristic_byid(catchment_chunks, 
                                                       fsource = 'comid',
                                                       char_type= "local")
        
        print(f'Rank {rank} done with NLDI retrieval for {len(catchment_chunks)} catchments.')
        
        # Gather at the root process
        tot_chars = comm.gather(rank_tot_chars, root=0)
        local_chars = comm.gather(rank_local_chars, root=0)
        # div_chars = comm.gather(rank_div_chars, root=0)
        
        if rank == 0:
            tot_chars = pd.concat(tot_chars, axis=0)
            local_chars = pd.concat(local_chars, axis=0)

            # combine into 1 dataframe
            cat_chars = pd.concat([tot_chars, local_chars], axis=1) 
            cat_chars['comid'] = cat_chars.index
            cat_chars.to_csv(USGS_NLDI_CHARACTERISTICS_FILE, index=True)
            print(f'Found characteristics for {cat_chars.shape} of {len(station_comids)} basins.')

        ### Repeat for PywrDRB nodes
        if rank ==0:
            print(f'Getting NLDI catchment characteristics for {len(pywrdrb_node_metadata_with_comid)} PywrDRB nodes.')
            pywrdrb_node_comids = pywrdrb_node_metadata_with_comid['comid'].values.tolist()
            tot_chars = nldi.getcharacteristic_byid(pywrdrb_node_comids, fsource = 'comid', 
                                                    char_type= "tot")
            local_chars = nldi.getcharacteristic_byid(pywrdrb_node_comids, fsource = 'comid',
                                                        char_type= "local")
            
            cat_chars = pd.concat([tot_chars, local_chars], axis=1)
            cat_chars['comid'] = cat_chars.index
            cat_chars.to_csv(PYWRDRB_NLDI_CHARACTERISTICS_FILE, index=True)
            print(f'Found characteristics for {cat_chars.shape} of {pywrdrb_node_metadata_with_comid.shape[0]} PywrDRB nodes.')


    ####################################################################################################
    ### Get streamflow timeseries from NWIS
    ####################################################################################################
    if rank ==0 and GET_NWIS_DATA:
        print('Getting streamflow timeseries from NWIS...')
        ## List of stations that made it this far
        all_stations = station_catchments.index.unique()
        
        ## Use NWIS to get timeseries data
        retriever = RegionalGageDataRetriever()
        Q_all = retriever.get_streamflow_data(all_stations, DATES)

        # Export
        Q_all.to_csv(ALL_USGS_DAILY_FLOW_FILE, sep=',')        


        print('Getting observed streamflow timeseries for PywrDRB nodes...')
        pywrdrb_stations = []
        for node, sites in obs_site_matches.items():
            if len(sites)>0:
                for s in sites:
                    pywrdrb_stations.append(s)
        
        Q_pywrdrb = retriever.get_streamflow_data(pywrdrb_stations, DATES)
    
        for s in pywrdrb_stations:
            assert(f'{s}' in Q_pywrdrb.columns),'PywrDRB gauge {s} is missing from the data.'

        # Export
        print(f'Got streamflow timeseries at all PywrDRB nodes. Saving.')
        Q_pywrdrb.to_csv(PYWRDRB_USGS_DAILY_FLOW_FILE, sep=',')
        
        # Q_pywrdrb.to_csv(f'{PYWRDRB_DIR}/input_data/usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', sep=',')
    
    
    ### Close out
    if rank == 0:
        print('DONE WITH DATA RETRIEVAL!')
        
        


    ##############################################################################
    ### Old code
    ##############################################################################



    # comm.barrier()

    # def safe_get_data_for_catchment(retriever, catchment, idx, max_retries=5, base_delay=5.0):
    #     import random
    #     from requests.exceptions import RequestException
    #     attempt = 0
    #     while attempt <= max_retries:
    #         try:
    #             return retriever.get_data_for_catchment(catchment, idx)
    #         except RequestException as e:
    #             wait_time = base_delay * (2 ** attempt) + random.uniform(0, 2)
    #             logging.info(f'Rank {rank}: Error processing {idx}, retrying after {wait_time:.1f}s...')
    #             time.sleep(wait_time)
    #             attempt += 1
    #         except Exception as e:
    #             logging.error(f'Rank {rank}: Unexpected error processing {idx}: {e}')
    #             break
    #     return None
        
    # if GET_DAYMET_DATA:
    #     for catchment_type in ['gauge', 'pywrdrb']:
    #         for ts in ['monthly', 'annual']:
    #             daymet_timescale = ts
    #             freq = 'YS' if daymet_timescale == 'annual' else 'MS'

    #             if rank == 0:
    #                 print(f'Rank {rank}: Getting DayMet data for {catchment_type} locations at {ts} timescale...')

    #             retriever = DayMetDataRetriever(
    #                 start_date=daymet_start_date,
    #                 end_date=daymet_end_date,
    #                 variables=daymet_vars,
    #                 timescale=daymet_timescale,
    #                 aggregations=aggregation_types,
    #                 pet=daymet_pet,
    #                 bygeom=False,
    #                 bystac=True,
    #             )

    #             if rank == 0:
    #                 # Load and preprocess catchments on the root process only
    #                 if catchment_type == 'gauge':
    #                     catchments = station_catchments.copy()
    #                 else:
    #                     catchments = pywrdrb_node_catchments.copy()

    #                 # Drop duplicates to ensure each catchment is unique
    #                 catchments = catchments.drop_duplicates()

    #                 # Pre-simplify geometries
    #                 catchments['geometry'] = catchments['geometry'].apply(
    #                     lambda g: g.simplify(0.01) if isinstance(g, Polygon) else g
    #                 )

    #                 # Reset index to ensure a clean and unique integer index for splitting
    #                 catchments = catchments.reset_index(drop=True)

    #                 # Create a list of tuples (new_index, row_dict)
    #                 indices = list(catchments.iterrows())

    #                 # Sort to ensure stable ordering
    #                 indices.sort(key=lambda x: x[0])

    #                 # Split indices into nearly equal parts for each process
    #                 chunks = [indices[i::size] for i in range(size)]
    #             else:
    #                 chunks = None

    #             comm.barrier()
    #             # Scatter chunks to each process
    #             catchment_chunks = comm.scatter(chunks, root=0)

    #             # Each rank now operates on a unique subset of catchments
    #             results_dict = {v: {} for v in daymet_vars_agg}

    #             for idx, row in catchment_chunks:
    #                 # If needed, a small sleep to avoid server overload
    #                 time.sleep(np.random.uniform(0, 0.5))

    #                 print(f"Rank {rank} processing row {row['site_no']} with {row.index}...")
    #                 catchment_dayment_data = retriever.get_data_for_catchment(row, row['site_no'])


    #                 if catchment_dayment_data is not None:
    #                     for v in daymet_vars_agg:
    #                         results_dict[v][catchment_dayment_data['index']] = catchment_dayment_data[v]
    #                     logging.info(f'Rank {rank} done with DayMet retrieval for {idx}.')
    #                 else:
    #                     logging.info(f'Rank {rank} failed with DayMet retrieval for {idx}. Got None.')

    #             # Gather at the root process
    #             all_results = comm.gather(results_dict, root=0)

    #             if rank == 0:
    #                 logging.info('Rank 0: Organizing results...')
    #                 final_data = {v: {} for v in daymet_vars_agg}
    #                 for result in all_results:
    #                     if result is None:
    #                         continue
    #                     for v in daymet_vars_agg:
    #                         final_data[v].update(result[v])

    #                 final_results = {}
    #                 # Convert to DataFrames after gathering all data
    #                 for v in daymet_vars_agg:
    #                     sorted_keys = sorted(final_data[v].keys())
    #                     if not sorted_keys:
    #                         final_results[v] = pd.DataFrame()
    #                         continue
    #                     arr = np.column_stack([final_data[v][k] for k in sorted_keys])
    #                     final_results[v] = pd.DataFrame(arr, columns=sorted_keys)

    #                 if not final_results[daymet_vars_agg[0]].empty:
    #                     n_periods = final_results[daymet_vars_agg[0]].shape[0]
    #                     datetime_index = pd.date_range(daymet_start_date, periods=n_periods, freq=freq)

    #                     for v in daymet_vars_agg:
    #                         if v not in final_results or final_results[v].empty:
    #                             continue
    #                         if catchment_type == 'gauge':
    #                             daymet_fname = f'{DATA_DIR}/Daymet/{boundary}_catchment_{v}_{daymet_timescale}.csv'
    #                         else:
    #                             daymet_fname = f'{DATA_DIR}/Daymet/pywrdrb_node_{v}_{daymet_timescale}.csv'

    #                         final_results[v].index = datetime_index
    #                         final_results[v].index.name = 'datetime'
    #                         final_results[v].to_csv(daymet_fname)

    #                     logging.info(f'Rank 0: DayMet data gathered and saved for {catchment_type} locations.')
    #                 else:
    #                     logging.info("Rank 0: No data retrieved for this configuration.")


