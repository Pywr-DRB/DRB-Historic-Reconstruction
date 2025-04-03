from mpi4py import MPI
import pandas as pd
import json
import numpy as np
import pickle
import sys
import os

from methods.load.data_loader import Data
from methods.spatial.upstream import UpstreamGaugeManager

from config import FDC_QUANTILES
from config import STATION_UPSTREAM_GAGE_FILE, SEED
from config import START_YEAR, END_YEAR, N_ENSEMBLE
from config import ENSEMBLE_K_MIN, ENSEMBLE_K_MAX
from config import OUTPUT_DIR

from methods.single_site_generator import predict_single_gauge

from methods.processing.hdf5 import export_ensemble_to_hdf5
from methods.processing.hdf5 import combine_hdf5_files

from methods.bias_correction.apply import apply_bias_correction_to_ensemble

    

def leave_one_out_prediction_mpi(Q_obs_unmanaged,
                                 nhm_fdcs,
                                 diagnostic_gauge_meta,
                                 unmanaged_gauge_meta,
                                 K,
                                 gauge_subcatchments,
                                 n_realizations=1,
                                 start_year=1983,
                                 end_year=2020,
                                 output_filename=None,
                                 rank=None):
    """
    Parallel version of the leave-one-out experiment for QPPQ reconstruction.
    Similar arguments as the original function.

    Args:
    """
    
    # Distribute rows to processes
    if rank == 0:
        # Split the gauge matches DataFrame and distribute it
        split_data = np.array_split(diagnostic_gauge_meta, size)
    else:
        split_data = None
    
    # Scatter the split data to all processes
    local_data = comm.scatter(split_data, root=0)

    # Each process runs predictions on its chunk of data
    local_results = []
    for row in local_data.itertuples():
         
        # get fdc from nhm_fdcs
        fdc_prediction = nhm_fdcs.loc[row.Index, :].values
        
        
        result = predict_single_gauge(row = row, 
                                      fdc_prediction=fdc_prediction,
                                      unmanaged_gauge_flows=Q_obs_unmanaged,
                                      unmanaged_gauge_meta=unmanaged_gauge_meta,
                                      K=K,
                                      gauge_subcatchments=gauge_subcatchments,
                                      n_realizations=n_realizations,
                                      start_year=start_year,
                                      end_year=end_year)

        local_results.append(result)

    # Gather the results from all processes
    # pickle to make data size more managable
    pickled_local_results = [pickle.dumps(result[1]) for result in local_results]
    local_station_ids = [result[0] for result in local_results]
    
    all_pickled_results = comm.gather(pickled_local_results, root=0)
    all_station_ids = comm.gather(local_station_ids, root=0)

    if rank == 0:
        # de-pickle after gathering
        all_results = [pickle.loads(pickled_result) 
                        for sublist in all_pickled_results
                        for pickled_result in sublist]
        all_ids = [id for sublist in all_station_ids for id in sublist]
        
        
        flat_results = [(all_ids[i], all_results[i]) for i in range(len(all_results))]
        
        
        ##############
        ### Export ###
        ##############
        
        # Compile results into final data structure
        datetime_index = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
        
        if n_realizations == 1:
            # Combine into a DataFrame and save to csv
            Q_predicted_df = pd.DataFrame({station_id: Q for station_id, Q in flat_results}, index=datetime_index)
            Q_predicted_df.to_csv(f'{output_filename}.csv', sep=',')
        else:
            # Combine into a dictionary of DataFrames for ensemble & export to hdf5
            Q_predicted_ensemble = {station_id: pd.DataFrame(Q, columns=[f'{i}' for i in range(n_realizations)], 
                                                             index=datetime_index) for station_id, Q in flat_results}
            export_ensemble_to_hdf5(Q_predicted_ensemble, output_file=f'{output_filename}_ensemble.hdf5')
    return
    
    









##############################################################################
##############################################################################
##############################################################################

if __name__ == '__main__':

    # parallelization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set seed
    np.random.seed(SEED)

    # Specifications
    # limited reconstruction period since 
    # NHM data is only available for comparison 1983-2016    
    
    start_year = 1980
    end_year = 2017

    RERUN_QPPQ = False
    FDC_BIAS_CORRECTION = True
    
    N_ENSEMBLE = 500
    N_ENSEMBLE_SET = 100
    N_SETS = N_ENSEMBLE // N_ENSEMBLE_SET
    fdc_quantiles= FDC_QUANTILES
    
    #################
    ### Load data ###
    #################
    
    if rank == 0:

        data_loader = Data()

        ## Streamflow    
        Q_obs = data_loader.load(datatype='streamflow', sitetype='usgs', flowtype='obs')
        nhm_fdcs = data_loader.load(datatype='fdc', sitetype='usgs', flowtype='nhm', timescale='daily')
        
        diagnostic_gauge_meta = data_loader.load(datatype='metadata', sitetype='diagnostic')
        diagnostic_gauge_meta.set_index('site_no', inplace=True)
        
        unmanaged_gage_meta = data_loader.load(datatype='metadata', sitetype='unmanaged')
        unmanaged_gage_meta.set_index('site_no', inplace=True)
        
        # leave-one-out (loo) diagnostic sites
        loo_sites = diagnostic_gauge_meta.index.values
        

        # keep only unmanaged flow data
        unmanaged_gauge_flows = Q_obs.loc[:, unmanaged_gage_meta.index]

        # dict containing station:[upstream stations]
        with open(STATION_UPSTREAM_GAGE_FILE, 'r') as f:
            station_upstream_gauges = json.load(f)

        upstream_manager = UpstreamGaugeManager()

        for site in loo_sites:
            if site not in station_upstream_gauges:
                upstream_gauges = upstream_manager.get_gauges_for_single_site(fid=site, 
                                                                              fsource='nwissite')
                station_upstream_gauges[site] = upstream_gauges                

    else:
        unmanaged_gauge_flows = None
        nhm_fdcs = None
        diagnostic_gauge_meta = None
        unmanaged_gage_meta = None
        station_upstream_gauges = None
        
        
    # broadcast data to all processes
    unmanaged_gauge_flows = comm.bcast(unmanaged_gauge_flows, root=0)
    nhm_fdcs = comm.bcast(nhm_fdcs, root=0)
    diagnostic_gauge_meta = comm.bcast(diagnostic_gauge_meta, root=0)
    unmanaged_gage_meta = comm.bcast(unmanaged_gage_meta, root=0)
    station_upstream_gauges = comm.bcast(station_upstream_gauges, root=0)


    ####################################
    ### Run leave-one-out experiment ###
    ####################################
                                     
    ## Ensemble QPPQ
    # Loop through K values
    for K in range(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX):
        
        if rank == 0:
            print(f'Generating predictions with K={K} and {N_ENSEMBLE} realizations')
        
        # Memory issues arise when ensemble size is too large
        # Instead, run N sets of 100 realizations and combine after
        ensemble_set_filenames = []
        for set in range(N_SETS):   

            output_filename = f'{OUTPUT_DIR}/LOO/set{set}_loo_reconstruction_nhmv10_K{K}'
            ensemble_set_filenames.append(output_filename + '_ensemble.hdf5')

            if RERUN_QPPQ:
                leave_one_out_prediction_mpi(Q_obs_unmanaged=unmanaged_gauge_flows,
                                             nhm_fdcs=nhm_fdcs,
                                             diagnostic_gauge_meta=diagnostic_gauge_meta,
                                             unmanaged_gauge_meta=unmanaged_gage_meta,
                                             K=K,
                                             gauge_subcatchments=station_upstream_gauges,
                                             n_realizations=N_ENSEMBLE_SET,
                                             start_year=start_year,
                                             end_year=end_year,
                                             output_filename=output_filename,
                                             rank=rank)
        
    
        comm.barrier()
        
        ## Combine ensemble sets into a single file
        if rank == 0 and RERUN_QPPQ:
            print(f'Rank {rank} combining ensemble sets into a single file')
            
            output_filename = f'{OUTPUT_DIR}/LOO/loo_reconstruction_nhmv10_K{K}_ensemble.hdf5'
            
            combine_hdf5_files(ensemble_set_filenames, output_filename)
            
            # Delete individual files
            for f in ensemble_set_filenames:
                os.remove(f)

        
    ## Apply bias correction of FDCs    
    # split K values among processes
    K_values = np.arange(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX)
    K_split = np.array_split(K_values, size)
    rank_K_values = K_split[rank]
    
    for K in rank_K_values:        
        print(f'Rank {rank} applying bias correction to NHMv10 based predictions with K={K}')
        
        save_filtered_biases = True if K == ENSEMBLE_K_MIN else False
        bias_corrected_ensemble = apply_bias_correction_to_ensemble('nhmv10', 
                                                                    K,
                                                                    save_filtered_biases=save_filtered_biases)
        
        # Export bias corrected ensemble to HDF5
        fname = f'{OUTPUT_DIR}/LOO/loo_reconstruction_nhmv10_BC_K{K}_ensemble.hdf5'
        export_ensemble_to_hdf5(bias_corrected_ensemble, output_file=fname)
    
    comm.barrier()
    
    if rank == 0:
        print('### DONE WITH LEAVE-ONE-OUT RECONSTRUCTIONS! ###')