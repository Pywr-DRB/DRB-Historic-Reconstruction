import os
import numpy as np
import pandas as pd
import glob

from mpi4py import MPI

from config import SEED, N_ENSEMBLE, K_AGGREGATE, K_ENSEMBLE
from config import OUTPUT_DIR
from methods.pywrdrb_generator import generate_pywrdrb_reconstruction
from methods.processing.hdf5 import export_ensemble_to_hdf5

from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5

if __name__ == '__main__':

    # parallelization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # set seed
    np.random.seed(SEED)

    # Specifications
    start = 1945
    end = 2023
    N_ENSEMBLE_per_rank= N_ENSEMBLE // size
    fdc_source = 'nhmv10'
    
    ############################################
    ### Generate ensemble of reconstructions ###
    ############################################
    if rank == 0:    
        print(f'Generating ensemble of size {N_ENSEMBLE} with {fdc_source} based FDCs at PUB locations.')

    generate_pywrdrb_reconstruction(start_year= start, 
                                    end_year= end, 
                                    N_REALIZATIONS= N_ENSEMBLE_per_rank, 
                                    K= K_ENSEMBLE, 
                                    inflow_scaling_regression=True,
                                    rank=rank)
    
    comm.Barrier()
    
    
    ### Combine ensemble files
    if rank == 0:
        print('Combining ensemble files...')
        
        catchment_rank_filenames = glob.glob(f'{OUTPUT_DIR}ensembles/catchment_inflow_obs_pub_{fdc_source}_ObsScaled_ensemble_rank*.hdf5')
        gageflow_rank_filenames = glob.glob(f'{OUTPUT_DIR}ensembles/gage_flow_obs_pub_{fdc_source}_ObsScaled_ensemble_rank*.hdf5')


        # load all rank files
        gage_flow_ensemble = {}
        catchment_inflow_ensemble = {}
        realization = 0
        
        for i, catchment_filename in enumerate(catchment_rank_filenames):
            if i%10 == 0:
                print(f'Processing rank {i}...')
                
            gageflow_filename = gageflow_rank_filenames[i]
            
            rank_realization_numbers = get_hdf5_realization_numbers(catchment_filename)
            for rank_realization in rank_realization_numbers:
            
                catchment_inflow_ensemble[realization] = extract_realization_from_hdf5(catchment_filename, 
                                                                                    rank_realization,
                                                                                    stored_by_node=True)
                gage_flow_ensemble[realization] = extract_realization_from_hdf5(gageflow_filename, 
                                                                                rank_realization, 
                                                                                stored_by_node=True)        
                realization += 1
        
        # Reorganize so that they are stored {node:pd.DataFrame(time, realization)}
        n_realizations = realization
        node_catchment_inflow_ensemble = {}
        node_gage_flow_ensemble = {}
        for node in catchment_inflow_ensemble[0].columns:
            node_catchment_inflow_ensemble[node] = pd.concat([catchment_inflow_ensemble[realization][node] for realization in range(n_realizations)], axis=1)
            node_gage_flow_ensemble[node] = pd.concat([gage_flow_ensemble[realization][node] for realization in range(n_realizations)], axis=1)
        
            # relabel columns as string integers
            node_catchment_inflow_ensemble[node].columns = [str(i) for i in range(n_realizations)]
            node_gage_flow_ensemble[node].columns = [str(i) for i in range(n_realizations)]

        # Export to a single hdf5
        print('Exporting combined ensembles to hdf5...')
        output_fname = f'{OUTPUT_DIR}ensembles/catchment_inflow_obs_pub_{fdc_source}_ObsScaled_ensemble.hdf5'            
        export_ensemble_to_hdf5(node_catchment_inflow_ensemble, output_file= output_fname)
        
        output_fname = f'{OUTPUT_DIR}ensembles/gage_flow_obs_pub_{fdc_source}_ObsScaled_ensemble.hdf5'
        export_ensemble_to_hdf5(node_gage_flow_ensemble, output_file= output_fname)

        # Delete rank files
        for catchment_filename in catchment_rank_filenames:
            os.remove(catchment_filename)
        for gageflow_filename in gageflow_rank_filenames:
            os.remove(gageflow_filename)


        print(f'Done generating NHM-based reconstructions (without bias correction)!')
