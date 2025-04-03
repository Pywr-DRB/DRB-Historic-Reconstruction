import numpy as np
import pandas as pd
import os
from mpi4py import MPI

from config import FIG_DIR, OUTPUT_DIR

from methods.load.data_loader import Data
from methods.processing.load import load_leave_one_out_datasets
from methods.diagnostics import get_error_summary
from methods.diagnostics.metrics import get_leave_one_out_filenames, _file_exists

from config import AGG_K_MAX, AGG_K_MIN
from config import ENSEMBLE_K_MAX, ENSEMBLE_K_MIN
from config import SEED

np.random.seed(SEED)

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



if __name__ == "__main__":
    
    ##################################
    ### Loading ######################
    ##################################
    
    if rank == 0:
        
        data_loader = Data()
        
        ### Streamflow
        # Load observed and NHM streamflow
        Q_nhm = data_loader.load(datatype='streamflow', 
                                    flowtype='nhm', sitetype='diagnostic')
        Q_obs = data_loader.load(datatype='streamflow', 
                                    flowtype='obs', sitetype='diagnostic')
        Q_obs = Q_obs.loc["1983-10-01":"2016-12-31", :]

        print(f"Q obs loaded with shape {Q_obs.shape}")

        # Get list of leave-one-out ensemble filenames
        loo_filenames, loo_models = get_leave_one_out_filenames()
        loo_sites = list(Q_obs.columns)

        # Split across rank
        split_indices = np.array_split(np.arange(len(loo_filenames)), size)

        
    else:
        Q_obs = None
        split_indices = None
        loo_filenames = None
        loo_models = None
        loo_sites = None
        
        

    # Broadcast
    Q_obs = comm.bcast(Q_obs, root=0)
    split_indices = comm.bcast(split_indices, root=0)
    loo_filenames = comm.bcast(loo_filenames, root=0)
    loo_models = comm.bcast(loo_models, root=0)
    loo_sites = comm.bcast(loo_sites, root=0)
    
    # Get filenames for leave-one-out datasets to be processed by this rank
    rank_indices = split_indices[rank]
    rank_loo_filenames = [loo_filenames[i] for i in rank_indices]
    rank_loo_models = [loo_models[i] for i in rank_indices]
    
    # Load leave-one-out datasets for this rank
    Q = load_leave_one_out_datasets(rank_loo_filenames, rank_loo_models)

    # Add Q_obs
    Q['obs'] = Q_obs.loc[:,loo_sites]
    
    # Only rank 0 will handle NHM
    if rank == 0:
        Q['nhmv10'] = Q_nhm.loc[:,loo_sites]
        rank_loo_models += ['nhmv10']

    if len(rank_loo_models) > 0:
        print(f'Rank {rank} successfully loaded leave-one-out datasets: {rank_loo_models}')

    ##################################
    ### Get errors ###################
    ##################################
        
    # Each process executes get_error_summary on its chunk
    error_summary_chunk = get_error_summary(Q, rank_loo_models, 
                                            loo_sites, 
                                            by_year=False)

    # Gather all chunks back at the root process
    error_summary_gathered = comm.gather(error_summary_chunk, root=0)

    
    # Root process concatenates and saves the results
    comm.barrier()
    if rank == 0:
        error_summary = pd.concat(error_summary_gathered)
        error_summary.reset_index(inplace=True)
        error_summary.to_csv(f'{OUTPUT_DIR}/LOO/loo_error_summary.csv')

        print('### DONE WITH LEAVE-ONE-OUT ERROR CALCULATIONS ###')