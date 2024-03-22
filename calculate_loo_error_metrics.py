"""
Gathers leave-one-out datasets and calculates different metrics. 

"""

from mpi4py import MPI
import numpy as np
import pandas as pd

from methods.processing.load import load_model_segment_flows
from methods.processing.load import load_leave_one_out_datasets
from methods.diagnostics import get_error_summary
from methods.utils.directories import OUTPUT_DIR

from run_leave_one_out_experiment_parallel import AGG_K_MAX, AGG_K_MIN
from run_leave_one_out_experiment_parallel import ENSEMBLE_K_MAX, ENSEMBLE_K_MIN

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


### Different filenames and models
loo_filenames = []
loo_models = []
ensemble_methods = [1,2]

for nxm in ['nhmv10', 'nwmv21']:
    
    ## Different QPPQ aggregate models
    for k in range(AGG_K_MIN, AGG_K_MAX):
        loo_models.append(f'obs_pub_{nxm}_K{k}')
        
        fname = f'{OUTPUT_DIR}/LOO/loo_reconstruction_{nxm}_K{k}.csv'
        loo_filenames.append(fname)
        
    ## Different QPPQ ensemble models
    for k in range(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX):
        for e in ensemble_methods:
            loo_models.append(f'obs_pub_{nxm}_K{k}_ensemble_m{e}')

            fname = f'{OUTPUT_DIR}/LOO/method_{e}/loo_reconstruction_{nxm}_K{k}_ensemble_m{e}.hdf5'
            loo_filenames.append(fname)
            
### Load leave-one-out datasets
Q = load_leave_one_out_datasets(loo_filenames, loo_models)


## NHM and NWM model outputs
Q['nhmv10'] = load_model_segment_flows('nhmv10', station_number_columns=True)
Q['nwmv21'] = load_model_segment_flows('nwmv21', station_number_columns=True)


# Get list of sites from results
# These may be different for NWM and NHM
model_loo_sites = {}
for m in ['nhmv10', 'nwmv21']:
    # Use a model spec from the list that matches
    nxm_model = [i for i in loo_models if m in i]
    if len(nxm_model) == 0:
        continue
    nxm_model = nxm_model[0]
    if 'ensemble' in nxm_model:
        model_loo_sites[m] = list(Q[nxm_model].keys())
    else:
        model_loo_sites[m] = Q[nxm_model].columns.to_list()

# Get union of loo sites
loo_sites = list(set(model_loo_sites['nhmv10']).union(set(model_loo_sites['nwmv21'])))

# Keep only observations at loo sites
Q['obs'] = Q['obs'].loc[:,loo_sites]
Q['nhmv10'] = Q['nhmv10'].loc[:,loo_sites]
Q['nwmv21'] = Q['nwmv21'].loc[:,loo_sites]



####################################
### Get Error Metrics ###
####################################

### Parallelize
if rank == 0:
    # Root process splits loo_sites and distributes them
    chunks = np.array_split(loo_sites, size)
else:
    chunks = None

# Scatter the chunks of loo_sites to all processes
loo_sites_chunk = comm.scatter(chunks, root=0)

# Take only the sites that are in the chunk
Q_chunk = {}
for m in loo_models:
    if 'ensemble' in m:
        for s in loo_sites_chunk:
            Q_chunk[m] = Q[m][s]
    else:
        Q_chunk[m] = Q[m].loc[:,loo_sites_chunk]
    
# Each process executes get_error_summary on its chunk
error_summary_chunk = get_error_summary(Q_chunk, loo_models, loo_sites_chunk, 
                                        by_year=False, 
                                        start_date='1945-01-01', end_date='2016-12-31')

# Gather all chunks back at the root process
comm.barrier()
error_summary_gathered = comm.gather(error_summary_chunk, root=0)

# Root process concatenates and saves the results
if rank == 0:
    error_summary = pd.concat(error_summary_gathered)
    error_summary.reset_index(inplace=True)
    error_summary.to_csv(f'{OUTPUT_DIR}/LOO/loo_error_summary.csv')

## Repeat for annual error summary
print('GETTING ANNUAL ERROR SUMMARY')
error_summary_annual = get_error_summary(Q_chunk, loo_models, 
                                         loo_sites_chunk, 
                                         by_year=True, 
                                         start_date='1945-01-01', end_date='2016-12-31')

# Gather all chunks back at the root process
comm.barrier()
error_summary_annual_gathered = comm.gather(error_summary_annual, root=0)

if rank == 0:
    error_summary_annual = pd.concat(error_summary_annual_gathered)
    error_summary_annual.reset_index(inplace=True)
    error_summary_annual.to_csv(f'{OUTPUT_DIR}/LOO/loo_error_summary_annual.csv')

print('DONE')