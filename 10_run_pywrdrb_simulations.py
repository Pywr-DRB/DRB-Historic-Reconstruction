"""

"""
import glob
import numpy as np
import sys
import os
import math
import time
import shutil
from pywr.model import Model
from pywr.recorders import TablesRecorder


from pywrdrb import ModelBuilder

import pywrdrb.parameters.general
import pywrdrb.parameters.ffmp
import pywrdrb.parameters.starfit
import pywrdrb.parameters.lower_basin_ffmp

from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.load.get_results import get_keys_and_column_names_for_results_set
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts

def get_parameter_subset_to_export(all_parameter_names):
    keep_keys = []
    for results_set in pywrdrb_results_set_opts:
        if results_set == "all":
            continue
        keys_subset, _ = get_keys_and_column_names_for_results_set(all_parameter_names, results_set)
        
        keep_keys.extend(keys_subset)
    return keep_keys


from mpi4py import MPI

t0 = time.time()
from config import SEED
from config import START_YEAR, END_YEAR

# Set the directores for pywrdrb
from config import PYWRDRB_INPUT_DIR, PYWRDRB_OUTPUT_DIR, OUTPUT_DIR

directories = pywrdrb.get_directory()
model_data_dir = directories.model_data_dir

output_dir = PYWRDRB_OUTPUT_DIR
input_dir = PYWRDRB_INPUT_DIR
pywrdrb.set_directory(input_dir=PYWRDRB_INPUT_DIR)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(SEED)

inflow_type = "obs_pub_nhmv10_BC_ObsScaled_ensemble"

## Copy input files into /venv/lib/python3.11/site-packages/pywrdrb/input_data/ 
# to be used by pywrdrb
# rf"{input_dir}/predicted_inflows_diversions_{inflow_type}.hdf5",

ensemble_input_filename = f"{input_dir}/historic_ensembles/catchment_inflow_{inflow_type}.hdf5"

files_to_copy = [
    ensemble_input_filename,
    rf"{input_dir}/predicted_inflows_diversions_{inflow_type}.hdf5",
    rf"{input_dir}/deliveryNJ_DRCanal_extrapolated.csv",
    rf"{input_dir}/deliveryNJ_DRCanal_extrapolated.csv",
    rf"{input_dir}/sw_avg_wateruse_Pywr-DRB_Catchments.csv",
]

## Specifications
use_mpi = True

start_date = f"{START_YEAR}-01-01"
end_date = f"2022-12-31"

# number of inflow scenarios to run per sim using pywr parallel
batch_size = 20

if __name__ == "__main__":
    
    ## Clear old batched output files if they exist
    if rank == 0:
        batched_filenames = glob.glob(
            f"{output_dir}drb_output_{inflow_type}_rank*_batch*.hdf5"
        )
        batched_modelnames = glob.glob(
            f"{model_data_dir}drb_model_full_{inflow_type}_rank*.json"
        )


        ### Combine batched output files
        if len(batched_filenames) > 0:
            for file in batched_filenames:
                os.remove(file)
        if len(batched_modelnames) > 0:
            for file in batched_modelnames:
                os.remove(file)
    
    
    
    if rank == 0:
        for file in files_to_copy:
            # get current directory
            cwd = os.getcwd()
            
            dst = f"{cwd}/venv/lib/python3.11/site-packages/pywrdrb/input_data/"
            
            if "historic_ensembles" in file:
                dst = f"{dst}/historic_ensembles/"
            
            # if the directory does not exist, create it
            if not os.path.exists(dst):
                os.makedirs(dst)
            
            print(f"Copying {file} to {dst}")
            shutil.copy(file, dst)
    comm.Barrier()
    
    ### Setup simulation batches 
    # Get the IDs for the realizations
    
    if rank == 0:
        realization_ids = get_hdf5_realization_numbers(ensemble_input_filename)
        
    else:
        realization_ids = None
    realization_ids = comm.bcast(realization_ids, root=0)

    # Split the realizations into batches
    rank_realization_ids = np.array_split(realization_ids, size)[rank]
    n_rank_realizations = len(rank_realization_ids)

    # Split the realizations into batches
    n_batches = math.ceil(n_rank_realizations / batch_size)
    batched_indices = {
        i: rank_realization_ids[
            (i * batch_size) : min((i * batch_size + batch_size), n_rank_realizations)
        ]
        for i in range(n_batches)
    }
    batched_filenames = []

    # Run individual batches
    for batch, indices in batched_indices.items():
        
        indices = indices.tolist()
        if len(indices) == 0:
            continue
        
        print(f"Rank {rank} setting up simulation batch {batch+1} of {n_batches} with inflow scenarios {indices}")
        sys.stdout.flush()


        model_filename = (f"{model_data_dir}drb_model_full_{inflow_type}_rank{rank}.json")
        output_filename = (f"{output_dir}drb_output_{inflow_type}_rank{rank}_batch{batch}.hdf5")

        batched_filenames.append(output_filename)

        ### make model json files
        options = {"inflow_ensemble_indices": indices}

        mb = ModelBuilder(inflow_type, 
                        start_date, 
                        end_date, 
                        options=options) 
        mb.make_model()
        mb.write_model(model_filename)

        ### Load the model
        model = pywrdrb.Model.load(model_filename)

        all_parameter_names = [p.name for p in model.parameters if p.name]
        export_parameters = get_parameter_subset_to_export(all_parameter_names)

        ### Add a storage recorder
        TablesRecorder(model, 
                    output_filename, 
                    parameters=[p for p in model.parameters if p.name in export_parameters])

        ### Run the model
        print(rf"Rank {rank} running pywrdrb simulation...")
        stats = model.run()

        print(f"Rank {rank} batch {batch+1} of {n_batches} complete")

    if rank == 0:
        print(f"Rank {rank}: all pywrdrb simulations complete!")