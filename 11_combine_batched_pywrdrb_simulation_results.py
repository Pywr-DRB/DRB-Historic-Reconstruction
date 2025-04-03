import sys
import os
import glob
import warnings

import pywrdrb
from pywrdrb.utils.hdf5 import combine_batched_hdf5_outputs

# Set the directores for pywrdrb
from config import PYWRDRB_INPUT_DIR, PYWRDRB_OUTPUT_DIR

directories = pywrdrb.get_directory()
model_data_dir = directories.model_data_dir

output_dir = PYWRDRB_OUTPUT_DIR
input_dir = PYWRDRB_INPUT_DIR
pywrdrb.set_directory(input_dir=PYWRDRB_INPUT_DIR)


## Specifications
use_mpi = True
inflow_type = "obs_pub_nhmv10_BC_ObsScaled_ensemble"

batched_filenames = glob.glob(
    f"{output_dir}drb_output_{inflow_type}_rank*_batch*.hdf5"
)
batched_modelnames = glob.glob(
    f"{model_data_dir}drb_model_full_{inflow_type}_rank*.json"
)


### Combine batched output files
try:
    print(f"Combining {len(batched_filenames)} pywrdrb output files...")
    combined_output_filename = f"{output_dir}drb_output_{inflow_type}.hdf5"
    combine_batched_hdf5_outputs(
        batch_files=batched_filenames, combined_output_file=combined_output_filename
    )

    # Delete batched files
    print("Deleting individual batch results files")
    for file in batched_filenames:
        os.remove(file)
    for file in batched_modelnames:
        os.remove(file)
except Exception as e:
    warnings.warn(f"Error combining batched files: {e}")
    pass
