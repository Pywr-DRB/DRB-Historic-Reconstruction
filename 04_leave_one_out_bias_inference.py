from mpi4py import MPI
import subprocess
import os
import numpy as np

from config import BART_USE_X_SCALED, SEED
from methods.bias_correction.prep import load_bias_correction_inputs, load_bias_correction_training_outputs
from methods.bias_correction.utils import combine_leave_one_out_bias_samples

prediction_script = 'make_posterior_predictions.py'

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Must be set to make sure array_split works correctly
np.random.seed(SEED)

if __name__ == '__main__':
    
    ### Load data
    x_train_full = load_bias_correction_inputs(scaled=BART_USE_X_SCALED, 
                                            pywrdrb=False)
    y = load_bias_correction_training_outputs(percent_bias=True)

    y = y.loc[x_train_full.index, :]
    training_sites = y.index

    # Split site indices among ranks
    n_sites = len(training_sites)
    if rank == 0:
        rank_site_numbers = np.array_split(np.array(training_sites), size)
    else:
        rank_site_numbers = None
        
    # Broadcast the split site indices to all ranks
    rank_site_numbers = comm.bcast(rank_site_numbers, root=0)

    # Get the sites for this rank
    test_site_numbers = rank_site_numbers[rank]

    # Write the site indices to a file for the subprocess to read
    site_file = f'/tmp/sites_rank_{rank}.txt'
    with open(site_file, 'w') as f:
        for site in test_site_numbers:
            f.write(f"{site}\n")

    # Launch the subprocess
    try:
        # if rank has no test_site_numbers, skip the subprocess
        if len(test_site_numbers) == 0:
            result = None
        else:
            result = subprocess.run(['python3', prediction_script, site_file, str(rank)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess for rank {rank} failed with error: {e}")
        raise

    # Optional: Cleanup the site file
    # os.remove(site_file)
    print(f'DONE: Rank {rank} completed successfully')
    