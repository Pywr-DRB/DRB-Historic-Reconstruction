#!/bin/bash
#SBATCH --job-name=PUB
#SBATCH --output=logs/run_main.out
#SBATCH --error=logs/run_main.err
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=30
#SBATCH --exclusive


module load python/3.11.5
source venv/bin/activate

module load gnu9/9.3.0
export PYTENSOR_FLAGS="gcc__cxxflags='-I/opt/ohpc/pub/compiler/gcc/9.3.0/include'"
PYMC_CACHE_DIR=/home/fs02/pmr82_0001/tja73/.pytensor/compiledir_Linux-4.18-el8_4.x86_64-x86_64-with-glibc2.28-x86_64-3.11.5-64/

# Number of processors
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

#################################################################
### Data retrieval & preprocessing
#################################################################

# Copy NHMv1.0 streamflow data from Input-Data-Retrieval repo into DRB-Historic-Reconstruction/data
cd ..
cp -r ./Input-Data-Retrieval/datasets/NHMv10/* ./DRB-Historic-Reconstruction/data/NHMv10/
cd DRB-Historic-Reconstruction


echo "Retrieving USGS, NID, catchment geometry, Daymet and NLDI data for DRB gauges..."
time mpirun -np 20 python -u 01_get_source_data.py


echo "Preprocessing data..."
time mpirun -np 1 python -u 02_preprocess_data.py


echo "Preparing bias correciton input data..."
time mpirun -np 1 python -u 03_prep_bias_correction_inputs.py

#################################################################
### LEAVE ONE OUT AT DIAGNOSTIC SITES ###########################
#################################################################

# # # Clear pymc cache
rm -R ${PYMC_CACHE_DIR}/tmp*

echo "Running $nxm BART bias correction LEAVE-ONE-OUT MCMC..."
time mpirun -n $np python3 -u -W ignore 04_leave_one_out_bias_inference.py > pymc_status.out

echo "Combining leave-one-out bias samples..."
time mpirun -n 1 python3 -u -W ignore combine_loo_samples.py


echo "Running full bias-corrected QPPQ LEAVE-ONE-OUT prediction..."
time mpirun -n $np python3 -u -W ignore 05_leave_one_out_reconstructions.py

echo "Calculating errors for LEAVE-ONE-OUT prediction..."
time mpirun -n $np python3 -u -W ignore 12_calculate_leave_one_out_errors.py


#################################################################
### GENERATE RECONSTRUCTIONS AT PYWRDRB NODES ###################
#################################################################

echo "Generating reconstructions with default NHM FDCs..."
time mpirun -np $np python3 05_generate_non_bias_corrected_pywrdrb_reconstructions.py

### Bias prediction prediction at PywrDRB nodes
echo "Preparing bias posterior samples at PywrDRB nodes..."
time mpirun -np 1 python3 07_predict_bias_at_pywrdrb_nodes.py

### Generate Pywr-DRB reconstructions with bias correction
echo "Generating nhm bias-corrected ensembles at PywrDRB nodes..."
time python3 -u 08_generate_bias_corrected_pywrdrb_reconstructions.py

### Copy reconstruction files to pywrdrb_inputs folder
echo "Copying reconstruction files to pywrdrb_inputs repo..."
cp ./outputs/ensembles/catchment_inflow_obs_pub_nhmv10_BC_ObsScaled_ensemble.hdf5 ./pywrdrb_inputs/historic_ensembles/

#################################################################
### Prepare PywrDRB input data ##################################
#################################################################

# In addition to the inflow ensemble, pywrdrb requires:
# 1. Predicted future flows, to determine additional releasses
# 2. Extrapolated diversions to match the inflow timeseries

### Generate Pywr-DRB reconstructions with bias correction
echo "Preparing input data for PywrDRB..."
time mpirun -np $np python3 -u 09_prep_pywrdrb_input_data.py

### Run PywrDRB simulations with ensemble
echo "Running PywrDRB simulations..."
time mpirun -np $np python3 -u -W ignore 10_run_pywrdrb_simulations.py

### Combine simulation outputs from parallel jobs
time python3 -u 11_combine_batched_pywrdrb_simulation_results.py


#################################################################
### Plotting ####################################################
#################################################################

## Plotting
time python3 -u -W ignore 13_make_data_summary_plot.py
time python3 -u -W ignore 14_make_diagnostic_error_plot.py
time python3 -u -W ignore 15_make_drought_plot.py
time python3 -u -W ignore 16_make_flow_contribution_plot.py
time python3 -u -W ignore 17_make_full_period_plot.py

echo "Done!"