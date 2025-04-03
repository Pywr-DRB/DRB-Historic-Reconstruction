import sys
import shutil
import numpy as np
import pandas as pd


from methods.load.data_loader import Data

from methods.bias_correction.utils import export_posterior_samples_to_hdf5, load_posterior_samples_from_hdf5
from methods.bias_correction.utils import filter_biases
from methods.processing.transform import streamflow_to_nonexceedance, nonexceedance_to_streamflow
from methods.processing.add_subtract_upstream_flows import subtract_upstream_catchment_inflows
from methods.processing.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5, export_ensemble_to_hdf5
from methods.utils.directories import OUTPUT_DIR, PYWRDRB_DIR, DATA_DIR

from pywrdrb_node_data import obs_pub_site_matches

from config import FDC_QUANTILES

nxm = 'nhmv10'
dataset_name = f'obs_pub_{nxm}_BC_ObsScaled'



### Load
data_loader = Data()


## Prediction location metadata
gauge_meta = data_loader.load(datatype='metadata', sitetype='pywrdrb')
pywrdrb_comids = gauge_meta['comid'].values
pywrdrb_nodes = gauge_meta.index.values
pywrdrb_pub_nodes = [n for n in pywrdrb_nodes if obs_pub_site_matches[n] is None]

# NHM fdcs
fdc = {}
fdc[nxm] = data_loader.load(datatype='fdc', 
                            flowtype='nhm', 
                            sitetype='pywrdrb',
                            timescale='daily')


## Load reconstruction ensembles
gage_flow_ensemble_file = f'{OUTPUT_DIR}ensembles/gage_flow_obs_pub_{nxm}_ObsScaled_ensemble.hdf5'
catchment_inflow_ensemble_file = f'{OUTPUT_DIR}ensembles/catchment_inflow_obs_pub_{nxm}_ObsScaled_ensemble.hdf5'

realization_numbers = get_hdf5_realization_numbers(gage_flow_ensemble_file)

print(f'Loading {len(realization_numbers)} realizations of obs_pub_{nxm} flows...')
gage_flow_ensemble = {}
catchment_inflow_ensemble = {}
for realization in realization_numbers:
    gage_flow_ensemble[realization] = extract_realization_from_hdf5(gage_flow_ensemble_file, realization, 
                                                          stored_by_node=True)
    catchment_inflow_ensemble[realization] = extract_realization_from_hdf5(catchment_inflow_ensemble_file, 
                                                                           realization,
                                                                           stored_by_node=True)
    
print(f'Successfully loaded original {nxm} gage flow and catchment inflow ensembles.')


### Load bias estimate samples
output_fname = f'{OUTPUT_DIR}/{nxm}_pywrdrb_bias_posterior_samples.hdf5'
y_pred_load = load_posterior_samples_from_hdf5(output_fname, return_type='dict')

print('Successfully loaded bias estimate samples')


# Filter posterior bias samples to 
# ensure monotonicity and non-negativity of final FDC
n_filtered_bias_samples = len(realization_numbers)
bias_adjusted, y_corrected = filter_biases(fdc[nxm], 
                              y_pred_load, 
                              N_samples=n_filtered_bias_samples, 
                              max_sample_iter=1000,
                              return_corrected=True)

print(f'Filtered bias samples to ensure corrected FDC monotonicity and non-negativity.')

## Export filtered biases to hdf5
# re-arrange so bias_adjusted is a 3d np.array
bias_adjusted_array = np.empty((n_filtered_bias_samples, len(pywrdrb_nodes), 18))
fdc_corrected_array = np.empty((n_filtered_bias_samples, len(pywrdrb_nodes), 18))
for node in bias_adjusted.keys():
    node_idx = np.where(pywrdrb_nodes == node)[0][0]
    bias_adjusted_array[:, node_idx, :] = bias_adjusted[node]
    fdc_corrected_array[:, node_idx, :] = y_corrected[node]


# check if na
if np.isnan(bias_adjusted_array).sum() > 0:
    print(f'WARNING: NaNs in bias_adjusted_array, count: {np.isnan(bias_adjusted_array).sum()}')
elif np.isnan(fdc_corrected_array).sum() > 0:
    print(f'WARNING: NaNs in fdc_corrected_array, count: {np.isnan(fdc_corrected_array).sum()}')
    
output_fname = f'{OUTPUT_DIR}{nxm}_pywrdrb_bias_posterior_samples_filtered.hdf5'
export_posterior_samples_to_hdf5(bias_adjusted_array, pywrdrb_nodes, output_fname)

output_fname = f'{OUTPUT_DIR}{nxm}_pywrdrb_bias_corrected_fdc.hdf5'
export_posterior_samples_to_hdf5(fdc_corrected_array, pywrdrb_nodes, output_fname)

print(f'Saved filtered bias samples and ensemble of corrected FDCs to {OUTPUT_DIR}')

####################################################
### Apply bias correction to ensemble timeseries ###
####################################################

print('Applying bias correction to PywrDRB ensemble flows...')

# Go from streamflow to non-exceedance using original FDC 
# Then go from non-exceedance to streamflow using bias corrected FDC
bias_corrected_gage_flow = {}

for ri, realization in enumerate(realization_numbers):
    bias_corrected_gage_flow[realization] = pd.DataFrame(index=gage_flow_ensemble[realization].index, 
                                                        columns=pywrdrb_nodes)
    
    for node in pywrdrb_nodes:
        if node not in pywrdrb_pub_nodes:
            bias_corrected_gage_flow[realization][node] = gage_flow_ensemble[realization][node]
            
        # Trenton doesn't have direct flow
        elif node == 'delTrenton':
            bias_corrected_gage_flow[realization][node] = gage_flow_ensemble[realization][node]*0.0
            
        else:
            
            Q_realization = gage_flow_ensemble[realization].drop(['datetime'], axis=1)
            Q_realization = Q_realization.loc[:, node].values.astype(float)
            
            nep_realization = streamflow_to_nonexceedance(Q_realization, 
                                                          FDC_QUANTILES, 
                                                          log_fdc_interpolation=True)
            
            Q_corrected = nonexceedance_to_streamflow(nep_realization, FDC_QUANTILES, 
                                                    y_corrected[node][ri, :], 
                                                    log_fdc_interpolation=True)
            
            bias_corrected_gage_flow[realization][node] = Q_corrected.copy()

### Calculate catchment inflows by removing upstream catchment flows
bias_corrected_catchment_inflow = {}
for ri, realization in enumerate(realization_numbers):
    bias_corrected_catchment_inflow[realization] = pd.DataFrame(index=catchment_inflow_ensemble[realization].index, 
                                                        columns=pywrdrb_nodes)
    
    Q_gage_flow_realization = bias_corrected_gage_flow[realization]
    
    # subtract upstream catchment inflows
    Q_catchment_inflow_realization = subtract_upstream_catchment_inflows(Q_gage_flow_realization)

    for node in pywrdrb_nodes:
        if node not in pywrdrb_pub_nodes:
            bias_corrected_catchment_inflow[realization][node] = catchment_inflow_ensemble[realization][node]
        else:
            bias_corrected_catchment_inflow[realization][node] = Q_catchment_inflow_realization[node].copy()


## Re-organize so that nodes are keys
node_bias_corrected_gage_flow = {}
node_bias_corrected_catchment_inflow = {}
for node in pywrdrb_nodes:
    node_bias_corrected_gage_flow[node] = pd.concat([bias_corrected_gage_flow[ri][node] for ri in realization_numbers], axis=1)
    node_bias_corrected_gage_flow[node].columns = [str(r) for r in realization_numbers]
    node_bias_corrected_gage_flow[node].index = gage_flow_ensemble[realization_numbers[0]]['datetime']
    
    node_bias_corrected_catchment_inflow[node] = pd.concat([bias_corrected_catchment_inflow[ri][node] for ri in realization_numbers], axis=1)
    node_bias_corrected_catchment_inflow[node].columns = [str(r) for r in realization_numbers]
    node_bias_corrected_catchment_inflow[node].index = catchment_inflow_ensemble[realization_numbers[0]]['datetime']
        
# Export corrected ensembles to hdf5
print('Exporting bias corrected PywrDRB ensemble files to hdf5...')

output_fname = f'{OUTPUT_DIR}ensembles/gage_flow_obs_pub_{nxm}_BC_ObsScaled_ensemble.hdf5'
export_ensemble_to_hdf5(node_bias_corrected_gage_flow, output_fname)

output_fname = f'{OUTPUT_DIR}ensembles/catchment_inflow_obs_pub_{nxm}_BC_ObsScaled_ensemble.hdf5'
export_ensemble_to_hdf5(node_bias_corrected_catchment_inflow, output_fname)


print('### Done with bias correction of PywrDRB ensemble flows! ###')
