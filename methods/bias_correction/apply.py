import os
import pandas as pd
import numpy as np

from config import FDC_QUANTILES
from methods.load.data_loader import Data

from methods.processing.hdf5 import extract_loo_results_from_hdf5
from methods.processing.transform import streamflow_to_nonexceedance, nonexceedance_to_streamflow

from methods.utils.directories import DATA_DIR, OUTPUT_DIR

from methods.bias_correction.utils import filter_biases
from methods.bias_correction.utils import load_posterior_samples_from_hdf5
from methods.bias_correction.utils import export_posterior_samples_to_hdf5

fdc_quantiles = FDC_QUANTILES

def apply_bias_correction_to_ensemble(nxm, 
                                      K, 
                                      OUTPUT_DIR=OUTPUT_DIR,
                                      save_filtered_biases=True):
    
    
    

    ## Load un-corrected LOO samples
    fname= f'{OUTPUT_DIR}/LOO/loo_reconstruction_{nxm}_K{K}_ensemble.hdf5'
    loo_predictions = extract_loo_results_from_hdf5(fname)
    loo_sites = list(loo_predictions.keys())
    realizations = list(loo_predictions[loo_sites[0]].columns)
    
    
    ### Get bias corrected fdcs
    
    # If filtered bias samples and corrected fdcs exist, load them
    filtered_bias_pred_file = f'{OUTPUT_DIR}/LOO/loo_bias_correction/{nxm}_bias_posterior_samples_filtered.hdf5'
    corrected_fdc_file = f'{OUTPUT_DIR}/LOO/loo_bias_correction/{nxm}_fdc_bias_corrected_samples.hdf5'
    if os.path.exists(filtered_bias_pred_file) and os.path.exists(corrected_fdc_file):            
            bias_adjusted = load_posterior_samples_from_hdf5(filtered_bias_pred_file, return_type='dict')
            fdc_corrected = load_posterior_samples_from_hdf5(corrected_fdc_file, return_type='dict')
        
    
    # if filtered bias samples do not exist, load all bias samples and filter
    # and apply bias correction to FDCs
    # save at the end to avoid re-doing this
    else:
        # Load bias posterior sample predictions
        fname = f'{OUTPUT_DIR}/LOO/loo_bias_correction/{nxm}_bias_posterior_samples.hdf5'
        bias_samples_all = load_posterior_samples_from_hdf5(fname, return_type='dict')
    
        # Get FDCs
        fdc = {}
        data_loader = Data()
        fdc[nxm] = data_loader.load(datatype='fdc', sitetype='usgs', 
                                    flowtype="nhm", timescale='daily')
        fdc[nxm] = fdc[nxm].loc[loo_sites, :]
        
        # Take only bias samples for loo sites    
        bias_samples = {}
        for site in loo_sites:
            bias_samples[site] = bias_samples_all[site]
        
            
        # Filter bias samples
        n_posterior_samples = 1000
        bias_adjusted, fdc_corrected = filter_biases(fdc[nxm],
                                                    bias_samples,
                                                    N_samples=n_posterior_samples,
                                                    max_sample_iter=1000,
                                                    return_corrected=True)
    
        # Export filtered bias samples
        bias_adjusted_array = np.zeros((n_posterior_samples, len(loo_sites), len(fdc_quantiles)))
        for si, site in enumerate(loo_sites):
            bias_adjusted_array[:, si, :] = bias_adjusted[site]
        fname = f'{OUTPUT_DIR}/LOO/loo_bias_correction/{nxm}_bias_posterior_samples_filtered.hdf5'
        export_posterior_samples_to_hdf5(bias_adjusted_array, loo_sites, fname)
    
        # Export bias corrected FDCs
        y_corrected_array = np.zeros((n_posterior_samples, len(loo_sites), len(fdc_quantiles)))
        for si, site in enumerate(loo_sites):
            y_corrected_array[:, si, :] = fdc_corrected[site]
        fname = f'{OUTPUT_DIR}/LOO/loo_bias_correction/{nxm}_fdc_bias_corrected_samples.hdf5'
        export_posterior_samples_to_hdf5(y_corrected_array, loo_sites, fname)
    
    bias_corrected_ensemble = {}
    for si, site in enumerate(loo_sites):
        
        bias_corrected_ensemble[site] = pd.DataFrame(index=loo_predictions[site].index, 
                                                     columns=realizations)

        for ri, realization in enumerate(realizations):
            Q_uncorrected = loo_predictions[site][realization].values.astype(float)
            
            nep_timeseries = streamflow_to_nonexceedance(Q_uncorrected, 
                                                         fdc_quantiles, 
                                                         log_fdc_interpolation=True)
            
            Q_corrected = nonexceedance_to_streamflow(nep_timeseries,
                                                        fdc_quantiles,
                                                        fdc_corrected[site][ri, :],
                                                        log_fdc_interpolation=True)

            bias_corrected_ensemble[site].loc[:, realization] = Q_corrected        

    return bias_corrected_ensemble