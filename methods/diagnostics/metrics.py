"""
Contains functions for calculating various hydrologic metrics.
"""

import numpy as np
import pandas as pd
import hydroeval as he
import os

from config import OUTPUT_DIR, AGG_K_MIN, AGG_K_MAX, ENSEMBLE_K_MIN, ENSEMBLE_K_MAX

error_metrics = ['nse', 'kge', 'r', 'alpha', 'beta',
                'log_nse', 'log_kge', 'log_r', 'log_alpha', 'log_beta', 
                'Q0.1pbias', 'Q0.2pbias', 'Q0.3pbias']


def _file_exists(fname):
    return os.path.exists(fname)

def get_leave_one_out_filenames():
    loo_filenames = []
    loo_models = []

    ### Get lists of filenames
    for nxm in ['nhmv10']:
        
        ## Different QPPQ aggregate models
        for k in range(AGG_K_MIN, AGG_K_MAX):
            fname = f'{OUTPUT_DIR}/LOO/loo_reconstruction_{nxm}_K{k}.csv'
            if _file_exists(fname):
                loo_filenames.append(fname)
                loo_models.append(f'obs_pub_{nxm}_K{k}')
            
        ## Different QPPQ ensemble models
        for k in range(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX):
            
            # Bias correction options
            for bc in ['', '_BC']:

                fname = f'{OUTPUT_DIR}/LOO/loo_reconstruction_{nxm}{bc}_K{k}_ensemble.hdf5'
                if _file_exists(fname):
                    loo_filenames.append(fname)
                    loo_models.append(f'obs_pub_{nxm}{bc}_K{k}_ensemble')
    return loo_filenames, loo_models


def get_errors(Q_obs, Q_sim, bias=False):
    
    errors = {}
    errors['nse'] = he.evaluator(he.nse, Q_sim, Q_obs)
    errors['kge'], errors['r'], errors['alpha'], errors['beta'] = he.evaluator(he.kge, Q_sim, Q_obs)
    errors['rmse'] = he.evaluator(he.rmse, Q_sim, Q_obs)
    errors['mare'] = he.evaluator(he.mare, Q_sim, Q_obs)
    errors['pbias'] = he.evaluator(he.pbias, Q_sim, Q_obs)
    
    errors['log_nse'] = he.evaluator(he.nse, Q_sim, Q_obs, transform='log')
    errors['log_kge'], errors['log_r'], errors['log_alpha'], errors['log_beta'] = he.evaluator(he.kge, Q_sim, Q_obs, transform='log')
    errors['log_rmse'] = he.evaluator(he.rmse, Q_sim, Q_obs, transform='log')
    errors['log_mare'] = he.evaluator(he.mare, Q_sim, Q_obs, transform='log')
    if bias:
        for q_n in [0.1, 0.2, 0.3]:
            Q_obs_low = Q_obs.loc[Q_obs < Q_obs.quantile(q_n)]
            Q_sim_low = Q_sim.loc[Q_obs < Q_obs.quantile(q_n)]
                
            # Get percent bias
            errors[f'Q{q_n}pbias'] = he.evaluator(he.pbias, Q_sim_low, Q_obs_low)
    return errors
    



def get_error_summary(data, 
                      models, 
                      sites,
                      by_year = False):
    
    summary_columns = ['model', 'site', 'realization', 'metric', 'value']
    if by_year:
        summary_columns= summary_columns + ['year']
    
    # Initialize dataframe to store all data
    error_summary = pd.DataFrame(columns=summary_columns)
    
    for model in models:
        print(f'Getting errors for {model} dataset')
        subset_start_date = '1983-10-01'
        subset_end_date = '2016-12-31'
        
        if 'ensemble' in model:
            for si, site_number in enumerate(sites):
                n_realizations= data[model][site_number].shape[1]
                
                # Get errors for every realization (and year if by_year)
                if by_year:
                    year_min = Q_obs.index.min().year
                    year_max = Q_obs.index.max().year
                    for yr in range(year_min, year_max):
                        Q_obs_yr = Q_obs.loc[Q_obs.index.year == yr].dropna()
                        Q_sim_yr = data[model][site_number].loc[Q_obs_yr.index, :]
                        assert(len(Q_obs_yr) == len(Q_sim_yr)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'

                        model_errors = get_errors(Q_obs_yr, Q_sim_yr, bias=True)
                        
                        # Reformat to meet error_summary format
                        model_errors_summary = []
                        for ri, real in enumerate(range(n_realizations)):
                            for k in model_errors.keys():
                                summary_row = [model, site_number, ri, k, model_errors[k][ri], yr]
                                model_errors_summary.append(summary_row)
                        model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                        error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                        
                else:
                    Q_obs = data['obs'].loc[subset_start_date:subset_end_date, site_number].dropna()
                    Q_sim = data[model][site_number].loc[Q_obs.index, :]
                    assert(len(Q_obs) == len(Q_sim)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'
                    
                    # Get all error metrics 
                    mod_errors = get_errors(Q_obs, Q_sim, bias=True)
                    
                    # Reformat to meet error_summary format
                    model_errors_summary = []
                    for ri, real in enumerate(range(n_realizations)):
                        for k in mod_errors.keys():
                            summary_row = [model, site_number, ri, k, mod_errors[k][ri]]
                            model_errors_summary.append(summary_row)
                    model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                    error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                        
        ### NON-ENSEMBLE MODELS
        else:      
            for si, site_number in enumerate(sites):
                
                # Get overlapping obs and sim timeseries
                Q_obs = data['obs'].loc[subset_start_date:subset_end_date, site_number].dropna()
                Q_sim = data[model].loc[Q_obs.index, site_number]
                assert(len(Q_obs) == len(Q_sim)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'

                if by_year:
                    year_min = Q_obs.index.min().year
                    year_max = Q_obs.index.max().year
                    for yr in range(year_min, year_max):
                        Q_obs_yr = Q_obs.loc[Q_obs.index.year == yr].dropna()
                        Q_sim_yr = Q_sim.loc[Q_obs_yr.index]
                        assert(len(Q_obs_yr) == len(Q_sim_yr)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'
                        if len(Q_obs_yr) < 30:
                            continue
                        
                        # Get all error metrics and reformat into list
                        mod_errors = get_errors(Q_obs_yr, Q_sim_yr, bias=False)
                        model_errors_summary = []
                        for k in mod_errors.keys():
                            summary_row = [model, site_number, 0, k, mod_errors[k][0], yr]
                            model_errors_summary.append(summary_row)
                        model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                        error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                
                else:
                    # Get errors   
                    mod_errors = get_errors(Q_obs, Q_sim, bias=True)

                    # Reformat to meet error_summary format
                    model_errors_summary = []
                    for k in mod_errors.keys():
                        summary_row = [model, site_number, 0, k, mod_errors[k][0]]
                        model_errors_summary.append(summary_row)
                    model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                    error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
    return error_summary


def group_metric_data_yearbin(error_summary_annual, bin_size, metric):
    # Filter the DataFrame for the 'nse' metric
    metric_data = error_summary_annual[error_summary_annual['metric'] == metric]
    metric_data = metric_data.dropna(subset=['value'])

    # Bin the years into N-year bins
    year_min, year_max = 1945, 2022  # Define your min and max years
    bins = range(year_min, year_max + bin_size, bin_size)  # Create bins
    labels = [f'{i}-{i + bin_size - 1}' for i in bins[:-1]]  # Create labels for bins
    
    bins = np.array([1945, 1955, 1965, 1975, 1983, 1995, 2005, 2016])
    labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]

    
    metric_data['year_bin'] = pd.cut(metric_data['year'], bins=bins, labels=labels, right=False)

    # Ensure the 'value' column is numeric
    metric_data['value'] = pd.to_numeric(metric_data['value'], errors='coerce')

    # Drop NaNs that were coerced during the numeric conversion
    metric_data.dropna(subset=['value'], inplace=True)
    
    return metric_data



def get_xQn_flow(data, x, n):
    ### find the worst x-day rolling average each year, then get the value of this with n-year return interval
    data_rolling = data.rolling(x).mean()[x:]
    data_rolling_annualWorst = data_rolling.resample('A').min()
    xQn = np.percentile(data_rolling_annualWorst, 100 / n)
    return xQn


def calculate_xQN_with_bootstrap(flow_series, x, N, n_bootstrap=1000):

    def single_xQN(sorted_flows, N):
        n = len(sorted_flows)
        ranks = np.arange(1, n + 1)
        P = ranks / (n + 1)
        closest_idx = np.argmin(np.abs(P - 1/N))
        return sorted_flows.iloc[closest_idx]
    
    # Calculate x-day moving averages
    x_day_avg = flow_series.rolling(window=x).mean()[x:]
    sorted_flows = x_day_avg.sort_values()
    
    # Calculate xQN without bootstrapping
    x_QN = single_xQN(sorted_flows, N)
    
    # Bootstrap to estimate confidence intervals
    bootstrap_xQN = []
    for _ in range(n_bootstrap):
        resampled_flows = sorted_flows.sample(frac=1, replace=True)
        bootstrap_xQN.append(single_xQN(resampled_flows, N))
    
    bootstrap_xQN = np.array(bootstrap_xQN)
    lower_bound = np.percentile(bootstrap_xQN, 5)
    upper_bound = np.percentile(bootstrap_xQN, 95)
    
    return x_QN, lower_bound, upper_bound