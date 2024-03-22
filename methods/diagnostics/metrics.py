"""
Contains functions for calculating various hydrologic metrics.
"""

import numpy as np
import pandas as pd
import hydroeval as he


error_metrics = ['nse', 'kge', 'r', 'alpha', 'beta',
                'log_nse', 'log_kge', 'log_r', 'log_alpha', 'log_beta', 
                'Q0.1pbias', 'Q0.2pbias', 'Q0.3pbias',
                'AbsQ0.1pbias', 'AbsQ0.2pbias', 'AbsQ0.3pbias']


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


def get_errors(Q_obs, Q_sim, bias=False):
    errors = {}
    errors['nse'] = he.evaluator(he.nse, Q_sim, Q_obs).flatten()[0]
    errors['kge'], errors['r'], errors['alpha'], errors['beta'] = he.evaluator(he.kge, Q_sim, Q_obs).flatten()
    errors['log_nse'] = he.evaluator(he.nse, Q_sim, Q_obs, transform='log').flatten()[0]
    errors['log_kge'], errors['log_r'], errors['log_alpha'], errors['log_beta'] = he.evaluator(he.kge, Q_sim, Q_obs, transform='log').flatten()    
    
    if bias:
        for q_n in [0.1, 0.2, 0.3]:
            Q_obs_low = Q_obs.loc[Q_obs < Q_obs.quantile(q_n)]
            Q_sim_low = Q_sim.loc[Q_obs < Q_obs.quantile(q_n)]
                
            # Get percent bias
            errors[f'Q{q_n}pbias'] = (Q_sim_low.sum() - Q_obs_low.sum())/Q_obs_low.sum()
            errors[f'AbsQ{q_n}pbias'] = abs(Q_sim_low.sum() - Q_obs_low.sum())/Q_obs_low.sum()
    return errors
    



def get_error_summary(data, 
                      models, 
                      sites,
                      by_year = False,
                      start_date='1983-10-01', 
                      end_date='2016-12-31'):
    
    summary_columns = ['model', 'site', 'realization', 'metric', 'value']
    if by_year:
        print('Getting error metrics at all sites for each year...')
        error_metrics = ['nse', 'kge', 'r', 'alpha', 'beta',
                'log_nse', 'log_kge', 'log_r', 'log_alpha', 'log_beta']
        summary_columns= summary_columns + ['year']
    else:
        print('Getting error metrics at all sites...')
        error_metrics = ['nse', 'kge', 'r', 'alpha', 'beta',
                'log_nse', 'log_kge', 'log_r', 'log_alpha', 'log_beta', 
                'Q0.1pbias', 'Q0.2pbias', 'Q0.3pbias',
                'AbsQ0.1pbias', 'AbsQ0.2pbias', 'AbsQ0.3pbias']
    
    # Initialize dataframe to store all data
    error_summary = pd.DataFrame(columns=summary_columns)
    
    for model in models:
        print(f'Getting errors across {model} dataset')
        subset_start_date = '1983-10-01' if model in ['nhmv10', 'nwmv21'] else start_date
        subset_end_date = '2016-12-31' if model in ['nhmv10'] else end_date
        subset_end_date = '2020-12-31' if model in ['nwmv21'] else subset_end_date
        
        if 'ensemble' in model:
            for si, site_number in enumerate(sites):
                print(f'Site {si+1} of {len(sites)}')
                n_realizations= data[model][site_number].shape[1]
                
                # Get overlapping obs and sim timeseries
                Q_obs = data['obs'].loc[subset_start_date:subset_end_date, site_number].dropna()
                Q_obs = Q_obs.loc[Q_obs > 0]
                assert(len(Q_obs) > 0), f'No overlap between Q_obs and Q_sim for {site_number}'
    
                # Get errors for every realization (and year if by_year)
                for real in range(n_realizations):
                    if by_year:
                        year_min = Q_obs.index.min().year
                        year_max = Q_obs.index.max().year
                        for yr in range(year_min, year_max):
                            Q_obs_yr = Q_obs.loc[Q_obs.index.year == yr].dropna()
                            Q_sim_yr = data[model][site_number].loc[Q_obs_yr.index, f'realization_{real}']
                            assert(len(Q_obs_yr) == len(Q_sim_yr)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'
                            if len(Q_obs_yr) < 30:
                                continue
                            
                            model_errors = get_errors(Q_obs_yr, Q_sim_yr)
                            
                            # Reformat to meet error_summary format
                            model_errors_summary = []
                            for k in model_errors.keys():
                                summary_row = [model, site_number, int(real), k, model_errors[k], yr]
                                model_errors_summary.append(summary_row)
                            model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                            error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                            
                    else:
                        Q_sim = data[model][site_number].loc[Q_obs.index, f'realization_{real}']
                        assert(len(Q_obs) == len(Q_sim)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'
                        
                        # Get all error metrics 
                        mod_errors = get_errors(Q_obs, Q_sim, bias=True)
                        
                        # Reformat to meet error_summary format
                        model_errors_summary = []
                        for k in mod_errors.keys():
                            summary_row = [model, site_number, int(real), k, mod_errors[k]]
                            model_errors_summary.append(summary_row)
                            
                        model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                        error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                        
        ### NON-ENSEMBLE MODELS
        else:      
            for si, site_number in enumerate(sites):
                
                # Get overlapping obs and sim timeseries
                Q_obs = data['obs'].loc[subset_start_date:subset_end_date, site_number].dropna()
                Q_obs = Q_obs.loc[Q_obs > 0]
                Q_sim = data[model].loc[Q_obs.index, site_number]
                assert(len(Q_obs) == len(Q_sim)), f'Lengths of Q_obs and Q_sim do not match for {site_number}'
                assert(len(Q_obs) > 30), f'No overlap between Q_obs and Q_sim for {site_number}'
                
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
                            mod_error_val = mod_errors[k]
                            if np.isnan(mod_error_val):
                                print(f'Q_obs: {Q_obs.shape} Q_sim: {Q_sim.shape}, {site_number}')
                            summary_row = [model, site_number, 0, k, mod_error_val, yr]
                            model_errors_summary.append(summary_row)
                        model_errors_summary = pd.DataFrame(model_errors_summary, columns=summary_columns)
                        error_summary = pd.concat([error_summary, model_errors_summary], ignore_index=True)
                
                else:
                    # Get errors   
                    mod_errors = get_errors(Q_obs, Q_sim, bias=True)

                    # Reformat to meet error_summary format
                    model_errors_summary = []
                    for k in mod_errors.keys():
                        mod_error_val = mod_errors[k]
                        if np.isnan(mod_error_val):
                            print(f'Q_obs: {Q_obs.shape} Q_sim: {Q_sim.shape}, {site_number}')
                            
                        summary_row = [model, site_number, 0, k, mod_error_val]
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
