"""
Contains functions for calculating various hydrologic metrics.
"""

import numpy as np
import pandas as pd

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