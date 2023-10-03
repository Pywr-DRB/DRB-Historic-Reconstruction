"""
Contains functions for detecting drought periods in a time series.

Trevor Amestoy
"""

import numpy as np
import pandas as pd
import spei as si
import scipy.stats as scs


def aggregate_and_rolling_sum(Q_i,
                              window, 
                              aggregate = None):
    if aggregate is not None:
        Q_i = Q_i.resample(aggregate).sum()
    Q_i = Q_i.rolling(window).sum().iloc[window:, :].dropna()
    return Q_i
    

def calculate_ssi_values(data, 
                         window,
                         nodes,
                         aggregate = None):
    
    ssi = {}
    models = list(data.keys())
    
    for m in models:
        print(f'Calculating SSI for {m}...')
        if 'ensemble' in m:
            real_keys = list(data[m].keys())
            ssi[m] = {}
            for r in real_keys:
                Q_ri = data[m][r][nodes]
                Q_ri = aggregate_and_rolling_sum(Q_ri, window, aggregate)
                
                ssi[m][r] = pd.DataFrame(index = Q_ri.index, columns=nodes)
                for n in nodes:
                    ssi[m][r][n] = si.ssfi(Q_ri[n], dist=scs.gamma)
        else:
            Q_i = data[m][nodes]
            Q_i = aggregate_and_rolling_sum(Q_i, window, aggregate)
            
            ssi[m] = pd.DataFrame(index = Q_i.index, columns=nodes)
            for n in nodes:
                ssi[m][n] = si.ssfi(Q_i[n], dist=scs.gamma)
    
    return ssi




def get_drought_metrics(ssi):
    """Get drought start and end dates, magnitude, severity, and duration.

    Args:
        ssi (pd.Series): Array of SSI values.  

    Returns:
        pd.DataFrame: DataFrame containing all drought metrics for each drought period.
    """
    
    drought_data = {}
    drought_counter = 0
    in_critical_drought = False
    drought_days = []

    for ind in range(len(ssi)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
        else:
            # Record drought info once it ends
            if in_critical_drought:
                drought_counter += 1
                drought_data[drought_counter] = {
                    'start':ssi.index[drought_days[0]],
                    'end': ssi.index[drought_days[-1]],
                    'duration': len(drought_days),
                    'magnitude': sum(ssi.values[drought_days]),
                    'severity': min(ssi.values[drought_days])
                }
            
            # Reset counters
            in_critical_drought = False
            drought_days = [] 

    drought_metrics = pd.DataFrame(drought_data).transpose()
    return drought_metrics

