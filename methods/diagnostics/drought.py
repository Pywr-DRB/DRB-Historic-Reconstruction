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
    Q_i = Q_i.iloc[:-window, :]
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
                ssi[m][r] = ssi[m][r].astype(float)
        else:
            Q_i = data[m][nodes]
            Q_i = aggregate_and_rolling_sum(Q_i, window, aggregate)
            
            ssi[m] = pd.DataFrame(index = Q_i.index, columns=nodes)
            for n in nodes:
                ssi[m][n] = si.ssfi(Q_i[n], dist=scs.gamma)
            ssi[m] = ssi[m].astype(float)
    return ssi