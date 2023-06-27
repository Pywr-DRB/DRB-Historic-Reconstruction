import numpy as np
import pandas as pd
import statsmodels.api as sm

def get_quarter(m):
    if m in (12,1,2):
        return 'DJF'
    elif m in (3,4,5):
        return 'MAM'
    elif m in (6,7,8):
        return 'JJA'
    elif m in (9,10,11):
        return 'SON'

def train_inflow_scale_regression_models(reservoir, nyc_inflows):
    
    quarters = ('DJF','MAM','JJA','SON')
    nyc_inflows['quarter'] = [get_quarter(m) for m in nyc_inflows['month']]
    nyc_inflows[f'{reservoir} scaling'] = nyc_inflows[f'{reservoir}_hru'] / nyc_inflows[f'{reservoir}_gages']
    
    lrms = {q: sm.OLS(nyc_inflows[f'{res} scaling'].loc[nyc_inflows['quarter'] == q],
                    sm.add_constant(np.log(nyc_inflows[f'{res}_gages'].loc[nyc_inflows['quarter'] == q]))) for q in quarters}

    lrrs = {q: lrms[q].fit() for q in quarters}
    return lrms, lrrs


def predict_inflow_scaling(lrm, lrr, reservoir, log_flow, method = 'random'):
    exog = lrm.exog
    exog[:,1] = log_flow
    samples = lrm.get_distribution(lrr.params, scale = np.var(lrr.resid), exog = exog).rvs()
    if method == 'random':
        scaling = samples[0]
    elif method == 'median':
        scaling = np.median(samples)
    if scaling < 0:
        scaling = 
    return scaling