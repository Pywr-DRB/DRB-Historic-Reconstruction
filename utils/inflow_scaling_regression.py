import numpy as np
import pandas as pd
import statsmodels.api as sm

quarters = ('DJF','MAM','JJA','SON')

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
    
    nyc_inflows = nyc_inflows['1983-10-01':]
    nyc_inflows.loc[:,['month']] = nyc_inflows.index.month.values
    nyc_inflows.loc[:, ['quarter']] = [get_quarter(m) for m in nyc_inflows['month']]
    nyc_inflows.loc[:, [f'{reservoir} scaling']] = nyc_inflows[f'{reservoir}_hru'] / nyc_inflows[f'{reservoir}_gages']
    
    lrms = {q: sm.OLS(nyc_inflows[f'{reservoir} scaling'].loc[nyc_inflows['quarter'] == q].values.flatten(),
                    sm.add_constant(np.log(nyc_inflows[f'{reservoir}_gages'].loc[nyc_inflows['quarter'] == q].values.flatten()))) for q in quarters}

    lrrs = {q: lrms[q].fit() for q in quarters}
    return lrms, lrrs



def predict_inflow_scaling(lrm, lrr, log_flow, method = 'regression'):

    X = sm.add_constant(log_flow)
    if method == 'sample':
        exog = lrm.exog
        exog[:,1] = log_flow
        samples = lrm.get_distribution(lrr.params, scale = np.var(lrr.resid), exog = exog).rvs()
        scaling = samples[0]
    elif method == 'regression':
        scaling = lrr.predict(X)
        
    scaling[scaling<1] = 1
    scaling = pd.DataFrame(scaling, index = log_flow.index, 
                           columns =['scale'])
    return scaling