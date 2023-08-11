import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

# PywrDRB directory
pywrdrb_dir = '../Pywr-DRB'
cms_to_mgd = 22.82

# Dict of different gauge/HRU IDs for different datasets
# Dict of different gauge/HRU IDs for different datasets
scaling_site_matches = {'cannonsville':{'nhmv10_gauges': ['1556', '1559'],
                                'nhmv10_hru': ['1562'],
                                'nwmv21_gauges': ['01423000', '0142400103'],
                                'nwmv21_hru': ['2613174'],
                                'obs_gauges': ['01423000', '0142400103']},
                        'neversink': {'nhmv10_gauges': ['1645'],
                                      'nhmv10_hru': ['1638'],
                                      'nwmv21_gauges': [],
                                      'nwmv21_hru': [],
                                      'obs_gauges': ['01435000']},
                        'pepacton': {'nhmv10_gauges': ['1440', '1441', '1443', '1437'],
                                        'nhmv10_hru': ['1449'],
                                        'nwmv21_gauges': ['01415000', '01414500', '01414000', '01413500'],
                                        'nwmv21_hru': ['1748473'],
                                        'obs_gauges': ['01415000', '01414500', '01414000', '01413500']},
                        'fewalter': {'nhmv10_gauges': ['1684', '1691'],
                                        'nhmv10_hru': ['1684', '1691', '1694'],
                                        'nwmv21_gauges': ['01447720', '01447500'],
                                        'nwmv21_hru': ['4185065'],
                                        'obs_gauges': ['01447720', '01447500']},
                        'beltzvilleCombined': {'nhmv10_gauges': ['1703'],
                                        'nhmv10_hru': ['1710'],
                                        'nwmv21_gauges': ['01449360'],
                                        'nwmv21_hru': ['4185065'],
                                        'obs_gauges': ['01449360']}}

# List of all reservoirs able to be scaled
nhm_scaled_reservoirs = list(scaling_site_matches.keys())

# Quarters to perform regression over
quarters = ('DJF','MAM','JJA','SON')


# Load observed, NHM, and NWM flows
obs_flows = pd.read_csv(f'data/historic_unmanaged_streamflow_1900_2022_cms.csv', 
                        index_col=0, parse_dates=True)*cms_to_mgd
obs_flows.columns = [i.split('-')[1] for i in obs_flows.columns]
obs_flows.index = pd.to_datetime(obs_flows.index.date)

nhmv10_flows = pd.read_csv(f'{pywrdrb_dir}/input_data/modeled_gages/streamflow_daily_nhmv10_mgd.csv', 
                        index_col=0, parse_dates=True)


# Function for compiling flow data for regression
def prep_inflow_scaling_data():
    data = pd.DataFrame()
    for node, flowtype_ids in scaling_site_matches.items():
        for flowtype, ids in flowtype_ids.items():
            if 'nhm' in flowtype:            
                data[f'{node}_{flowtype}'] = nhmv10_flows[ids].sum(axis=1)
            elif 'obs' in flowtype:
                data[f'{node}_{flowtype}'] = obs_flows[ids].sum(axis=1)
            else:
                # Not yet implemented for NWM
                pass
    return data



def get_quarter(m):
    """Return a string indicator for the quarter of the month.

    Args:
        m (int): The numerical month.

    Returns:
        str: Either 'DJF', 'MAM', 'JJA', or 'SOR'.
    """
    if m in (12,1,2):
        return 'DJF'
    elif m in (3,4,5):
        return 'MAM'
    elif m in (6,7,8):
        return 'JJA'
    elif m in (9,10,11):
        return 'SON'

def train_inflow_scale_regression_models(reservoir, inflows, 
                                         dataset='nhmv10',
                                         rolling = True, 
                                         window =3):
    """_summary_

    Args:
        reservoir (_type_): _description_
        inflows (_type_): _description_

    Returns:
        (dict, dict): Tuple with OLS model, and fit model
    """
    dataset_opts = ['nhmv10']
    assert(dataset in dataset_opts), f'Specified dataset invalid. Options: {dataset_opts}'
    
    # Rolling mean flows
    if rolling:
        inflows = inflows.rolling(f'{window}D').mean()
        inflows = inflows[window:-window]
        inflows = inflows.dropna()
    
    inflows.loc[:,['month']] = inflows.index.month.values
    inflows.loc[:, ['quarter']] = [get_quarter(m) for m in inflows['month']]
    inflows.loc[:, [f'{reservoir}_{dataset}_scaling']] = inflows[f'{reservoir}_{dataset}_hru'] / inflows[f'{reservoir}_{dataset}_gauges']
    
    lrms = {q: sm.OLS(inflows[f'{reservoir}_{dataset}_scaling'].loc[inflows['quarter'] == q].values.flatten(),
                    sm.add_constant(np.log(inflows[f'{reservoir}_{dataset}_gauges'].loc[inflows['quarter'] == q].values.flatten()))) for q in quarters}

    lrrs = {q: lrms[q].fit() for q in quarters}
    return lrms, lrrs


def predict_inflow_scaling(lrr, log_flow):

    X = sm.add_constant(log_flow)
    scaling = lrr.predict(X)
        
    scaling[scaling<1] = 1
    scaling = pd.DataFrame(scaling, index = log_flow.index, 
                           columns =['scale'])
    return scaling