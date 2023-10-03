


import numpy as np
import pandas as pd
import h5py
import sys

from methods.utils.directories import pywrdrb_dir, output_dir, data_dir
from methods.utils.constants import cms_to_mgd

from .hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers

def load_historic_datasets(models,
                           start_date = '1945-01-01',
                           end_date = '2022-12-31', 
                           flowtype='gage_flow'):
    
    load_ensemble_nodes = ['cannonsville', 'pepacton', 'neversink', 'delTrenton', 
                           'delMontague']
    
    # Storage
    Q = {}
    
    for m in models:
        print(f'Loading {m}...')
        # if m == 'obs':
        #     Q_observed = pd.read_csv(f'{data_dir}historic_unmanaged_streamflow_1900_2023_cms.csv', sep = ',', 
        #                         dtype = {'site_no':str}, index_col=0, parse_dates=True)*cms_to_mgd
        #     Q['obs'] = Q_observed.copy()
        #     Q['obs'].index = pd.to_datetime(Q['obs'].index.date)
        #     Q['obs'].columns = [c.split('-')[1] for c in Q['obs'].columns]
        #     Q['obs'] = Q['obs'].loc[start_date:end_date]
        
        if 'ensemble' in m:
            Q[m] = {}
            fname = f'{output_dir}/ensembles/{flowtype}_{m}.hdf5'
            realization_numbers= get_hdf5_realization_numbers(fname)
            for i in realization_numbers:
                Q[m][f'realization_{i}'] = extract_realization_from_hdf5(fname, i,
                                                                         nodes = load_ensemble_nodes)
                Q[m][f'realization_{i}'] = Q[m][f'realization_{i}'].loc[start_date:end_date]
        else:
            if 'obs_pub' in m:
                fname = f'{output_dir}/{flowtype}_{m}.csv'    
            else:
                fname = f'{pywrdrb_dir}/input_data/{flowtype}_{m}.csv'

            # Load
            Q[m] = pd.read_csv(fname, index_col=0, parse_dates=True)
            Q[m] = Q[m].loc[start_date:end_date]
    return Q