


import numpy as np
import pandas as pd
import h5py
import sys

from methods.utils.directories import pywrdrb_dir, output_dir, data_dir, path_to_nhm_data, path_to_nwm_data
from methods.utils.constants import cms_to_mgd
from methods.processing import extract_loo_results_from_hdf5, get_upstream_gauges
from methods.processing.prep_loo import get_basin_catchment_area
from .hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers



def load_gauge_matches():
    gauge_matches = {}
    
    # NHM-Gauge matches
    # Change column name from nhm_segment_id to comid  ## TODO: Fix this in the NHM-Data-Retrieval code
    gauge_matches['nhmv10'] = pd.read_csv(f'{path_to_nhm_data}/../drb_nhm_gage_segment_ids.csv', sep = ',', 
                                    dtype = {'gage_id':'string', 'nhm_segment_id':'string'})
    gauge_matches['nhmv10']['comid'] = gauge_matches['nhmv10']['nhm_segment_id']
    gauge_matches['nhmv10']['site_no'] = gauge_matches['nhmv10']['gage_id']
    
    ## NWM-Gauge matches
    gauge_matches['nwmv21'] = pd.read_csv(f'{data_dir}NWMv21/nwmv21_unmanaged_gauge_metadata.csv', 
                                    sep = ',', dtype={'site_no':'string', 'comid':'string'})

    return gauge_matches


def load_unmanaged_gauge_metadata():
    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv(f'{data_dir}drb_unmanaged_usgs_metadata.csv', sep = ',', 
                                    dtype = {'site_no':str,'comid':str})
    unmanaged_gauge_meta.set_index('site_no', inplace=True)
    return unmanaged_gauge_meta


def load_leave_one_out_datasets():
    print('Loading all leave-one-out datasets...')
    # Comparison dates
    start_date = '1983-10-01'
    end_date = '2016-12-31'

    ## QPPQ aggregate K
    K = 5

    # Bad gauge: see https://waterdata.usgs.gov/nwis/uv?site_no=01422389&legacy=1
    bad_gauge = '01422389'
    managed_main_nodes = ['01464000', '01474500']
    
    pub_datasets = ['obs_pub_nhmv10', 'obs_pub_nwmv21']
    ensemble_datasets = ['obs_pub_nhmv10_ensemble', 'obs_pub_nwmv21_ensemble']
    
    ## Unmanaged gauge metadata
    unmanaged_gauge_meta = load_unmanaged_gauge_metadata()
    gauge_matches = load_gauge_matches()
    
    # A dict to store it all
    Q = {}

    # Flows: DateTime index with USGS-{station_id} column names
    Q_observed = pd.read_csv('../data/drb_historic_unmanaged_streamflow_cms.csv', sep = ',', 
                                    dtype = {'site_no':str}, index_col=0, parse_dates=True)*cms_to_mgd
    Q['obs'] = Q_observed.copy()
    Q['obs'].index = pd.to_datetime(Q['obs'].index.date)
    Q['obs'].columns = [c.split('-')[1] for c in Q['obs'].columns]
    
    ### LOO results
    # Aggregate QPPQ predictions; keys are FDC donor model names (nhmv10, nwmv21)
    Q['obs_pub_nhmv10'] = pd.read_csv(f'../outputs/LOO/loo_reconstruction_nhmv10_K{K}.csv', index_col=0, parse_dates=True)
    Q['obs_pub_nwmv21'] = pd.read_csv(f'../outputs/LOO/loo_reconstruction_nwmv21_K{K}.csv', index_col=0, parse_dates=True)

    # Ensemble QPPQ; dictionary of dictionaries with keys of FDC donor model names (nhmv10, nwmv21) 
    Q['obs_pub_nhmv10_ensemble'] = extract_loo_results_from_hdf5(f'../outputs/LOO/loo_reconstruction_nhmv10_K{7}_ensemble.hdf5')
    Q['obs_pub_nwmv21_ensemble'] = extract_loo_results_from_hdf5(f'../outputs/LOO/loo_reconstruction_nwmv21_K{7}_ensemble.hdf5')

    ## NHMv1.0
    # Segment outflows
    Q['nhmv10'] = pd.read_hdf(f'{path_to_nhm_data}/drb_seg_outflow_mgd.hdf5', key = 'df')
    Q['nhmv10'].index = pd.to_datetime(Q['nhmv10'].index)
    Q['nhmv10'] = Q['nhmv10'].loc['1983-10-01':, :]

    ## NWMv2.1
    # Streamflows
    Q['nwmv21'] = pd.read_csv('../data/NWMv21/nwmv21_unmanaged_gauge_streamflow_daily_mgd.csv', 
                                        sep = ',', index_col=0, parse_dates=True)
    Q['nwmv21']= Q['nwmv21'].loc['1983-10-01':, :]
    
    ## Rename from COMID to gauge station numbers
    for model in ['nhmv10', 'nwmv21']:
        temp_new_col_names = []
        for comid in Q[model].columns:
            if comid in gauge_matches[model]['comid'].values:
                temp_new_col_names.append(gauge_matches[model].loc[gauge_matches[model]['comid'] == comid, 'site_no'].values[0])
            else:
                temp_new_col_names.append(comid)
        Q[model].columns = temp_new_col_names
    
    ### Leave-one-out (LOO) prediction data
    # List of sites (should be 44)
    loo_sites = list(Q['nhmv10'].columns.intersection(Q['nwmv21'].columns))

    if bad_gauge in loo_sites:
        loo_sites.remove(bad_gauge)
    for g in managed_main_nodes:
        if g in loo_sites:
            loo_sites.remove(g)
    subcatchment_gauges = get_upstream_gauges(loo_sites, unmanaged_gauge_meta,
                                          simplify=True)
    marginal_loo_sites = []
    for site in loo_sites:
        if len(subcatchment_gauges[site]) > 0:
            marginal_loo_sites.append(f'{site}_marginal')
    
    ## Sum marginal flow predictions in PUB sets
    # For single trace sets
    for model in pub_datasets:
        for s_marg in marginal_loo_sites:
            contributing_gauges = subcatchment_gauges[s_marg.split('_')[0]]
            if bad_gauge in contributing_gauges:
                contributing_gauges.remove(bad_gauge)
            Q[model][s_marg.split('_')[0]] = Q[model].loc[start_date:end_date, s_marg]  + Q[model].loc[start_date:end_date, contributing_gauges].sum(axis=1)

    # For ensemble: total flow is predicted marginal plus sum of contributing gauges
    for model in ensemble_datasets:
        for s_marg in marginal_loo_sites:
            contributing_gauges = subcatchment_gauges[s_marg.split('_')[0]]
            if bad_gauge in contributing_gauges:
                contributing_gauges.remove(bad_gauge)
            contributing_sum_flow = 0
            for col in Q[model][s_marg.split('_')[0]].columns:
                for contrib in contributing_gauges:
                    contributing_sum = contributing_sum_flow + Q[model][contrib][col]
            Q[model][s_marg.split('_')[0]][col] = Q[model][s_marg][col] + contributing_sum_flow

    # Keep just loo sites
    for model in ['obs', 'nhmv10', 'nwmv21']:
        Q[model] = Q[model][loo_sites]

    # Take only the LOO sites from each ensemble dataset
    for model in ensemble_datasets:
        for key in list(Q[model].keys()):
            if key not in loo_sites:
                Q[model].pop(key)
    
    return Q 

















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