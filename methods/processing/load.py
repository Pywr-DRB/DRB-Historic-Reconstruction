import numpy as np
import pandas as pd
import h5py
import sys

from methods.utils.directories import PYWRDRB_DIR, OUTPUT_DIR, DATA_DIR, path_to_nhm_data
from methods.utils.constants import cms_to_mgd
from methods.processing import extract_loo_results_from_hdf5, get_upstream_gauges
from methods.processing.prep_loo import get_leave_one_out_sites

from .hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers



def load_gauge_matches(boundary='drb', 
                       bbox=None):
    
    if boundary == 'bbox' and bbox is None:
        print('Error: bbox must be provided if boundary is bbox')
        return None
    
    gauge_matches = {}
    
    # NHM-Gauge matches
    # Change column name from nhm_segment_id to comid  
    # ## TODO: Fix this in the NHM-Data-Retrieval code
    gauge_matches['nhmv10'] = pd.read_csv(f'{path_to_nhm_data}/meta/{boundary}_nhm_gage_segment_ids.csv', sep = ',', 
                                    dtype = {'gage_id':'string', 'nhm_segment_id':'string'})
    gauge_matches['nhmv10']['comid'] = gauge_matches['nhmv10']['nhm_segment_id']
    gauge_matches['nhmv10']['site_no'] = gauge_matches['nhmv10']['gage_id']
    if '$id' in gauge_matches['nhmv10'].columns:
        gauge_matches['nhmv10'].drop('$id', axis=1, inplace=True)
    
    ## NWM-Gauge matches
    gauge_matches['nwmv21'] = pd.read_csv(f'{DATA_DIR}NWMv21/nwmv21_gauge_metadata.csv', 
                                    sep = ',', dtype={'site_no':'string', 'comid':'string'})

    return gauge_matches


def load_model_segment_flows(model,
                             station_number_columns=False):
    
    if model == 'nhmv10':
        segment_flows = pd.read_hdf(f'{path_to_nhm_data}/hdf/drb_seg_outflow_mgd.hdf5', 
                                    key = 'df')
        segment_flows.index = pd.to_datetime(segment_flows.index)
    elif model == 'nwmv21':
        segment_flows = pd.read_csv(f'{DATA_DIR}NWMv21/nwmv21_gauge_streamflow_daily_mgd.csv', 
                                    sep = ',', index_col=0, parse_dates=True)
    segment_flows = segment_flows.loc['1983-10-01':, :]
    
    # Relabel columns using USGS station numbers instead of model ids
    if station_number_columns:
        gauge_matches = load_gauge_matches()
        
        station_id_col = 'gage_id' if model == 'nhmv10' else 'site_no'
        
        model_station_number_columns = []
        for comid in segment_flows.columns:
            if comid in gauge_matches[model]['comid'].values:
                model_station_number_columns.append(gauge_matches[model].loc[gauge_matches[model]['comid'] == comid, 
                                                                             station_id_col].values[0])
            else:
                segment_flows.drop(comid, axis=1, inplace=True)
        segment_flows.columns = model_station_number_columns
    return segment_flows





def load_unmanaged_gauge_metadata():
    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv(f'{DATA_DIR}/USGS/drb_unmanaged_usgs_metadata.csv', sep = ',', 
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
    
    pub_datasets = ['obs_pub_nhmv10', 'obs_pub_nwmv21']
    ensemble_datasets = ['obs_pub_nhmv10_ensemble', 'obs_pub_nwmv21_ensemble']
    
    ## Unmanaged gauge metadata
    unmanaged_gauge_meta = load_unmanaged_gauge_metadata()
    gauge_matches = load_gauge_matches()
    
    # A dict to store it all
    Q = {}

    # Flows: DateTime index with USGS-{station_id} column names
    Q_observed = pd.read_csv(f'{DATA_DIR}/USGS/drb_historic_unmanaged_streamflow_cms.csv', 
                             sep = ',', 
                             dtype = {'site_no':str}, 
                             index_col=0, parse_dates=True)*cms_to_mgd
    Q['obs'] = Q_observed.copy()
    Q['obs'].index = pd.to_datetime(Q['obs'].index.date)
    Q['obs'].columns = [c.split('-')[1] for c in Q['obs'].columns]
    
    ### LOO results
    # Aggregate QPPQ predictions; keys are FDC donor model names (nhmv10, nwmv21)
    Q['obs_pub_nhmv10'] = pd.read_csv(f'{output_dir}LOO/loo_reconstruction_nhmv10_K{K}.csv', index_col=0, parse_dates=True)
    Q['obs_pub_nwmv21'] = pd.read_csv(f'{output_dir}LOO/loo_reconstruction_nwmv21_K{K}.csv', index_col=0, parse_dates=True)

    # Ensemble QPPQ; dictionary of dictionaries with keys of FDC donor model names (nhmv10, nwmv21) 
    Q['obs_pub_nhmv10_ensemble'] = extract_loo_results_from_hdf5(f'{output_dir}LOO/loo_reconstruction_nhmv10_K{7}_ensemble.hdf5')
    Q['obs_pub_nwmv21_ensemble'] = extract_loo_results_from_hdf5(f'{output_dir}LOO/loo_reconstruction_nwmv21_K{7}_ensemble.hdf5')

    ## NHMv1.0
    # Segment outflows
    Q['nhmv10'] = pd.read_hdf(f'{path_to_nhm_data}/hdf/drb_seg_outflow_mgd.hdf5', key = 'df')
    Q['nhmv10'].index = pd.to_datetime(Q['nhmv10'].index)
    Q['nhmv10'] = Q['nhmv10'].loc['1983-10-01':, :]

    ## NWMv2.1
    # Streamflows
    Q['nwmv21'] = pd.read_csv(f'{DATA_DIR}NWMv21/nwmv21_unmanaged_gauge_streamflow_daily_mgd.csv', 
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
    loo_sites = get_leave_one_out_sites(usgs_site_ids=Q['obs'].columns,
                                        modeled_site_ids=Q['nhmv10'].columns,
                                        second_modeled_site_ids=Q['nwmv21'].columns)
    print(f'{len(loo_sites)} leave-one-out sites identified.')
    
    subcatchment_gauges = get_upstream_gauges(loo_sites, 
                                              unmanaged_gauge_meta,
                                              filename = 'leave_one_out_upstream_gauges.json',
                                              simplify=True)
    
    marginal_loo_sites = []
    for site in loo_sites:
        if len(subcatchment_gauges[site]) > 0:
            marginal_loo_sites.append(f'{site}_marginal')
    print(f'{len(marginal_loo_sites)} marginal sites identified. Summing upstream flows.')
    
    ## Sum marginal flow predictions in PUB sets
    for model in pub_datasets:
        for site_marginal in marginal_loo_sites:
            site = site_marginal.split('_')[0]
            Q[model][site] = Q[model][site_marginal]
        
            contributing_gauges = subcatchment_gauges[site]
            if site in contributing_gauges:
                contributing_gauges.remove(site)
            for gauge in contributing_gauges:
                Q[model][site] += Q[model][gauge]
            
            # Check if nans
            if Q[model][site].isna().sum() > 0:
                print(f'Warning: {model} {site} has nans')
            else:
                print(f'{model} {site} has no nans')
    

    # For ensemble: total flow is predicted marginal plus sum of contributing gauges
    for model in ensemble_datasets:
        for s_marg in marginal_loo_sites:
            contributing_gauges = subcatchment_gauges[s_marg.split('_')[0]]
            contributing_sum_flow = 0
            for col in Q[model][s_marg.split('_')[0]].columns:
                for contrib in contributing_gauges:
                    contributing_sum_flow += Q[model][contrib][col]
            Q[model][s_marg.split('_')[0]][col] = Q[model][s_marg][col] + contributing_sum_flow

    # Keep just loo sites
    for model in ['obs', 'nhmv10', 'nwmv21',
                  'obs_pub_nhmv10', 'obs_pub_nwmv21']:
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