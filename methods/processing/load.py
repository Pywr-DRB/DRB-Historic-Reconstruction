import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, PYWRDRB_DIR
from methods.utils.directories import path_to_nhm_data
from methods.utils.constants import cms_to_mgd
from methods.processing.hdf5 import extract_loo_results_from_hdf5


from methods.processing.hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers


reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}


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


def load_nldi_characteristics(gauge_meta, 
                              station_no_index=True,
                              pywrdrb_nodes=False):
    ## NLDI features
    if pywrdrb_nodes:
        
        
        nldi_data = pd.read_csv(f'{DATA_DIR}/NLDI/pywrdrb_node_nldi_catchment_characteristics.csv', 
                                index_col=0)
        nldi_data.index = nldi_data.index.astype(str)
        # print(nldi_data.index.values)
        
        # node_comid_map = gauge_meta['comid'].to_dict()
        # comid_node_map = {v: k for k, v in node_comid_map.items()}
        # print(comid_node_map)
        # nldi_node_idx = [comid_node_map[comid] for comid in nldi_data.index.values]
        # nldi_data.index = nldi_node_idx
        
        # for n in ['pepacton', 'neversink', 'blueMarsh']:
        #     nldi_data.loc[n,:] = nldi_data.loc[reservoir_link_pairs[n],:]
        # nldi_data.loc['delDRCanal',:] = nldi_data.loc['delTrenton', :]
        
    else:
        nldi_data = pd.read_csv(f'{DATA_DIR}/NLDI/drb_usgs_nldi_catchment_characteristics.csv', 
                                index_col=0)

        if station_no_index:
            # convert index from comid to station_id using gauge_meta
            nldi_data['site_no'] = np.array([np.nan]*len(nldi_data), dtype=object)
            for cid in nldi_data.index.values:
                if str(cid) in gauge_meta['comid'].values:
                    s_id = gauge_meta[gauge_meta['comid'] == str(cid)].index.values[0]
                    nldi_data.loc[cid, 'site_no'] = s_id
                else:
                    nldi_data.drop(cid, inplace=True, axis=1)
            nldi_data = nldi_data.set_index('site_no')
    return nldi_data



def load_unmanaged_gauge_metadata():
    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv(f'{DATA_DIR}/USGS/drb_unmanaged_usgs_metadata.csv', sep = ',', 
                                    dtype = {'site_no':str,'comid':str})
    unmanaged_gauge_meta.set_index('site_no', inplace=True)
    return unmanaged_gauge_meta


def load_leave_one_out_datasets(loo_filenames, 
                                models, 
                                load_observed=True):
    """
    Loads data from leave-one-out experiment. 
    
    Args:
        loo_filenames (list): List of string filenames, including directory path, for the leave-one-out datasets.
        models (_type_): List of model names of the form "obs_pub_<MODEL>_K<k>_ensemble". Order must match loo_filenames.

    Returns:
        dict: Structure dict[model][site] = pd.DataFrame if model is ensemble else dict[model]=pd.DataFrame. Datetime index. Site IDs columns.
    """

    ## filename, model pairs
    loo_filename_model = zip(loo_filenames, models)
    
    ## Unmanaged gauge metadata
    # A dict to store it all
    Q = {}

    ## Observed streamflow
    if load_observed:
        Q_observed = pd.read_csv(f'{DATA_DIR}/USGS/drb_streamflow_daily_usgs_cms.csv', 
                             sep = ',', 
                             dtype = {'site_no':str}, 
                             index_col=0, parse_dates=True)*cms_to_mgd
        Q['obs'] = Q_observed.copy()
        Q['obs'].index = pd.to_datetime(Q['obs'].index.date)
        Q['obs'].columns = [c if '-' not in c else c.split('-')[1] for c in Q['obs'].columns]
    
    ### LOO results
    for filename, model in loo_filename_model:
        print(f'Loading {model}...')
        if 'ensemble' in model:
            Q[model] = extract_loo_results_from_hdf5(filename)
        else:
            Q[model] = pd.read_csv(filename, index_col=0, parse_dates=True)

    return Q 




def load_historic_datasets(models,
                           start_date = '1945-01-01',
                           end_date = '2022-12-31', 
                           flowtype='gage_flow'):
    
    # Storage
    Q = {}
    
    for m in models:
        print(f'Loading {m}...')

        if 'ensemble' in m:
            Q[m] = {}
            fname = f'{OUTPUT_DIR}/ensembles/{flowtype}_{m}.hdf5'
            realization_numbers= get_hdf5_realization_numbers(fname)
            for i in realization_numbers:
                Q[m][f'{i}'] = extract_realization_from_hdf5(fname, i,
                                                             stored_by_node=True)
                Q[m][f'{i}'] = Q[m][f'{i}'].loc[start_date:end_date]
                if 'datetime' in Q[m][f'{i}'].columns:
                    Q[m][f'{i}'].drop('datetime', axis=1, inplace=True)
        else:
            if 'obs_pub' in m:
                fname = f'{OUTPUT_DIR}/{flowtype}_{m}.csv'    
            else:
                fname = f'{PYWRDRB_DIR}/input_data/{flowtype}_{m}.csv'

            # Load
            Q[m] = pd.read_csv(fname, index_col=0, parse_dates=True)
            Q[m] = Q[m].loc[start_date:end_date]
    return Q