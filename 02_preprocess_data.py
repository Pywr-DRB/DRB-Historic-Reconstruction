"""

Script does the following:
- Calculate FDC biases at each of the training sites
- Use mutual information to select best features for bias correction
- Save the bias correction inputs to a CSV file

"""

import pandas as pd
import numpy as np

from methods.utils.directories import DATA_DIR, OUTPUT_DIR, PYWRDRB_DIR, FIG_DIR
from methods.plotting.rsr import plot_gauge_rsr_curve
from methods.retrieval.NHM import load_nhm_streamflow_from_hdf5
from methods.bias_correction.prep import calculate_fdc
from methods.bias_correction.prep import calculate_quantile_biases
from methods.plotting.bias_heatmap import plot_bias_heatmap

from config import DATA_DIR, FIG_DIR 
from config import FDC_QUANTILES
from config import OBS_DIAGNOSTIC_STREAMFLOW_FILE, NHM_DIAGNOSTIC_STREAMFLOW_FILE
from config import DIAGNOSTIC_SITE_METADATA_FILE, UNMANAGED_GAGE_METADATA_FILE


def calculate_rsr(Q, catchment_nldi):
    """
    Calculate the RSR for each gauge
    """
    mg_to_af = 3.06888

    # Calculate mean annual flow
    Q_mean_annual = Q.resample('YS').sum().mean() * mg_to_af

    # calculate the RSR : ratio of storage to mean annual flow
    rsr = catchment_nldi['TOT_NID_STORAGE2013'] / catchment_nldi['site_no'].map(Q_mean_annual)
    
    return rsr


def get_unmanaged_gauge_metadata_using_rsr(Q, all_site_metadata, catchment_nldi):
    """
    Calculate the RSR for each gauge and identify "unmanaged" gauges
    """    
    # calculate the RSR : ratio of storage to mean annual flow
    rsr = calculate_rsr(Q, catchment_nldi)
    
    # filter
    unmanaged_gauges = rsr.loc[rsr<0.2]
    unmanaged_gauge_metadata = all_site_metadata.loc[all_site_metadata['site_no'].isin(unmanaged_gauges.index)].reset_index(drop=True)
    
    return unmanaged_gauge_metadata


def get_diagnostic_gauge_data(Q_obs, unmanaged_gauge_metadata):
    """
    Identify diagnostic sites
    """
        
    # Must have NHM ID match for each site
    diagnostic_sites = unmanaged_gauge_metadata.dropna(how='any', axis=0)
    diagnostic_sites['nhm_id'] = diagnostic_sites['nhm_id'].astype(int).astype(str)
    
    Q_nhm = load_nhm_streamflow_from_hdf5(id_subset = diagnostic_sites['nhm_id'].values, 
                                      site_metadata=diagnostic_sites,
                                      column_labels='site_no')
    
    
    # For each site, keep just the overlapping data dates for diagnostic comparison
    Q_obs_diagnostic = Q_obs.loc[Q_nhm.index, Q_nhm.columns]

    # Drops columns with fewer than `n_years` worth of non-NaN daily values
    min_diagnostic_years = 15
    min_diagnostic_days = min_diagnostic_years * 365
    keep_sites = Q_obs_diagnostic.columns[Q_obs_diagnostic.notna().sum() >= min_diagnostic_days]
    Q_obs_diagnostic = Q_obs_diagnostic.loc[:, keep_sites]


    Q_nhm_diagnostic = Q_nhm.copy()
    Q_nhm_diagnostic = Q_nhm_diagnostic.loc[:, Q_obs_diagnostic.columns]
    Q_nhm_diagnostic = Q_nhm_diagnostic.where(~Q_obs_diagnostic.isna())
    
    
    diagnostic_site_metadata = diagnostic_sites.set_index('site_no').loc[Q_obs_diagnostic.columns]
    diagnostic_site_metadata['site_no'] = diagnostic_site_metadata.index.values
    diagnostic_site_metadata.reset_index(inplace=True, drop=True)
    
    return diagnostic_site_metadata, Q_obs_diagnostic, Q_nhm_diagnostic


def load_diagnostic_gauge_streamflow(type='obs'):
    if type == 'obs':
        return pd.read_csv(OBS_DIAGNOSTIC_STREAMFLOW_FILE, 
                            index_col=0, parse_dates=True)
    elif type == 'nhm':
        return pd.read_csv(NHM_DIAGNOSTIC_STREAMFLOW_FILE, 
                            index_col=0, parse_dates=True)


if __name__ == "__main__":

#####################################
### Load observed streamflow data ###
#####################################

    from methods.load.data_loader import Data
    data_loader = Data()

    # # STREAMFLOW
    Q = data_loader.load(datatype='streamflow', sitetype='usgs', flowtype='obs')
    Q_nhm_pywrdrb = data_loader.load(datatype='streamflow', sitetype='pywrdrb', flowtype='nhm')

    # # pywrdrb node specific data
    pywrdrb_metadata = data_loader.load(datatype='metadata', sitetype='pywrdrb')
    pywrdrb_nldi = data_loader.load(datatype='nldi', sitetype='pywrdrb')
    pywrdrb_catchments = data_loader.load(datatype='catchments', sitetype='pywrdrb')
    pywrdrb_prcp = data_loader.load(datatype='daymet', sitetype='pywrdrb')

    # usgs gage catchment data
    gage_metadata = data_loader.load(datatype='metadata', sitetype='all')
    unmanaged_gage_metadata = data_loader.load(datatype='metadata', sitetype='unmanaged')
    gage_nldi = data_loader.load(datatype='nldi', sitetype='usgs')
    gage_nldi['site_no'] = gage_nldi.index

    gage_catchments = data_loader.load(datatype='catchments', sitetype='usgs')
    gage_prcp = data_loader.load(datatype='daymet', sitetype='usgs')


    #####################################
    ### filter out managed catchments ###
    #####################################

    rsr = calculate_rsr(Q, gage_nldi)
    plot_gauge_rsr_curve(rsr, FIG_DIR)

    unmanaged_gauge_metadata = get_unmanaged_gauge_metadata_using_rsr(Q, gage_metadata, gage_nldi)
    unmanaged_gauge_metadata.to_csv(UNMANAGED_GAGE_METADATA_FILE)

    ##############################
    ### Identify diagnostic sites
    ##############################

    diagnostic_site_metadata, Q_obs_diagnostic, Q_nhm_diagnostic = get_diagnostic_gauge_data(Q, unmanaged_gauge_metadata)

    # Save diagnostic data
    diagnostic_site_metadata.to_csv(DIAGNOSTIC_SITE_METADATA_FILE)
    Q_obs_diagnostic.to_csv(OBS_DIAGNOSTIC_STREAMFLOW_FILE)
    Q_nhm_diagnostic.to_csv(NHM_DIAGNOSTIC_STREAMFLOW_FILE)


    ######################################################################
    ### Calculate FDCs for diagnostic/training sites and Pywrdrb nodes ###
    ######################################################################

    ### Calc flow duration curves
    fdc = {}
    fdc['obs'] = calculate_fdc(Q_obs_diagnostic, np.array(FDC_QUANTILES))
    fdc['nhm'] = calculate_fdc(Q_nhm_diagnostic, np.array(FDC_QUANTILES))

    fdc['pywrdrb_nhm'] = calculate_fdc(Q_nhm_pywrdrb, np.array(FDC_QUANTILES))


    monthly_fdc = {}
    monthly_fdc['obs'] = calculate_fdc(Q_obs_diagnostic.resample('MS').sum().replace(0.0, np.nan), np.array(FDC_QUANTILES))
    monthly_fdc['nhm'] = calculate_fdc(Q_nhm_diagnostic.resample('MS').sum().replace(0.0, np.nan), np.array(FDC_QUANTILES))
    monthly_fdc['pywrdrb_nhm'] = calculate_fdc(Q_nhm_pywrdrb.resample('MS').sum().replace(0.0, np.nan), np.array(FDC_QUANTILES))

    # Save FDCs
    fdc['nhm'].index.name = 'site_no'
    fdc['obs'].index.name = 'site_no'
    fdc['pywrdrb_nhm'].index.name = 'name'

    monthly_fdc['nhm'].index.name = 'site_no'
    monthly_fdc['obs'].index.name = 'site_no'
    monthly_fdc['pywrdrb_nhm'].index.name = 'name'

    fdc['nhm'].to_csv(f'{OUTPUT_DIR}/nhm_diagnostic_gauge_daily_fdc.csv')
    fdc['obs'].to_csv(f'{OUTPUT_DIR}/obs_diagnostic_gauge_daily_fdc.csv')
    fdc['pywrdrb_nhm'].to_csv(f'{OUTPUT_DIR}/nhm_pywrdrb_node_daily_fdc.csv')

    monthly_fdc['nhm'].to_csv(f'{OUTPUT_DIR}/nhm_diagnostic_gauge_monthly_fdc.csv')
    monthly_fdc['obs'].to_csv(f'{OUTPUT_DIR}/obs_diagnostic_gauge_monthly_fdc.csv')
    monthly_fdc['pywrdrb_nhm'].to_csv(f'{OUTPUT_DIR}/nhm_pywrdrb_node_monthly_fdc.csv')

    ######################################################################
    ### Calculate FDC bias at diagnostic/training sites ###
    ######################################################################

    # calculate bias
    bias = calculate_quantile_biases(fdc['obs'], fdc['nhm'], percentage=False)
    pbias = calculate_quantile_biases(fdc['obs'], fdc['nhm'], percentage=True)

    # save bias data
    from config import NHM_FDC_BIAS_FILE, NHM_FDC_PERCENT_BIAS_FILE
    
    bias.index.name = 'site_no'
    pbias.index.name = 'site_no'
    bias.to_csv(NHM_FDC_BIAS_FILE)
    pbias.to_csv(NHM_FDC_PERCENT_BIAS_FILE)


    ### Plot bias (supplemental)
    from methods.plotting.bias_heatmap import plot_bias_heatmap
    plot_bias_heatmap(pbias, fig_dir=FIG_DIR)


