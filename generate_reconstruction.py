# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import datetime as dt
import seaborn as sns
import h5py

# Import the QPPQ model
from QPPQModel import StreamflowGenerator

from utils.data_processing import export_dict_ensemble_to_hdf5
from utils.inflow_scaling_regression import train_inflow_scale_regression_models, predict_inflow_scaling
from utils.inflow_scaling_regression import get_quarter

# Directory to pywrdrb project
pywrdrb_directory = '../Pywr-DRB/'
sys.path.append(pywrdrb_directory)

from pywrdrb.pywr_drb_node_data import obs_pub_site_matches


def generate_reconstruction(start_year, end_year, 
                        N_REALIZATIONS= 1, donor_fdc= 'nhmv10', 
                        K = 5, regression_nhm_inflow_scaling= False, 
                        remove_mainstem_gauges= True,
                        debugging= False):
    """Generates reconstructions of historic naturalized flows across the DRB. 
    
    Timeseries are generated using a combination of observed data and QPPQ prediction to estimate ungauged and historically managed streamflows. 
    Estimate flow duration curves (fdc) are derived from either NHMv10 or NWMv21 modeled streamflows and used to generate PUB 
    predictions. 
    
    If creating a single QPPQ estimate, exports CSV to ./outputs/. 
    For KNN-QPPQ ensemble of estimates, exports an hdf5 file to ./outputs/ensembles/.

    Args:
        start_year (int): Reconstruction start year; currently only set up to start on Jan 1 of given year.
        end_year (int): Reconstruction start year; currently only set up to end on Dec 31 of given year.
        N_REALIZATIONS (int, optional): Set as 1 to generate single QPPQ aggregate timeseries. Increase to generate N samples of QPPQ-KNN sampled timeseries. Defaults to 1.
        donor_fdc (str, optional): Options: 'nwmv21','nhmv10'.  FDCs are derived from donor_fdc modeled streamflows, used to generate unguaged/managed flows. Defaults to 'nhmv10'.
        K (int, optional): Number of KNN gauges to use in QPPQ. Defaults to 5.
        regression_nhm_inflow_scaling (bool, optional): If True, Cannonsville and Pepacton inflows will be scaled to estimate total HRU flow using NHM-based regression. Defaults to False.
        remove_mainstem_gauges (bool, optional): If True, the Trenton and Montague gauges are removed from observed dataset, and QPPQ is used. Defaults to True.
        debugging (bool, optional): To toggle print statements. Defaults to False.
    """

    ### Setup
    # Range for desired reconstruciton
    max_daterange = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31')

    # Number of realizations for QPPQ KNN method                
    probabalistic_QPPQ = True if (N_REALIZATIONS > 1) else False
    log_fdc_interpolation= True

    max_annual_NA_fill = 20                 # Amount of data allowed to be missing in 1 year for that year to be used.
    log_fdc_interpolation = True            # Keep True. If the QPPQ should be done using log-transformed FDC 

    # Set output name based on specs
    output_filename = f'historic_reconstruction_daily_{donor_fdc}'
    output_filename = f'{output_filename}_NYCscaled' if regression_nhm_inflow_scaling else output_filename

    # Constants
    cms_to_mgd = 22.82
    fdc_quantiles = np.linspace(0.00001, 0.99999, 200)

    ### Load data 
    Q = pd.read_csv(f'./data/historic_unmanaged_streamflow_1900_2023_cms.csv', sep = ',', index_col = 0, parse_dates=True)*cms_to_mgd
    nhm_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nhmv10.csv', sep =',',  index_col = 0, parse_dates=True)
    nwm_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nwmv21.csv', sep =',',  index_col = 0, parse_dates=True)
    nyc_nhm_inflows = pd.read_csv('./data/nyc_inflow_nhm_streamflow.csv', sep = ',', index_col = 0, parse_dates=True)

    prediction_locations = pd.read_csv(f'./data/prediction_locations.csv', sep = ',', index_col=0)
    gauge_meta = pd.read_csv(f'./data/drb_unmanaged_usgs_metadata.csv', sep = ',', dtype = {'site_no':str})
    gauge_meta.set_index('site_no', inplace=True)

    # Some gauge data is faulty
    gauge_meta.loc['01414000', 'begin_date'] = '1996-12-05'
    gauge_meta.loc['0142400103', 'begin_date'] = '1996-12-05'

    # Get estiamtes of FDCs at all nodes; to be used for QPPQ when no data is available
    node_fdcs = pd.DataFrame(index = prediction_locations.index, columns=fdc_quantiles, dtype='float64')
    if donor_fdc == 'nhmv10':
        fdc_donor_flow = nhm_flow
    elif donor_fdc == 'nwmv21':
        fdc_donor_flow = nwm_flow
        # NWM has a few 0s that need to be replaced
        for col in fdc_donor_flow.columns:
            min_positive = fdc_donor_flow.loc[fdc_donor_flow[col] > 0, col].min()
            fdc_donor_flow[col]= np.where(fdc_donor_flow[col]==0, min_positive, fdc_donor_flow[col])
    else:
        print('Invalid donor_fdc specification. Options: nhmv10, nwmv21')
        return

    for i, node in enumerate(prediction_locations.index):    
        node_fdcs.loc[node, :] = np.quantile(fdc_donor_flow.loc[:,node], fdc_quantiles)
            
    # Remove outflow gauges from flow data
    for node, sites in obs_pub_site_matches.items():
        if f'USGS-{node}' in Q.columns:
            print(f'Removing {node} from data.')
            Q = Q.drop(f'USGS-{node}', axis=1)

    # Remove Trenton mainstem gauge
    if remove_mainstem_gauges:
        if f'USGS-{obs_pub_site_matches["delTrenton"]}' in Q.columns:
            print(f'Removing Trenton gauge from data.')
            Q = Q.drop(f'USGS-{obs_pub_site_matches["delTrenton"]}', axis=1)
        if f'USGS-{obs_pub_site_matches["delMontague"]}' in Q.columns:
            print(f'Removing Montague gauge from data.')
            Q = Q.drop(f'USGS-{obs_pub_site_matches["delMontague"]}', axis=1)

        obs_pub_site_matches['delTrenton'] = None
        obs_pub_site_matches['delDRCanal'] = None
        obs_pub_site_matches['delMontague'] = None

    # Make sure other inflow gauges are in the dataset
    missing = 0
    for node, sites in obs_pub_site_matches.items():
        if sites is not None:
            for s in sites:
                if f'USGS-{s}' not in Q.columns:
                    print(f'Site {s} for node {node} is not available')
    #assert(missing == 0), 'Atleast one of the inflow gauge timeseries if not available in the data.'

    ############
    ### QPPQ ###
    ############

    # Set-up QPPQ
    reconstructed_sites = []
    for node, sites in obs_pub_site_matches.items():
        if node == 'delDRCanal':
            pass
        elif sites is None:
            reconstructed_sites.append(node)
        else:
            for s in sites:
                reconstructed_sites.append(s)

    # Intialize storage
    Q_reconstructed = pd.DataFrame(index=max_daterange, columns = reconstructed_sites)
    ensemble_Q_reconstructed = {}

    # Partition the date range into yearly subsets (this can be improved, to allow non-complete years)
    N_YEARS = int(np.floor(len(max_daterange)/365))
    starts = [f'{1945+i}-01-01' for i in range(N_YEARS)]
    ends = [f'{1945+i}-12-31' for i in range(N_YEARS)]
    daterange_subsets = np.vstack([starts, ends]).transpose()
    assert(pd.to_datetime(daterange_subsets[-1,-1]).date() <= Q.index.max().date()), 'The historic data must be more recent than QPPQ daterange.'


    ## Fit regression models for inflow scaling based on NHM flows
    if regression_nhm_inflow_scaling:
        linear_models = {}
        linear_results = {}
        for reservoir in ['cannonsville', 'pepacton']:
            linear_models[reservoir], linear_results[reservoir] = train_inflow_scale_regression_models(reservoir, 
                                                                                                    nyc_nhm_inflows)
            
    ## QPPQ prediction
    # Generate 1 year at a time, to maximize the amount of data available for each years QPPQ
    for real in range(N_REALIZATIONS):
        print(f'Generation realization {real+1} of {N_REALIZATIONS}.')
        for i, dates in enumerate(daterange_subsets):
            # Run predictions one location at a time
            for node, sites in obs_pub_site_matches.items():
                
                # Pull gauges that have flow during daterange
                Q_subset = Q.loc[dates[0]:dates[1], :].dropna(axis=1)
                subset_sites = [f'{i.split("-")[1]}' for i in Q_subset.columns]
                gauge_meta_subset = gauge_meta.loc[subset_sites, :]
                gauge_meta_subset.index = Q_subset.columns
                
                ## Initialize the model
                model = StreamflowGenerator(K= K,
                                            observed_flow = Q_subset, 
                                            fdc_quantiles= fdc_quantiles,
                                            observation_locations=gauge_meta_subset,
                                            probabalistic_sample = probabalistic_QPPQ,
                                            log_fdc_interpolation= log_fdc_interpolation)

                # Handle sites with historic data
                if sites is not None:
                    for s in sites:
                        # First, use observation data if available
                        number_of_nas = Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna().sum()
                        if (number_of_nas == 0):
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q.loc[dates[0]:dates[1], f'USGS-{s}'].values
                        elif (number_of_nas <= max_annual_NA_fill):
                            # print(f'Filling {number_of_nas} NAs for site {s} using median.')
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q.loc[dates[0]:dates[1], f'USGS-{s}'].values
                            
                            # Fill NA using median                    
                            na_indices = Q.loc[dates[0]:dates[1],:].loc[Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), :].index
                            median_flow = np.median(Q.loc[dates[0]:dates[1], :].loc[~Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), f'USGS-{s}'])
                            Q_reconstructed.loc[na_indices.date, s] = median_flow
                                
                        # If flow data isn't available, use historic observation to generate FDC and make PUB predictions
                        else:
                            # print(f'Using partial record for {s} during {dates}')
                            location = gauge_meta.loc[s, ['long', 'lat']].values
                            incomplete_site_flows = Q.loc[:, f'USGS-{s}'].dropna(axis=0)
                            
                            # Only use site flows for FDC if longer than 10-year record
                            if len(incomplete_site_flows)/365 >= 10:
                                fdc = np.quantile(incomplete_site_flows.values, fdc_quantiles)
                            else:
                                fdc = node_fdcs.loc[node, :].astype('float').values            
                            
                            # Run QPPQ
                            Q_pub = model.predict_streamflow(location, fdc).values.flatten()
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q_pub
                else:
                    ## Full PUB for the year
                    location = prediction_locations.loc[node, ['long', 'lat']].values
                    fdc = node_fdcs.loc[node, :].astype('float64').values
                    
                    # Run QPPQ
                    Q_pub = model.predict_streamflow(location, fdc).values.flatten()                        
                    if node == 'wallenpaupack' and debugging:
                        print(f'NEP obs nep d1: {model.observed_nep_timeseries[5,:]}')
                        print(f'NEP wt obs d1: {model.weighted_nep_timeseries[5, :]}')
                        print(f'Wts: {model.norm_wts}')
                        print(f'NEP pred. {model.predicted_nep_timeseries[5]}')
                    Q_reconstructed.loc[dates[0]:dates[1], node] = Q_pub

        ### Apply NHM scaling at Cannonsville and Pepacton (Optional)
        if regression_nhm_inflow_scaling:
            for node in ['cannonsville', 'pepacton']:
                unscaled_inflows = Q_reconstructed.loc[:, obs_pub_site_matches[node]]
                        
                # Use linear regression to find inflow scaling coefficient
                # Different models are used fore each quarter; done by month batches
                for m in range(1,13):
                    quarter = get_quarter(m)
                    unscaled_month_inflows = unscaled_inflows.loc[unscaled_inflows.index.month == m, :].sum(axis=1)
                    unscaled_month_log_inflows = np.log(unscaled_month_inflows.astype('float64'))
                    
                    month_scaling_coefs = predict_inflow_scaling(linear_models[node][quarter], 
                                                linear_results[node][quarter], 
                                                log_flow= unscaled_month_log_inflows,
                                                method = 'regression')
                    
                    ## Apply scaling across gauges for full month batch 
                    # Match column names to map to df
                    for site in obs_pub_site_matches[node]:
                        month_scaling_coefs[site] = month_scaling_coefs.loc[:,'scale']
                        # Multiply
                        Q_reconstructed.loc[Q_reconstructed.index.month==m, site] = Q_reconstructed.loc[Q_reconstructed.index.month==m, site] * month_scaling_coefs[site]
                        
        assert(Q_reconstructed.isna().sum().sum() == 0), ' There are NAs in the reconstruction'
        
        ensemble_Q_reconstructed[f'realization_{real}'] = Q_reconstructed.copy()


    ##############
    ### Export ###
    ##############

    print(f'Exporting {N_REALIZATIONS} of reconstruction to {output_filename}')

    if N_REALIZATIONS == 1:
        Q_reconstructed.to_csv(f'./outputs/{output_filename}_mgd.csv', sep = ',')
        Q_reconstructed.to_csv(f'{pywrdrb_directory}/input_data/modeled_gages/{output_filename}_mgd.csv', sep = ',')
    elif N_REALIZATIONS > 1:
        output_filename = f'./outputs/ensembles/{output_filename}_ensemble_mgd.hdf5'
        export_dict_ensemble_to_hdf5(ensemble_Q_reconstructed, output_filename)

    return


if __name__ == "__main__":
    
    # Read function docstring for info
    start_year= 1945
    end_year= 2022
    donor_fdc= 'nhmv10'   # Options: 'nhmv10', 'nwmv21'
    n_realizations= 1
    K_knn= 5
    regression_nhm_inflow_scaling= False
    remove_mainstem_gauges= True
    
    generate_reconstruction(start_year=start_year, end_year=end_year,
                            N_REALIZATIONS=n_realizations,
                            donor_fdc= donor_fdc, K= K_knn, 
                            regression_nhm_inflow_scaling= regression_nhm_inflow_scaling,
                            remove_mainstem_gauges=remove_mainstem_gauges)   
    
    print('Done! Go to reconstruction_diagnostics.ipynb to see the result.')
