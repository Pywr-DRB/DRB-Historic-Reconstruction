# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import datetime as dt
import seaborn as sns
import h5py
import shutil

# Import the QPPQ model
from QPPQModel import StreamflowGenerator

from utils.data_processing import export_ensemble_to_hdf5
from utils.inflow_scaling_regression import train_inflow_scale_regression_models, predict_inflow_scaling
from utils.inflow_scaling_regression import get_quarter

# Directory to pywrdrb project
pywrdrb_directory = '../Pywr-DRB/'
sys.path.append(pywrdrb_directory)

from pywrdrb.pywr_drb_node_data import obs_pub_site_matches, obs_site_matches, upstream_nodes_dict, downstream_node_lags
from pywrdrb.utils.lists import reservoir_list


def aggregate_node_flows(df):
    """Sums flows from different sites in site_matches for each node.

    Args:
        df (pandas.DataFrame): Reconstructed site flows, for each gauge and or PUB location.

    Returns:
        pandas.DataFrame: Reconstructed flows aggregated for Pywr-DRB nodes
    """
    for node, sites in obs_pub_site_matches.items():
        if sites:
            df.loc[:,node] = df.loc[:, sites].sum(axis=1)    
    return df


def add_upstream_catchment_inflows(inflows):
    """
    Adds upstream catchment inflows to get cumulative flow at downstream nodes. THis is inverse of subtract_upstream_catchment_inflows()

    Inflow timeseries are cumulative. For each downstream node, this function adds the flow into all upstream nodes so
    that it represents cumulative inflows into the downstream node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows added.
    """
    ### loop over upstream_nodes_dict in reverse direction to avoid double counting
    for node in list(upstream_nodes_dict.keys())[::-1]:
        for upstream in upstream_nodes_dict[node]:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows[node].iloc[lag:] += inflows[upstream].iloc[:-lag].values
                ### add same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows[node].iloc[:lag] += inflows[upstream].iloc[:lag].values
            else:
                inflows[node] += inflows[upstream]

        ### if catchment inflow is negative after adding upstream, set to 0 (note: this shouldnt happen)
        inflows[node].loc[inflows[node] < 0] = 0
    return inflows



### main ###
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

    max_annual_NA_fill = 10                 # Amount of data allowed to be missing in 1 year for that year to be used.
    log_fdc_interpolation = True            # Keep True. If the QPPQ should be done using log-transformed FDC 

    # Set output name based on specs
    dataset_name = f'obs_pub_{donor_fdc}_NYCScaled' if regression_nhm_inflow_scaling else f'obs_pub_{donor_fdc}'

    # Constants
    cms_to_mgd = 22.82
    fdc_quantiles = np.linspace(0.00001, 0.99999, 200)

    ### Load data 
    Q = pd.read_csv(f'./data/historic_unmanaged_streamflow_1900_2023_cms.csv', sep = ',', index_col = 0, parse_dates=True)*cms_to_mgd
    nhm_gauge_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nhmv10.csv', sep =',',  index_col = 0, parse_dates=True)
    nhm_inflow = pd.read_csv(f'{pywrdrb_directory}/input_data/catchment_inflow_nhmv10.csv', sep =',',  index_col = 0, parse_dates=True)
    
    nwm_gauge_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nwmv21.csv', sep =',',  index_col = 0, parse_dates=True)
    nwm_inflow = pd.read_csv(f'{pywrdrb_directory}/input_data/catchment_inflow_nwmv21.csv', sep =',',  index_col = 0, parse_dates=True)
    
    nyc_nhm_inflows = pd.read_csv('./data/nyc_inflow_nhm_streamflow.csv', sep = ',', index_col = 0, parse_dates=True)

    prediction_locations = pd.read_csv(f'./data/prediction_locations.csv', sep = ',', index_col=0)
    gauge_meta = pd.read_csv(f'./data/drb_unmanaged_usgs_metadata.csv', sep = ',', dtype = {'site_no':str})
    gauge_meta.set_index('site_no', inplace=True)

    # Some gauge data is faulty
    gauge_meta.loc['01414000', 'begin_date'] = '1996-12-05'
    gauge_meta.loc['0142400103', 'begin_date'] = '1996-12-05'

    # Get estiamtes of FDCs at all nodes; to be used for QPPQ when no data is available
    node_fdcs = pd.DataFrame(index = prediction_locations.index, columns=fdc_quantiles, dtype='float64')

    # Calculate FDC values using donor model flow (NHM or NWM)
    for i, node in enumerate(prediction_locations.drop('delTrenton').index):    
        if donor_fdc == 'nhmv10':
            fdc_donor_flow = nhm_gauge_flow if node in reservoir_list else nhm_inflow
        elif donor_fdc == 'nwmv21':
            fdc_donor_flow = nwm_gauge_flow if node in reservoir_list else nwm_inflow
        else:
            print('Invalid donor_fdc specification. Options: nhmv10, nwmv21')
            return
        
        nonzero_flow= fdc_donor_flow.loc[:,node].values
        nonzero_flow= nonzero_flow[nonzero_flow > 0.0]
        assert(len(nonzero_flow) > 0), f'No nonzero flow data for {node} in {donor_fdc} data.'
        node_fdcs.loc[node, :] = np.quantile(nonzero_flow, fdc_quantiles)

    # Remove outflow gauges from flow data
    for node, sites in obs_pub_site_matches.items():
        if f'USGS-{node}' in Q.columns:
            Q = Q.drop(f'USGS-{node}', axis=1)

    # Remove Trenton mainstem gauge
    mainstem_nodes= ['delMontague', 'delTrenton', 'delDRCanal', 'delLordville', 'outletAssunpink', 'outletSchuylkill']
    if remove_mainstem_gauges:
        for node in mainstem_nodes:
            if f'USGS-{obs_site_matches[node][0]}' in Q.columns:
                Q = Q.drop(f'USGS-{obs_site_matches[node][0]}', axis=1)
    
            obs_pub_site_matches[node] = None

    # Make sure other inflow gauges are in the dataset
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
        if sites is None:
            reconstructed_sites.append(node)
        else:
            for s in sites:
                reconstructed_sites.append(s)

    # Intialize storage
    Q_reconstructed = pd.DataFrame(index=max_daterange, columns = reconstructed_sites)
    ensemble_Q_reconstructed_catchment_inflows = {}
    ensemble_Q_reconstructed_gage_flows = {}
    for node, sites in obs_pub_site_matches.items():
        ensemble_Q_reconstructed_catchment_inflows[node] = {}
        ensemble_Q_reconstructed_gage_flows[node] = {}

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
        print(f'Generating realization {real+1} of {N_REALIZATIONS} using {donor_fdc} FDCs.')
        for i, dates in enumerate(daterange_subsets):
            # Run predictions one location at a time
            for node, sites in obs_pub_site_matches.items():
                
                # delTrenton inflows occur at delDRCanal: set as zeros and go to next node
                if node == 'delTrenton':
                    Q_reconstructed.loc[dates[0]:dates[1], node] = np.zeros(len(Q_reconstructed.loc[dates[0]:dates[1], node]))
                    continue
                    
                # Pull gauges that have flow during daterange
                dates = [str(d) for d in dates]
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
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q.loc[dates[0]:dates[1], f'USGS-{s}'].values
                            
                            # Fill NA using median                    
                            na_indices = Q.loc[dates[0]:dates[1],:].loc[Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), :].index
                            median_flow = np.median(Q.loc[dates[0]:dates[1], :].loc[~Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), f'USGS-{s}'])
                            Q_reconstructed.loc[na_indices.date, s] = median_flow
                                
                        # If flow data isn't available, use historic observation to generate FDC and make PUB predictions
                        else:
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
                        
                    # Store results
                    Q_reconstructed.loc[dates[0]:dates[1], node] = Q_pub
                    assert(Q_reconstructed.loc[dates[0]:dates[1], node].isna().sum() == 0), ' There are NAs in the reconstruction at node {node}'
                    

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


        ### Process for Pywr-DRB
        # Aggregate to flows across inflow sites for each node
        Q_reconstructed_catchment_inflows= aggregate_node_flows(Q_reconstructed.copy())
        Q_reconstructed_catchment_inflows = Q_reconstructed_catchment_inflows.loc[:, list(obs_pub_site_matches.keys())]
        Q_reconstructed_catchment_inflows.index.name = 'datetime'
        
        
        ## Accumulate inflows to estimate unmanaged total gauge flows
        Q_reconstructed_gage_flows = add_upstream_catchment_inflows(Q_reconstructed_catchment_inflows.copy())

        ## Store realization data
        # We want ensemble data to be grouped by node, not by realization
        for node, sites in obs_pub_site_matches.items():
            ensemble_Q_reconstructed_catchment_inflows[node][f'realization_{real}'] = Q_reconstructed_catchment_inflows[node].values
            ensemble_Q_reconstructed_gage_flows[node][f'realization_{real}'] = Q_reconstructed_gage_flows[node].values
    

    ##############
    ### Export ###
    ##############

    if N_REALIZATIONS == 1:
        # Site flows
        Q_reconstructed.to_csv(f'./outputs/historic_reconstruction_{dataset_name}.csv', sep = ',')
        
        # Catchment inflows
        Q_reconstructed_catchment_inflows.to_csv(f'./outputs/catchment_inflow_{dataset_name}.csv', sep = ',')
        shutil.copyfile(f'./outputs/catchment_inflow_{dataset_name}.csv', 
                                f'{pywrdrb_directory}/input_data/catchment_inflow_{dataset_name}.csv')
        # Gage flows
        Q_reconstructed_gage_flows.to_csv(f'./outputs/gage_flow_{dataset_name}.csv', sep = ',')
        shutil.copyfile(f'./outputs/gage_flow_{dataset_name}.csv', 
                        f'{pywrdrb_directory}/input_data/gage_flow_{dataset_name}.csv')
        
    elif N_REALIZATIONS > 1:
        
        # Convert to dataframes
        df_ensemble_catchment_inflows = {}
        df_ensemble_gage_flows = {}
        for node, sites in obs_pub_site_matches.items():
            df_ensemble_catchment_inflows[node] = pd.DataFrame(ensemble_Q_reconstructed_catchment_inflows[node],
                                                        columns= list(ensemble_Q_reconstructed_catchment_inflows[node].keys()),
                                                        index = pd.to_datetime(Q_reconstructed.index))
            df_ensemble_gage_flows[node] = pd.DataFrame(ensemble_Q_reconstructed_gage_flows[node],
                                                columns=list(ensemble_Q_reconstructed_gage_flows[node].keys()),
                                                index = pd.to_datetime(Q_reconstructed.index))
        
        # Catchment inflows
        export_ensemble_to_hdf5(df_ensemble_catchment_inflows, output_file= f'./outputs/ensembles/catchment_inflow_{dataset_name}_ensemble.hdf5')
        shutil.copyfile(f'./outputs/ensembles/catchment_inflow_{dataset_name}_ensemble.hdf5',
                        f'{pywrdrb_directory}/input_data/historic_ensembles/catchment_inflow_{dataset_name}_ensemble.hdf5')

        # Gage flows
        export_ensemble_to_hdf5(df_ensemble_gage_flows, output_file= f'./outputs/ensembles/gage_flow_{dataset_name}_ensemble.hdf5')
        shutil.copyfile(f'./outputs/ensembles/gage_flow_{dataset_name}_ensemble.hdf5',
                        f'{pywrdrb_directory}/input_data/historic_ensembles/gage_flow_{dataset_name}_ensemble.hdf5')

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
