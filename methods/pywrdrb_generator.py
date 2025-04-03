import numpy as np
import pandas as pd

from methods.QPPQ import StreamflowGenerator
from methods.load.data_loader import Data

from methods.processing.hdf5 import export_ensemble_to_hdf5
from methods.processing.add_subtract_upstream_flows import subtract_upstream_catchment_inflows
from methods.processing.add_subtract_upstream_flows import aggregate_node_flows
from methods.inflow_scaling_regression import train_inflow_scale_regression_models, predict_inflow_scaling
from methods.inflow_scaling_regression import get_quarter, prep_inflow_scaling_data
from methods.inflow_scaling_regression import nhmv10_scaled_reservoirs
from methods.utils import output_dir

from methods.pywr_drb_node_data import obs_pub_site_matches

reservoir_list = ['cannonsville', 'pepacton', 'neversink', 
                  'wallenpaupack', 'prompton', 'shoholaMarsh', 
                   'mongaupeCombined', 'beltzvilleCombined', 
                   'fewalter', 'merrillCreek', 'hopatcong', 
                   'nockamixon', 'assunpink', 'ontelaunee', 
                   'stillCreek', 'blueMarsh', 'greenLane']


def train_inflow_scaling_models(scaled_reservoirs = nhmv10_scaled_reservoirs, 
                                scaling_rolling_window=3):
    linear_models = {}
    linear_results = {}
    for reservoir in scaled_reservoirs:
        scaling_training_flows = prep_inflow_scaling_data()
        linear_models[reservoir], linear_results[reservoir] = train_inflow_scale_regression_models(reservoir,
                                                                                                scaling_training_flows,
                                                                                                dataset='nhmv10',
                                                                                                rolling=True,
                                                                                                window=scaling_rolling_window)
    return linear_models, linear_results



def generate_pywrdrb_reconstruction(start_year, 
                            end_year, 
                            N_REALIZATIONS= 1, 
                            K = 5, inflow_scaling_regression= False, 
                            output_directory= output_dir,
                            rank=None):
    """Generates reconstructions of historic naturalized flows across the DRB. 
    
    Timeseries are generated using a combination of observed data and QPPQ prediction to estimate ungaged and historically managed streamflows. 
    Estimate flow duration curves (fdc) are derived from either NHMv10 or NWMv21 modeled streamflows and used to generate PUB 
    predictions. 
    
    If creating a single QPPQ estimate, exports CSV to ./outputs/. 
    For KNN-QPPQ ensemble of estimates, exports an hdf5 file to ./outputs/ensembles/.

    Args:
        start_year (int): Reconstruction start year; currently only set up to start on Jan 1 of given year.
        end_year (int): Reconstruction start year; currently only set up to end on Dec 31 of given year.
        N_REALIZATIONS (int, optional): Set as 1 to generate single QPPQ aggregate timeseries. Increase to generate N samples of QPPQ-KNN sampled timeseries. Defaults to 1.
        K (int, optional): Number of KNN gages to use in QPPQ. Defaults to 5.
        inflow_scaling_regression (bool, optional): If True, Cannonsville and Pepacton inflows will be scaled to estimate total HRU flow using NHM-based regression. Defaults to False.
        output_directory (str, optional): Directory to save output files. Defaults to './outputs/'.
        rank (int, optional): Rank of the MPI process for parallel work. Defaults to None.
    """

    ### Setup

    # Number of realizations for QPPQ KNN method                
    probabalistic_QPPQ = True if (N_REALIZATIONS > 1) else False

    max_annual_NA_fill = 10                 # Amount of data allowed to be missing in 1 year for that year to be used.
    log_fdc_interpolation = True            # Keep True. If the QPPQ should be done using log-transformed FDC 

    # Inflow scaling specifications
    scaling_rolling_window = 3
    scaled_reservoirs = nhmv10_scaled_reservoirs

    # Set output name based on specs
    dataset_name = f'obs_pub_nhmv10_ObsScaled' if inflow_scaling_regression else f'obs_pub_nhmv10'

    # Constants
    fdc_quantiles = np.linspace(0.00001, 0.99999, 200)

    ### Load data ### 
    data_loader = Data()

    ## Streamflow    
    Q_obs = data_loader.load(datatype='streamflow', sitetype='usgs', flowtype='obs')
    Q_nhm_nodes = data_loader.load(datatype='streamflow', sitetype='pywrdrb', flowtype='nhm')
    Q_nhm_gages = data_loader.load(datatype='streamflow', sitetype='usgs', flowtype='nhm')
        
    # Metadata
    prediction_locations = data_loader.load(datatype='prediction_locations')
    
    all_gage_meta = data_loader.load(datatype='metadata', sitetype='usgs')
    all_gage_meta.set_index('site_no', inplace=True)
    
    gage_meta = data_loader.load(datatype='metadata', sitetype='unmanaged')
    gage_meta.set_index('site_no', inplace=True)

    # keep only flow data for unmanaged gages
    unmanaged_gage_ids = gage_meta.index.intersection(Q_obs.columns).tolist()
    
    # Prompton inflow gage is dropped for some reason; add it back in
    prompton_id = '01428750'
    unmanaged_gage_ids.append(prompton_id)
    Q_obs = Q_obs.loc[:, unmanaged_gage_ids]
    gage_meta.loc[prompton_id, :] = all_gage_meta.loc[prompton_id, gage_meta.columns]

    
    ### Estimate FDCs ###
    # For ungaged catchments, this initial FDC estimate is taken from the NHM modeled flows
    # we will later apply the bias corrections to NHM derived FDCs
    nhm_site_fdcs = pd.DataFrame(index = prediction_locations.index, 
                             columns=fdc_quantiles, 
                             dtype='float64')

    for i, node in enumerate(prediction_locations.drop('delTrenton').index):
        fdc_donor_flow = Q_nhm_nodes.loc[:, node].values
        nonzero_flow= fdc_donor_flow[fdc_donor_flow > 0.0]
        assert(len(nonzero_flow) > 0), f'No nonzero flow data for {node} in NHM data.'
        nhm_site_fdcs.loc[node, :] = np.quantile(nonzero_flow, fdc_quantiles)
    
        # we also want NHM flow estimates for individual gage stations
        # that way, if 1/2 of the gages are missing, we can use NHM fdc for the missing 1/2 
        # while still using the observed data for the other 1/2
        if obs_pub_site_matches[node] is not None:
            for site in obs_pub_site_matches[node]:
                fdc_donor_flow = Q_nhm_gages.loc[:, site].values
                nonzero_flow= fdc_donor_flow[fdc_donor_flow > 0.0]
                assert(len(nonzero_flow) > 0), f'No nonzero flow data for {site} in NHM data.'
                nhm_site_fdcs.loc[site, :] = np.quantile(nonzero_flow, fdc_quantiles)
    
    nhm_site_fdcs = nhm_site_fdcs.astype('float64')

    # Remove outflow gages from flow data
    # these shouldn't be in the unmanaged set, but this is extra check
    for node, sites in obs_pub_site_matches.items():            
        if node in Q_obs.columns:
            Q_obs = Q_obs.drop(node, axis=1)

    # Make sure to remove all mainstem gages
    # these shouldn't be in the unmanaged set, but this is extra check
    mainstem_nodes= ['delMontague', 'delTrenton', 'delDRCanal', 
                     'delLordville', 'outletAssunpink', 'outletSchuylkill']

    for node in mainstem_nodes:
        if obs_pub_site_matches[node] is None:
            continue
        for s in obs_pub_site_matches[node]:                
            if s in Q_obs.columns:
                Q_obs = Q_obs.drop(s, axis=1)
        obs_pub_site_matches[node] = None

    # Make sure other inflow gages are in the dataset
    for node, sites in obs_pub_site_matches.items():
        if sites is not None:
            for s in sites:
                if s not in Q_obs.columns:
                    print(f'WARNING: Known obs inflow site {s} for node {node} is not available in flow data.')
                if s not in gage_meta.index:
                    print(f'WARNING: Known obs inflow site {s} for node {node} is not available in metadata.')
    
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
    Q_reconstructed = pd.DataFrame(index=pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31'), 
                                   columns = reconstructed_sites)
    
    ensemble_Q_reconstructed_gage_flows = {}
    ensemble_Q_reconstructed_catchment_inflows = {}
    for node, sites in obs_pub_site_matches.items():
        ensemble_Q_reconstructed_gage_flows[node] = {}
        ensemble_Q_reconstructed_catchment_inflows[node] = {}

    # Partition the date range into yearly subsets 
    # (this can be improved, to allow non-complete years)
    N_YEARS = Q_reconstructed.index.year.unique().shape[0]
    
    starts = [f'{start_year+i}-01-01' for i in range(N_YEARS)]
    ends = [f'{start_year+i}-12-31' for i in range(N_YEARS)]
    daterange_subsets = np.vstack([starts, ends]).transpose()
    assert(pd.to_datetime(daterange_subsets[-1,-1]).date() <= Q_obs.index.max().date()), 'The historic data must be more recent than QPPQ daterange.'


    ## Fit regression models for inflow scaling based on NHM flows
    if inflow_scaling_regression:
        linear_models, linear_results = train_inflow_scaling_models(scaled_reservoirs, scaling_rolling_window)
        
            
    ### QPPQ prediction ###

    ## Initialize the model
    model = StreamflowGenerator(K= K,
                                observed_flow = Q_obs.copy(),
                                observation_locations=gage_meta.copy(),
                                fdc_quantiles= fdc_quantiles,
                                probabalistic_sample = probabalistic_QPPQ,
                                probabalistic_aggregate = probabalistic_QPPQ,
                                log_fdc_interpolation= log_fdc_interpolation)
    
    
    # Generate 1 year at a time, to maximize the amount of data available for each years QPPQ
    for real in range(N_REALIZATIONS):
        if real % 10 == 0:
            print(f'Rank {rank} generating realization {real+1} of {N_REALIZATIONS} using NHM FDCs.')
            
        for i, daterange_subset_i in enumerate(daterange_subsets):
            dates = [str(d) for d in daterange_subset_i]
            
            # Run predictions one location at a time
            for node, sites in obs_pub_site_matches.items():
                
                # delTrenton inflows occur at delDRCanal (co-located nodes);
                # set as zeros and go to next node
                if node == 'delTrenton':
                    Q_reconstructed.loc[dates[0]:dates[1], node] = 0.0
                    continue
                    

                # Handle sites with historic data
                if sites is not None:
                    
                    for s in sites:
                        
                        # Pull site flow obs for this year
                        Q_site_yr = Q_obs.loc[dates[0]:dates[1], s]
                        Q_site_yr.loc[Q_site_yr <= 0.0] = pd.NA
                        number_of_nas = Q_site_yr.isna().sum()

                        # First, use observation data if completely available
                        if (number_of_nas == 0):
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q_site_yr.values
                            continue

                        # If < 10 days are missing this year, fill using median
                        elif (number_of_nas <= max_annual_NA_fill):
                            # Fill NA using median
                            Q_site_yr = Q_site_yr.fillna(Q_site_yr.median())
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q_site_yr.values
                            continue 
                        
                        
                        # Now use QPPQ since this year of data is missing; 
                        # first we determine if the obs data is available to make the fdc
                        # if now, we use the NHM fdc
                        else:
                            # Get all historic data for this site (not just this year)
                            Q_site_all = Q_obs.loc[:, s]
                            Q_site_all.loc[Q_site_all <= 0.0] = pd.NA
                            incomplete_site_flows = Q_site_all.dropna()
                            
                            # we need location for QPPQ
                            location = gage_meta.loc[s, ['long', 'lat']].values


                            # If flow data isn't available thus year, but >= 20years of data exists:
                            # use historic observation to generate FDC and make QPPQ predictions                            
                            if len(incomplete_site_flows)/365.0 > 20:
                                fdc = np.quantile(incomplete_site_flows.values,
                                                    fdc_quantiles)

                            # Otherwise, use NHM
                            else:                            
                                fdc = nhm_site_fdcs.loc[s, :].astype('float64').values


                            Q_pub = model.predict_streamflow(prediction_location=location, 
                                                                predicted_fdc=fdc,
                                                                start_date=dates[0],
                                                                end_date=dates[1]).flatten()
                                                        
                            Q_reconstructed.loc[dates[0]:dates[1], s] = Q_pub.copy()
    
                            
                
                    # Check reconstructed data is valid    
                    assert(Q_reconstructed.loc[dates[0]:dates[1], s].isna().sum() == 0), f'There are NAs in the reconstruction at site {s} of node {node}. df: {Q_reconstructed.loc[dates[0]:dates[1], s]}'
                    assert(Q_reconstructed.loc[dates[0]:dates[1], s].min() >= 0.0), f'There are negative flows in the reconstruction at site {s} of node {node}. df: {Q_reconstructed.loc[dates[0]:dates[1], s]}'

                
                ### Full PUB for the year for ungaged catchments ###
                # no historic data available, so we go straight to NHM for FDCs 
                # then proceed with QPPQ
                else:
                    
                    location = prediction_locations.loc[node, ['long', 'lat']].values
                    fdc = nhm_site_fdcs.loc[node, :].astype('float64').values
                    
                    # Run QPPQ
                    Q_pub = model.predict_streamflow(prediction_location=location, 
                                                     predicted_fdc=fdc,
                                                     start_date=dates[0],
                                                     end_date=dates[1]).flatten()                        
                        
                    # Store results
                    Q_reconstructed.loc[dates[0]:dates[1], node] = Q_pub

                    assert(Q_reconstructed.loc[dates[0]:dates[1], node].isna().sum() == 0), ' There are NAs in the reconstruction at node {node}: {}'
                    assert(Q_reconstructed.loc[dates[0]:dates[1], node].min() >= 0.0), ' There are negative flows in the reconstruction at node {node}'
                    

        ### Apply NHM scaling at Cannonsville, Pepacton, Neversink, FE Walter, Beltzville 
        if inflow_scaling_regression:
            for node in scaled_reservoirs:
                unscaled_inflows = Q_reconstructed.loc[:, obs_pub_site_matches[node]]
                        
                # Use linear regression to find inflow scaling coefficient
                # Different models are used for each quarter; done by month batches
                for m in range(1,13):
                    quarter = get_quarter(m)
                    unscaled_month_inflows = unscaled_inflows.loc[unscaled_inflows.index.month == m, :].sum(axis=1)
                    if (unscaled_month_inflows <= 0.0).any():
                        print(f'WARNING: Node {node} has zero or negative unscaled streamflow before scaling.')
                        
                    rolling_unscaled_month_inflows = unscaled_month_inflows.rolling(window=scaling_rolling_window, 
                                                                                    min_periods=1).mean()
                                        
                    rolling_unscaled_month_log_inflows = np.log(rolling_unscaled_month_inflows.astype('float64'))
                    
                    
                    month_scaling_coefs = predict_inflow_scaling(linear_results[node][quarter], 
                                                                log_flow= rolling_unscaled_month_log_inflows)
                                    
                    ## Apply scaling across gages for full month batch 
                    # Match column names to map to df
                    for site in obs_pub_site_matches[node]:
                        month_scaling_coefs[site] = month_scaling_coefs.loc[:,'scale']
                        # Multiply
                        Q_reconstructed.loc[Q_reconstructed.index.month==m, site] = Q_reconstructed.loc[Q_reconstructed.index.month==m, 
                                                                                                        site] * month_scaling_coefs[site]


        ### Process for Pywr-DRB
        # Aggregate to flows across inflow sites for each node
        Q_reconstructed_gage_flows = aggregate_node_flows(Q_reconstructed.copy())
        Q_reconstructed_gage_flows = Q_reconstructed_gage_flows.loc[:, list(obs_pub_site_matches.keys())]
        Q_reconstructed_gage_flows.index.name = 'datetime'
        
        
        # Subtract upstream gage flows from downstream node,
        # this is used to calculate marginal inflow at each node
        Q_reconstructed_catchment_inflows = subtract_upstream_catchment_inflows(Q_reconstructed_gage_flows.copy())


        ## Store realization data
        # We want ensemble data to be grouped by node, not by realization
        for node, sites in obs_pub_site_matches.items():
            ensemble_Q_reconstructed_gage_flows[node][str(real)] = Q_reconstructed_gage_flows[node].values
            ensemble_Q_reconstructed_catchment_inflows[node][str(real)] = Q_reconstructed_catchment_inflows[node].values
    

    ##############
    ### Export ###
    ##############

    if N_REALIZATIONS == 1:
        # Site flows
        Q_reconstructed.to_csv(f'{output_directory}historic_reconstruction_{dataset_name}.csv', sep = ',')
        
        # Catchment inflows
        Q_reconstructed_catchment_inflows.to_csv(f'{output_directory}catchment_inflow_{dataset_name}.csv', sep = ',')

        # Gage flows
        Q_reconstructed_gage_flows.to_csv(f'{output_directory}gage_flow_{dataset_name}.csv', sep = ',')

        
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
        output_fname = f'{output_directory}ensembles/catchment_inflow_{dataset_name}_ensemble'
        if rank is not None:
            output_fname += f'_rank_{rank}'
        output_fname += '.hdf5'            
        export_ensemble_to_hdf5(df_ensemble_catchment_inflows, output_file= output_fname)

        # Gage flows
        
        output_fname = f'{output_directory}ensembles/gage_flow_{dataset_name}_ensemble'
        if rank is not None:
            output_fname += f'_rank_{rank}'
        output_fname += '.hdf5'
        export_ensemble_to_hdf5(df_ensemble_gage_flows, output_file= output_fname)
    return

