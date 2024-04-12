"""
This script runs the leave-one-out experiment using myi4py to parallelize.
"""

from mpi4py import MPI
import pandas as pd
import json
import numpy as np
import pickle
import sys

from methods.generator.single_site_generator import generate_single_gauge_reconstruction
from methods.processing.hdf5 import export_ensemble_to_hdf5
from methods.spatial.upstream import get_immediate_upstream_sites
from methods.utils.constants import cms_to_mgd
from methods.processing.load import load_model_segment_flows, load_gauge_matches
from methods.processing.prep_loo import get_leave_one_out_sites
from methods.utils.directories import DATA_DIR, OUTPUT_DIR
from methods.processing.hdf5 import combine_hdf5_files

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

DONOR_FLOW = sys.argv[1]

def leave_one_out_prediction_mpi(model_streamflows, 
                                 model_gauge_matches, 
                                 unmanaged_gauge_flows,
                                 unmanaged_gauge_meta,
                                 K,
                                 output_filename,
                                 gauge_subcatchments,
                                 n_realizations=1,
                                 start_year=1983,
                                 end_year=2020,
                                 marginal_flow_prediction=False):
    """MPI version of the leave-one-out experiment for QPPQ reconstruction.
    Similar arguments as the original function.

    Args:
        model_streamflows (_type_): Dataframe with datetime index and station ID columns of modeled flow timeseries.
        model_gauge_matches (_type_): Dataframe with site_no, comid, long, lat columns.
        unmanaged_gauge_flows (_type_): Dataframe with datetime index and station ID columns of obs flow timeseries.
        unmanaged_gauge_meta (_type_): Dataframe with site_no, comid, long, lat columns for USGS gauges
        K (_type_): Number of neighbor gauges to consider in QPPQ.
        output_filename (str): Filename for output file without filetype extension.
        n_realizations (int, optional): Number of QPPQ predictions to make. If >1, then probabalistic sample is used. Defaults to 1.
        start_year (int, optional): First year of prediction. Defaults to 1983.
        end_year (int, optional): End year of prediction. Defaults to 2020.
    """
    
    # Distribute rows to processes
    if rank == 0:
        # Split the gauge matches DataFrame and distribute it
        split_data = np.array_split(model_gauge_matches, size)
    else:
        split_data = None
    
    # Scatter the split data to all processes
    local_data = comm.scatter(split_data, root=0)

    # Each process runs predictions on its chunk of data
    local_results = []
    for row in local_data.itertuples():
        result = predict_single_gauge(row, model_streamflows, 
                                          unmanaged_gauge_flows, 
                                          unmanaged_gauge_meta, 
                                          gauge_subcatchments, K, 
                                          n_realizations, start_year, 
                                          end_year, marginal_flow_prediction)
        local_results.append(result)

    # Gather the results from all processes
    # pickle to make data size more managable
    pickled_local_results = [pickle.dumps(result[1]) for result in local_results]
    local_station_ids = [result[0] for result in local_results]
    
    all_pickled_results = comm.gather(pickled_local_results, root=0)
    all_station_ids = comm.gather(local_station_ids, root=0)

    if rank == 0:
        # de-pickle after gathering
        all_results = [pickle.loads(pickled_result) 
                        for sublist in all_pickled_results
                        for pickled_result in sublist]
        all_ids = [id for sublist in all_station_ids for id in sublist]
        
        print(f'length of all_results {len(all_results)}')
        print(f'length of all_station_ids {len(all_ids)}')
        
        
        flat_results = [(all_ids[i], all_results[i]) for i in range(len(all_results))]
        
        
        ##############
        ### Export ###
        ##############
        
        # Compile results into final data structure
        datetime_index = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
        if n_realizations == 1:
            # Combine into a DataFrame
            Q_predicted_df = pd.DataFrame({station_id: Q for station_id, Q in flat_results}, index=datetime_index)
            # Export to CSV
            Q_predicted_df.to_csv(f'{output_filename}.csv', sep=',')
        else:
            # Combine into a dictionary of DataFrames for ensemble
            Q_predicted_ensemble = {station_id: pd.DataFrame(Q, columns=[f'realization_{i}' for i in range(n_realizations)], index=datetime_index) for station_id, Q in flat_results}
            # Export ensemble to HDF5
            export_ensemble_to_hdf5(Q_predicted_ensemble, output_file=f'{output_filename}_ensemble.hdf5')
    return
    
    
def predict_single_gauge(row, 
                         model_streamflows, 
                         unmanaged_gauge_flows, 
                         unmanaged_gauge_meta, 
                         gauge_subcatchments, 
                         K, 
                         n_realizations, 
                         start_year, 
                         end_year, 
                         marginal_flow_prediction):
    
    datetime_index= pd.date_range(f'{start_year}-01-01', 
                                  f'{end_year}-12-31', freq='D')

    # Check if marginal flow prediction (subcatchment flow is removed)
    predict_marginal_flow = True if row.n_upstream_sites > 0 else False    
    
    # Only continue if marginal_flow_prediction kwarg is True
    predict_marginal_flow = marginal_flow_prediction and predict_marginal_flow
    
    # Get station ID and NHM segment ID
    station_id = row.site_no
    test_station_no = row.site_no  # .split('_')[0] if predict_marginal_flow else row.site_no
    
    # Get subcatchment flows
    test_subcatchment_stations = gauge_subcatchments[test_station_no]
        
    # Coordinates of prediction site
    coords= (unmanaged_gauge_meta.loc[test_station_no, 'long'],
            unmanaged_gauge_meta.loc[test_station_no, 'lat'])
    

    # Pull out the model flow for the test station
    model_site_flow = model_streamflows.loc[:, test_station_no].copy()
    
    if predict_marginal_flow:
        
        assert(len(test_subcatchment_stations) == row.n_upstream_sites), f'Number of upstream gauges does not match n_upstream_sites for {row.site_no}.'

        
        model_upstream_site_flow = model_streamflows.loc[:, test_subcatchment_stations].sum(axis=1)
        
        # Write test_subcatchment_stations to a file: site: [upstream sites]
        with open(f'{OUTPUT_DIR}/removed_upstream_gauges_{test_station_no}.txt', 'w') as f:
            f.write(f'{test_station_no}: {test_subcatchment_stations}\n')
            
        # Subtract upstream catchment flows
        model_site_flow = model_site_flow - model_upstream_site_flow
        model_site_flow[model_site_flow < 0] = 0

    # Get FDC prediction derived from NHM
    model_site_flow = model_site_flow[model_site_flow > 0]
    model_site_flow = model_site_flow.dropna()
    fdc_quantiles = np.linspace(0.00001, 0.99999, 200)    
    fdc_prediction = np.quantile(model_site_flow, fdc_quantiles)
    
    # Check for NAs
    assert(np.isnan(fdc_prediction).sum() == 0), f'fdc_prediction has NAs for gauge {test_station_no}.'

    # Check for 0s or negative values
    n_zeros = (fdc_prediction == 0).sum()
    n_negatives = (fdc_prediction < 0).sum()
    if n_zeros > 0:
        warn = f'Warning: fdc_prediction has {n_zeros} zeros for gauge {test_station_no}.'
        print(warn)
    if n_negatives > 0:
        warn = f'Warning: fdc_prediction has {n_negatives} negatives for gauge {test_station_no}.'
        print(warn)
                
    ## Remove upstream sites from unmanaged flows and metadata
    # We don't want to cheat by using upstream sites with correlation 1 with the test site
    # Thanks Stedinger!
    unmanaged_gauge_flows_subset = unmanaged_gauge_flows.copy()
    unmanaged_gauge_meta_subset = unmanaged_gauge_meta.copy()
    
    if len(test_subcatchment_stations) > 0:
        for up_station in test_subcatchment_stations:
            if up_station in unmanaged_gauge_flows.columns:
                unmanaged_gauge_flows_subset = unmanaged_gauge_flows_subset.drop(columns=up_station, inplace=False)
                unmanaged_gauge_meta_subset = unmanaged_gauge_meta_subset.drop(index=up_station, inplace=False)    
                
    ### Predict ###
    Q_hat = generate_single_gauge_reconstruction(station_id= test_station_no,
                                                    Q_unmanaged= unmanaged_gauge_flows_subset,
                                                    unmanaged_gauge_meta= unmanaged_gauge_meta_subset,
                                                    fdc_prediction= fdc_prediction,
                                                    fdc_quantiles= fdc_quantiles,
                                                    long_lat= coords,
                                                    K=K,
                                                    N_REALIZATIONS= n_realizations,
                                                    start_year=start_year, end_year=end_year)
            
    ### clean up ###
    if n_realizations == 1:
        Q_hat= Q_hat.values.flatten()
        assert(Q_hat.shape[0] == len(datetime_index)), 'Q_hat and datetime_index must be the same length.'
    
    else:        
        assert(Q_hat['realization_1'].shape[0] == len(datetime_index)), f'Q_hat and datetime_index must be the same length but have sizes {Q_hat["realization_1"].shape} and {len(datetime_index)}.'
        
    return (station_id, Q_hat)


##############################################################################
##############################################################################
##############################################################################

## Relevant paths
path_to_nhm_data = '../Input-Data-Retrieval/datasets/NHMv10/'


# Restrict to DRB or broader region
filter_drb = True
boundary = 'drb' if filter_drb else 'regional'

USE_MARGINAL_UNMANAGED_FLOWS = False
MAX_ALLOWABLE_STORAGE = 2000 # Acre-ft per catchment

#################
### Load data ###
#################


# usgs gauge data
Q = pd.read_csv(f'{DATA_DIR}/USGS/drb_streamflow_daily_usgs_cms.csv',
                index_col=0, parse_dates=True)*cms_to_mgd

# usgs gauge metadata
gauge_meta = pd.read_csv(f'{DATA_DIR}/USGS/drb_usgs_metadata.csv', 
                        index_col=0, dtype={'site_no': str, 'comid': str})

## Get unmanaged catchments
unmanaged_gauge_meta = gauge_meta[gauge_meta['total_catchment_storage'] < MAX_ALLOWABLE_STORAGE]

## Get unmanaged flows
unmanaged_gauge_flows = Q[unmanaged_gauge_meta.index]

## NHMv1.0 streamflow
drb_nhm_segment_flows = load_model_segment_flows('nhmv10', station_number_columns=True)

## NWMv2.1 streamflow
drb_nwm_segment_flows = load_model_segment_flows('nwmv21', station_number_columns=True)

## Gauge matches: Metadata for matching sites
gauge_matches = load_gauge_matches()


########################
### Pre-process data ###
########################

# Find sites that are present in USGS, NHM, and NWM
loo_sites = get_leave_one_out_sites(Q, unmanaged_gauge_meta.index.values,
                                    gauge_matches['nwmv21']['site_no'].values,
                                    gauge_matches['nhmv10']['site_no'].values)

# dict containing station:[upstream stations]
with open(f'{DATA_DIR}/station_upstream_gauges.json', 'r') as f:
    station_upstream_gauges = json.load(f)

# dict containing only immediate upstream stations
loo_station_upstream_gauges = {s: [i for i in get_immediate_upstream_sites(station_upstream_gauges, s) if i in loo_sites] for s in loo_sites}

    
# Order the site_no in gauge_matches based on the number of immediate upstream sites
loo_gauge_matches = {}
loo_gauge_matches['nhmv10'] = gauge_matches['nhmv10'].loc[gauge_matches['nhmv10']['site_no'].isin(loo_sites)].copy()
loo_gauge_matches['nwmv21'] = gauge_matches['nwmv21'].loc[gauge_matches['nwmv21']['site_no'].isin(loo_sites)].copy()

# Align nhmv10 and nwmv21 by site number
loo_gauge_matches['nhmv10'] = loo_gauge_matches['nhmv10'].set_index('site_no')
loo_gauge_matches['nwmv21'] = loo_gauge_matches['nwmv21'].set_index('site_no')
loo_gauge_matches['nhmv10'] = loo_gauge_matches['nhmv10'].loc[loo_gauge_matches['nwmv21'].index]

loo_gauge_matches['nhmv10']['n_upstream_sites'] = np.nan
loo_gauge_matches['nwmv21']['n_upstream_sites'] = np.nan

for site_no in loo_gauge_matches['nhmv10'].index:
    loo_gauge_matches['nhmv10'].loc[site_no, 'n_upstream_sites'] = len(loo_station_upstream_gauges[site_no])
    loo_gauge_matches['nwmv21'].loc[site_no, 'n_upstream_sites'] = len(loo_station_upstream_gauges[site_no])

loo_gauge_matches['nhmv10'] = loo_gauge_matches['nhmv10'].sort_values('n_upstream_sites', ascending=True)
loo_gauge_matches['nwmv21'] = loo_gauge_matches['nwmv21'].sort_values('n_upstream_sites', ascending=True)

## When n_upstream_sites are equal for two sites,
# check if 1 is upstream of the other
# if so, move that one above the other
for n in loo_gauge_matches['nhmv10']['n_upstream_sites'].unique():
    sites_with_n_upstream = loo_gauge_matches['nhmv10'].loc[loo_gauge_matches['nhmv10']['n_upstream_sites'] == n, :]
    
    for site_no in sites_with_n_upstream.index:
        
        upstream_sites = loo_station_upstream_gauges[site_no]
        for up_site in upstream_sites:
            if up_site in sites_with_n_upstream.index:
                row_downstream = sites_with_n_upstream.index.get_loc(site_no)
                row_upstream = sites_with_n_upstream.index.get_loc(up_site)
                if row_downstream > row_upstream:
                    sites_with_n_upstream.loc[up_site], sites_with_n_upstream.loc[site_no] = sites_with_n_upstream.loc[up_site].copy(), sites_with_n_upstream.loc[site_no].copy()
                    break

for model in ['nhmv10', 'nwmv21']:
    loo_gauge_matches[model].loc[sites_with_n_upstream.index, :] = sites_with_n_upstream
    loo_gauge_matches[model].reset_index(drop=False, inplace=True)

# overwrite gauge matches 
gauge_matches = loo_gauge_matches

####################################
### Run leave-one-out experiment ###
####################################

## Specifications
AGG_K_MIN = 1
AGG_K_MAX = 7

ENSEMBLE_K_MIN = 2
ENSEMBLE_K_MAX = 11
N_ENSEMBLE = 10
N_ENSEMBLE_SET = 5
N_SETS = N_ENSEMBLE // N_ENSEMBLE_SET

START_YEAR = 2018 # 1945
END_YEAR = 2021

MARGINAL_FLOW_PREDICTION = False


# Add USGS-{} to column names
unmanaged_gauge_flows.columns = [f'USGS-{c}' for c in unmanaged_gauge_flows.columns]


## Loop through model FDC estimates
for DONOR_FLOW in ['nhmv10']:
    
    # Initialize variables that are copied for each process
    if DONOR_FLOW == 'nhmv10':
        model_streamflows = drb_nhm_segment_flows.copy()
        model_gauge_matches = gauge_matches['nhmv10'].copy()
    elif DONOR_FLOW == 'nwmv21':
        model_streamflows = drb_nwm_segment_flows.copy()
        model_gauge_matches = gauge_matches['nwmv21'].copy()
    
    ## Aggregage QPPQ
    ## Loop through K values
    """
    for K in range(AGG_K_MIN, AGG_K_MAX):
        if rank == 0:
            print(f'Generating {DONOR_FLOW} based predictions with K={K} and 1 realizations')
        output_filename = f'{OUTPUT_DIR}/LOO/loo_reconstruction_{DONOR_FLOW}_K{K}'
        
        leave_one_out_prediction_mpi(model_streamflows=model_streamflows, 
                                     model_gauge_matches=model_gauge_matches, 
                                     unmanaged_gauge_flows=unmanaged_gauge_flows,
                                     unmanaged_gauge_meta=unmanaged_gauge_meta,
                                     K=K,
                                     output_filename=output_filename,
                                     gauge_subcatchments=loo_station_upstream_gauges,
                                     n_realizations=1,
                                     start_year=START_YEAR,
                                     end_year=END_YEAR,
                                     marginal_flow_prediction=MARGINAL_FLOW_PREDICTION)
    """
                                     
    ## Ensemble QPPQ
    ## Loop through K values
    for K in range(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX):
        if rank == 0:
            print(f'Generating {DONOR_FLOW} based predictions with K={K} and {N_ENSEMBLE} realizations')
        
        # Memory issues arise when ensemble size is too large
        # Instead, run N sets of 100 realizations and combine after
        ensemble_set_filenames = []
        for set in range(N_SETS):      
            output_filename = f'{OUTPUT_DIR}/LOO/set{set}_loo_reconstruction_{DONOR_FLOW}_K{K}'
            ensemble_set_filenames.append(output_filename)
            
            leave_one_out_prediction_mpi(model_streamflows=model_streamflows, 
                                     model_gauge_matches=model_gauge_matches, 
                                     unmanaged_gauge_flows=unmanaged_gauge_flows,
                                     unmanaged_gauge_meta=unmanaged_gauge_meta,
                                     K=K,
                                     output_filename=output_filename,
                                     gauge_subcatchments=loo_station_upstream_gauges,
                                     n_realizations=N_ENSEMBLE_SET,
                                     start_year=START_YEAR,
                                     end_year=END_YEAR,
                                     marginal_flow_prediction=MARGINAL_FLOW_PREDICTION)

        ### Combine ensemble sets into a single file
        output_filename = f'{OUTPUT_DIR}/LOO/loo_reconstruction_{DONOR_FLOW}_K{K}'
        combine_hdf5_files(ensemble_set_filenames, output_filename)

print('DONE!')