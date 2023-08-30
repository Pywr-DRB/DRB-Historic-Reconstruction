"""
This script runs the leave-one-out experiment.

First, unmanaged streamflow data is loaded, then a single gauge is removed from the dataset.
The QPPQ method is used to reconstruct the streamflow at the removed gauge location.
This is repeated for each gauge in the dataset.
"""

import pandas as pd
import numpy as np
import h5py 

from methods.generator.QPPQModel import StreamflowGenerator
from methods.generator.single_site_generator import generate_single_gauge_reconstruction
from methods.processing.hdf5 import export_ensemble_to_hdf5

if __name__ == '__main__':

    # Constants
    cms_to_mgd = 22.82

    ## Relevant paths
    path_to_nhm_data = '../NHM-Data-Retrieval/outputs/hdf'
    path_to_nwm_data = '../NWM-Data-Retrieval/outputs/hdf'

    #################
    ### Load data ###
    #################
    
    ## USGS 
    # Flows: DateTime index with USGS-{station_id} column names
    unmanaged_gauge_flows = pd.read_csv('./data/historic_unmanaged_streamflow_1900_2023_cms.csv', sep = ',', 
                                    dtype = {'site_no':str}, index_col=0, parse_dates=True)*cms_to_mgd

    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv('./data/drb_unmanaged_usgs_metadata.csv', sep = ',', 
                                    dtype = {'site_no':str})
    unmanaged_gauge_meta.set_index('site_no', inplace=True)
    
    # List of just station numbers
    usgs_gauge_ids = [c.split('-')[1] for c in unmanaged_gauge_flows.columns]
    
    ## NHMv1.0
    # Segment outflows
    drb_nhm_segment_flows = pd.read_hdf(f'{path_to_nhm_data}/drb_seg_outflow_mgd.hdf5', key = 'df')
    drb_nhm_segment_flows.index = pd.to_datetime(drb_nhm_segment_flows.index)
    drb_nhm_segment_flows = drb_nhm_segment_flows.loc['1983-10-01':, :]
    
    # NHM-Gauge matches
    # Change column name from nhm_segment_id to comid  ## TODO: Fix this in the NHM-Data-Retrieval code
    nhm_gauge_matches = pd.read_csv(f'{path_to_nhm_data}/../drb_nhm_gage_segment_ids.csv', sep = ',', 
                                    dtype = {'gage_id':'string', 'nhm_segment_id':'string'})
    nhm_gauge_matches['comid'] = nhm_gauge_matches['nhm_segment_id']

    ## NWMv2.1
    # Not yet ready

    ####################################
    ### Run leave-one-out experiment ###
    ####################################
    
    ## Setup
    diagnostic_station_list = []
    donor_flow = 'nhmv10' # 'nwmv21'
    n_realizations = 30
    K= 5
    
    start_year = 1983
    end_year = 2020
    
    ## Storage
    Q_predicted_ensemble = {}
    Q_predicted = {}
    datetime_index= pd.date_range(f'{start_year}-01-01', 
                                  f'{end_year}-12-31', freq='D')

    output_filename = f'./outputs/LOO/loo_reconstruction_{donor_flow}'
    
    if donor_flow == 'nhmv10':
        model_streamflows = drb_nhm_segment_flows
        model_gauge_matches = nhm_gauge_matches
    elif donor_flow == 'nwmv21':
        pass 

    ## Loop through unmanaged stations
    for row in model_gauge_matches.itertuples():

        # Get station ID and NHM segment ID
        test_station_id = row.gage_id
        test_comid = row.comid
        
        ## Check if the test station is both unmanaged and in model set
        # If not, skip
        if (test_station_id in unmanaged_gauge_meta.index) and (test_comid in model_streamflows.columns):
            # print(f'Running prediction number {len(diagnostic_station_list)+1} for station {test_station_id}')
            diagnostic_station_list.append(test_station_id)
            
        else:
            continue
        
        # Coordinates of prediction site
        coords= (unmanaged_gauge_meta.loc[test_station_id, 'long'],
                unmanaged_gauge_meta.loc[test_station_id, 'lat'])
        

        # Pull out the NHM flow for the test station
        model_site_flow = model_streamflows.loc[:, test_comid].copy()

        # Get FDC prediction derived from NHM
        fdc_quantiles = np.linspace(0.00001, 0.99999, 200)    
        fdc_prediction = np.quantile(model_site_flow, fdc_quantiles)

        ### Predict ###
        Q_hat = generate_single_gauge_reconstruction(station_id= test_station_id,
                                                     Q_unmanaged= unmanaged_gauge_flows,
                                                     unmanaged_gauge_meta= unmanaged_gauge_meta,
                                                     fdc_prediction= fdc_prediction,
                                                     fdc_quantiles= fdc_quantiles,
                                                     long_lat= coords,
                                                     K=K,
                                                     N_REALIZATIONS= n_realizations,
                                                     start_year=start_year, end_year=end_year)


        ### Store ###
        if n_realizations == 1:
            Q_predicted[test_station_id] = Q_hat.values.flatten()
            assert(Q_hat.shape[0] == len(datetime_index)), 'Q_hat and datetime_index must be the same length.'
        else:        
            assert(Q_hat['realization_1'].shape[0] == len(datetime_index)), f'Q_hat and datetime_index must be the same length but have sizes {Q_hat["realization_1"].shape} and {len(datetime_index)}.'
            Q_predicted_ensemble[test_station_id] = Q_hat
            

    ##############
    ### Export ###
    ##############

    if n_realizations == 1:
        
        # Reorganize into dataframe
        Q_predicted_df= pd.DataFrame(Q_predicted, index= datetime_index)

        # Export to CSV
        Q_predicted_df.to_csv(f'{output_filename}.csv', sep = ',')

      
    elif n_realizations > 1:
        
        # Convert to dataframes
        df_ensemble_flows= {}
        ensemble_column_names= [f'realization_{i}' for i in range(n_realizations)]
        
        for station_id in Q_predicted_ensemble.keys():
            df_ensemble_flows[station_id] = pd.DataFrame(Q_predicted_ensemble[station_id],
                                                         columns=ensemble_column_names,
                                                         index = datetime_index)
        
        # Export ensemble to HDF5
        export_ensemble_to_hdf5(df_ensemble_flows, output_file= f'{output_filename}_ensemble.hdf5')


