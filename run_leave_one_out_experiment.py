"""
This script runs the leave-one-out experiment.

First, unmanaged streamflow data is loaded, then a single gauge is removed from the dataset.
The QPPQ method is used to reconstruct the streamflow at the removed gauge location.
This is repeated for each gauge in the dataset.
"""

import pandas as pd
import numpy as np
import h5py 
import pynhd as nhd

from methods.generator.QPPQModel import StreamflowGenerator
from methods.generator.single_site_generator import generate_single_gauge_reconstruction
from methods.processing.hdf5 import export_ensemble_to_hdf5
from methods.processing import get_upstream_gauges

def leave_one_out_prediction(model_streamflows, 
                             model_gauge_matches, 
                             unmanaged_gauge_flows,
                             unmanaged_gauge_meta,
                             K,
                             output_filename,
                             gauge_subcatchments,
                             n_realizations=1,
                             start_year=1983,
                             end_year=2020):
    """Leave-one-out experiment for QPPQ reconstruction.

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
    
    # Storage
    diagnostic_station_list = []   
    Q_predicted_ensemble = {}
    Q_predicted = {}
    datetime_index= pd.date_range(f'{start_year}-01-01', 
                                  f'{end_year}-12-31', freq='D')


    ## Add rows corresponding to marginal flow predictions
    for row in model_gauge_matches.itertuples():
        test_station_no = row.site_no
        if test_station_no in gauge_subcatchments.keys():
            test_subcatchment_stations = gauge_subcatchments[test_station_no]

            # Add the marginal flow as a new row
            if len(test_subcatchment_stations) > 0:
                add_data = {'site_no': f'{test_station_no}_marginal',
                            'comid': row.comid}
                model_gauge_matches = pd.concat([model_gauge_matches, 
                                                    pd.DataFrame(add_data, index=[0])], axis=0)
        else:
            continue
        
    ## Loop through unmanaged stations
    for row in model_gauge_matches.itertuples():

        # Check if marginal flow prediction (subcatchment flow is removed)
        predict_marginal_flow = True if 'marginal' in row.site_no else False    
        
        # Get station ID and NHM segment ID
        station_id = row.site_no
        test_station_no = row.site_no.split('_')[0] if predict_marginal_flow else row.site_no
        test_comid = row.comid
        
        ## Check if the test station is both unmanaged and in model set
        # If not, skip
        if (test_station_no in unmanaged_gauge_meta.index) and (test_comid in model_streamflows.columns):
            diagnostic_station_list.append(test_station_no)
        else:
            continue
        
        # Coordinates of prediction site
        coords= (unmanaged_gauge_meta.loc[test_station_no, 'long'],
                unmanaged_gauge_meta.loc[test_station_no, 'lat'])
        

        # Pull out the model flow for the test station
        model_site_flow = model_streamflows.loc[:, test_comid].copy()
        
        if predict_marginal_flow:
            # Get subcatchment flows
            test_subcatchment_stations = gauge_subcatchments[test_station_no]
            
            # Change subcatchment station IDs to comids
            test_subcatchment_comids = [model_gauge_matches.loc[model_gauge_matches.site_no == s, 
                                                                'comid'].values[0] for s in test_subcatchment_stations]
            
            for c in test_subcatchment_comids:
                assert(c in model_streamflows.columns), f'comid {c} not found in model_streamflows.columns.'
                
            model_upstream_site_flow = model_streamflows.loc[:, test_subcatchment_comids].sum(axis=1)
            
            # Subtract upstream catchment flows
            model_site_flow = model_site_flow - model_upstream_site_flow
            model_site_flow[model_site_flow < 0] = 0
            
        # Remove zeros from model flow
        model_site_flow[model_site_flow ==0] = model_site_flow[model_site_flow > 0].min()

        # Get FDC prediction derived from NHM
        fdc_quantiles = np.linspace(0.00001, 0.99999, 200)    
        fdc_prediction = np.quantile(model_site_flow, fdc_quantiles)
        
        assert(np.isnan(fdc_prediction).sum() == 0), f'fdc_prediction has NAs for gauge {test_station_no}.'


        ### Predict ###
        Q_hat = generate_single_gauge_reconstruction(station_id= test_station_no,
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
            Q_predicted[station_id] = Q_hat.values.flatten()
            assert(Q_hat.shape[0] == len(datetime_index)), 'Q_hat and datetime_index must be the same length.'
        else:        
            assert(Q_hat['realization_1'].shape[0] == len(datetime_index)), f'Q_hat and datetime_index must be the same length but have sizes {Q_hat["realization_1"].shape} and {len(datetime_index)}.'
            Q_predicted_ensemble[station_id] = Q_hat
            

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
    return



if __name__ == '__main__':

    # Constants
    cms_to_mgd = 22.82

    ## Relevant paths
    path_to_nhm_data = '../NHM-Data-Retrieval/outputs/hdf'

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
    nhm_gauge_matches['site_no'] = nhm_gauge_matches['gage_id']

    ## NWMv2.1
    # Streamflows
    drb_nwm_segment_flows = pd.read_csv('./data/NWMv21/nwmv21_unmanaged_gauge_streamflow_daily_mgd.csv', 
                                        sep = ',', index_col=0, parse_dates=True)
    drb_nwm_segment_flows= drb_nwm_segment_flows.loc['1983-10-01':, :]
    
    # Metadata for matching sites
    nwm_gauge_matches = pd.read_csv('./data/NWMv21/nwmv21_unmanaged_gauge_metadata.csv', 
                                    sep = ',', dtype={'site_no':'string', 'comid':'string'})
    
    
    ########################
    ### Pre-process data ###
    ########################

    # Find sites that are present in USGS, NHM, and NWM
    loo_sites = []
    for site in nwm_gauge_matches.site_no.values:
        if site in nhm_gauge_matches.site_no.values:
            if site in unmanaged_gauge_meta.index.values:
                loo_sites.append(site)



    ## use pynhd to identify upstream gauges for each model site
    catchment_subcatchments = get_upstream_gauges(loo_sites,
                                                  unmanaged_gauge_meta,
                                                  simplify=True)


    ####################################
    ### Run leave-one-out experiment ###
    ####################################
    
    ## Specs
    min_k = 5
    max_k = 7
    n_ensemble = 30
    ensemble_k = 7
    start_year = 1983
    end_year = 2020
    
    # Loop through different generation methods
    for donor_flow in ['nhmv10', 'nwmv21']:
        
        # Assign correct flow and metadata    
        if donor_flow == 'nhmv10':
            model_streamflows = drb_nhm_segment_flows
            model_gauge_matches = nhm_gauge_matches
        elif donor_flow == 'nwmv21':
            model_streamflows = drb_nwm_segment_flows
            model_gauge_matches = nwm_gauge_matches
        else:
            raise ValueError(f'Invalid donor_flow: {donor_flow}')
        
        for K in range(min_k, max_k+1):
                
                print(f'Generating {donor_flow} based predictions with K={K} and aggregate QPPQ')
        
                output_filename = f'./outputs/LOO/loo_reconstruction_{donor_flow}_K{K}'
            
                ## Aggregate QPPQ
                leave_one_out_prediction(model_streamflows= model_streamflows,
                                            model_gauge_matches= model_gauge_matches,
                                            unmanaged_gauge_flows= unmanaged_gauge_flows,
                                            unmanaged_gauge_meta= unmanaged_gauge_meta,
                                            K= K,
                                            gauge_subcatchments= catchment_subcatchments,
                                            n_realizations= 1,
                                            output_filename=output_filename,
                                            start_year= start_year,
                                            end_year= end_year)
                        
        # # Probabalistic QPPQ with only a single K parameterization
        # print(f'Generating {donor_flow} based ensemble QPPQ predictions with K={ensemble_k} and {n_ensemble} realizations')

        # # Ensemble filename
        # output_filename = f'./outputs/LOO/loo_reconstruction_{donor_flow}_K{ensemble_k}'

        # leave_one_out_prediction(model_streamflows= model_streamflows,
        #                             model_gauge_matches= model_gauge_matches,
        #                             unmanaged_gauge_flows= unmanaged_gauge_flows,
        #                             unmanaged_gauge_meta= unmanaged_gauge_meta,
        #                             K= ensemble_k,
        #                             gauge_subcatchments= catchment_subcatchments,
        #                             n_realizations= n_ensemble,
        #                             output_filename=output_filename,
        #                             start_year= start_year,
        #                             end_year= end_year)
    


