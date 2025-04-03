import numpy as np
import pandas as pd

from methods.QPPQ import StreamflowGenerator
from methods.load.data_loader import Data
from config import FDC_QUANTILES, OUTPUT_DIR
from methods.processing.hdf5 import export_ensemble_to_hdf5
from methods.processing.add_subtract_upstream_flows import subtract_upstream_catchment_inflows
from methods.processing.add_subtract_upstream_flows import aggregate_node_flows
from methods.inflow_scaling_regression import train_inflow_scale_regression_models, predict_inflow_scaling
from methods.inflow_scaling_regression import get_quarter, prep_inflow_scaling_data
from methods.inflow_scaling_regression import nhmv10_scaled_reservoirs
from methods.utils import output_dir

from methods.pywr_drb_node_data import obs_pub_site_matches


def generate_single_gauge_reconstruction(Q_unmanaged, 
                                         station_id, 
                                         fdc_prediction,
                                         fdc_quantiles,
                                         long_lat,
                                         unmanaged_gauge_meta,
                                         start_year=1945, 
                                         end_year=2023, 
                                         N_REALIZATIONS=1,
                                         K=5):
    """
    Makes a streamflow prediction using the aggregate or probabilistic QPPQ method, 
    given a provided fdc_prediction.

    Args:
        Q_unmanaged (pd.DataFrame): Dataframe of unmanaged gauge streamflow timeseries with datetime index and site_no column names.
        unmanaged_gauge_meta (pd.DataFrame): Dataframe of unmanaged gauge metadata with site_no as index.
        fdc_prediction (array): Array of flow values at specified fdc_quantiles. 
        fdc_quantiles (array): Array of fdc quantiles to discretize along.
        long_lat (tuple): (longitude, latitude) of the gauge to be reconstructed.
        station_id (str): USGS gauge number as a string.
        start_year (int, optional): _description_. Defaults to 1945.
        end_year (int, optional): _description_. Defaults to 2022.
        N_REALIZATIONS (int, optional): _description_. Defaults to 1.
        K (int, optional): _description_. Defaults to 5.
    """
    
    ### Setup
    ## Assertions
    assert(len(fdc_quantiles) == len(fdc_prediction)), 'fdc_quantiles and fdc_prediction must be the same length.'
    assert(len(long_lat) == 2), 'long_lat must be a tuple of length 2.'
    assert(station_id in unmanaged_gauge_meta.index), 'station_id not found in unmanaged_gauge_meta.index.'
    

    # Number of realizations for QPPQ KNN method                
    probabalistic_QPPQ = True if (N_REALIZATIONS > 1) else False
    log_fdc_interpolation = True
    
    
    # Initialize storage
    Q_reconstructed = pd.DataFrame(index=pd.date_range(f'{start_year}-01-01', 
                                                       f'{end_year}-12-31'), 
                                   columns = [station_id])
    ensemble_Q_predicted = {}
    
    # Partition the date range into yearly subsets (this can be improved, to allow non-complete years)
    N_YEARS = N_YEARS = Q_reconstructed.index.year.unique().shape[0]
    starts = [f'{start_year+i}-01-01' for i in range(N_YEARS)]
    ends = [f'{start_year+i}-12-31' for i in range(N_YEARS)]
    daterange_subsets = np.vstack([starts, ends]).transpose()
    assert(pd.to_datetime(daterange_subsets[-1,-1]).date() <= Q_unmanaged.index.max().date()), 'The historic data must be more recent than QPPQ daterange.'


    ### QPPQ prediction ###

    Q_donors = Q_unmanaged.copy()
    unmanaged_gauge_meta = unmanaged_gauge_meta.drop(index=[station_id])
    
    ## Initialize the model
    model = StreamflowGenerator(K= K,
                                observed_flow = Q_donors.copy(),
                                observation_locations=unmanaged_gauge_meta.copy(),
                                fdc_quantiles= fdc_quantiles,
                                probabalistic_sample = probabalistic_QPPQ,
                                probabalistic_aggregate = probabalistic_QPPQ,
                                log_fdc_interpolation= log_fdc_interpolation)
    
    ## QPPQ prediction
    # Generate 1 year at a time, to maximize the amount of data available for each years QPPQ
    for real in range(N_REALIZATIONS):


        for i, daterange_subset_i in enumerate(daterange_subsets):
            
            dates = [str(d) for d in daterange_subset_i]
                
            
     
            # Run QPPQ
            Q_pub = model.predict_streamflow(prediction_location=long_lat, 
                                             predicted_fdc=fdc_prediction,
                                             start_date=dates[0], 
                                             end_date=dates[1]).flatten()                        
                
            # Store results
            Q_reconstructed.loc[dates[0]:dates[1], station_id] = Q_pub.copy()
            assert(Q_reconstructed.loc[dates[0]:dates[1], station_id].isna().sum() == 0), f'There are NAs in the reconstruction for gauge {station_id}'
        
        # store realization
        ensemble_Q_predicted[f'{real}'] = Q_reconstructed.values.flatten()
    
    if N_REALIZATIONS > 1:
        return ensemble_Q_predicted
    else:
        return Q_reconstructed
    
    




def predict_single_gauge(row, 
                         fdc_prediction, 
                         unmanaged_gauge_flows, 
                         unmanaged_gauge_meta, 
                         gauge_subcatchments, 
                         K, 
                         n_realizations, 
                         start_year, 
                         end_year,
                         fdc_quantiles=FDC_QUANTILES):
    
    """
    Runs a QPPQ based flow prediction for a single row, 
    whcih contains information about a single gage such as site_no.
    
    Args:
    row (pd.Series): A row of a dataframe containing information about a single gage.
    fdc_prediction (np.array): The FDC prediction for the gage.
    unmanaged_gauge_flows (pd.DataFrame): The unmanaged gage flows.
    unmanaged_gauge_meta (pd.DataFrame): The unmanaged gage metadata.
    gauge_subcatchments (dict): A dictionary containing the subcatchments for each gage.
    K (int): The number of nearest neighbors to use in the QPPQ method.
    n_realizations (int): The number of realizations to generate.
    start_year (int): The start year for the prediction.
    end_year (int): The end year for the prediction.
    
    Returns:
    tuple: A tuple containing the station ID and the predicted flows.
    """
    
    datetime_index= pd.date_range(f'{start_year}-01-01', 
                                  f'{end_year}-12-31', freq='D')
    
    # Get station ID and NHM segment ID
    station_id = row.Index
    test_station_no = row.Index  
    
    # Get subcatchment flows
    test_subcatchment_stations = gauge_subcatchments[test_station_no]
        
    # Coordinates of prediction site
    coords= (unmanaged_gauge_meta.loc[test_station_no, 'long'],
            unmanaged_gauge_meta.loc[test_station_no, 'lat'])
    
    
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
        assert(Q_hat['1'].shape[0] == len(datetime_index)), f'Q_hat and datetime_index must be the same length but have sizes {Q_hat["1"].shape} and {len(datetime_index)}.'
        
    return (station_id, Q_hat)
