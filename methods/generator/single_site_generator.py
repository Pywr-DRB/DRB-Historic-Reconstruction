
import pandas as pd
import numpy as np

from .QPPQModel import StreamflowGenerator


def generate_single_gauge_reconstruction(Q_unmanaged, 
                                         station_id, 
                                         fdc_prediction,
                                         fdc_quantiles,
                                         long_lat,
                                         unmanaged_gauge_meta,
                                         start_year=1945, 
                                         end_year=2022, 
                                         N_REALIZATIONS=1,
                                         K=5, 
                                         debugging=False):
    """_summary_

    Args:
        Q_unmanaged (pd.DataFrame): Dataframe of unmanaged gauge streamflow timeseries with datetime index and USGS-gauge-number column names.
        fdc_prediction (array): Array of flow values at specified fdc_quantiles. 
        long_lat (tuple): (longitude, latitude) of the gauge to be reconstructed.
        station_id (str): USGS gauge number as a string.
        start_year (int, optional): _description_. Defaults to 1945.
        end_year (int, optional): _description_. Defaults to 2022.
        N_REALIZATIONS (int, optional): _description_. Defaults to 1.
        donor_fdc (str, optional): _description_. Defaults to 'nhmv10'.
        K (int, optional): _description_. Defaults to 5.
        regression_nhm_inflow_scaling (bool, optional): _description_. Defaults to True.
        remove_mainstem_gauges (bool, optional): _description_. Defaults to True.
        debugging (bool, optional): _description_. Defaults to False.
    """
    
    ### Setup
    ## Assertions
    assert(len(fdc_quantiles) == len(fdc_prediction)), 'fdc_quantiles and fdc_prediction must be the same length.'
    assert(len(long_lat) == 2), 'long_lat must be a tuple of length 2.'
    assert(station_id in unmanaged_gauge_meta.index), 'station_id not found in unmanaged_gauge_meta.index.'
    
    # Range for desired reconstruciton
    max_daterange = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', 
                                  freq='D')

    # Number of realizations for QPPQ KNN method                
    probabalistic_QPPQ = True if (N_REALIZATIONS > 1) else False
    
    
    # Pull station data if present
    if f'USGS-{station_id}' in Q_unmanaged.columns:    
        Q_train= Q_unmanaged.drop(f'USGS-{station_id}', axis=1)
    else:
        Q_train= Q_unmanaged.copy()
    
    # Initialize storage
    Q_predicted = pd.DataFrame(index = max_daterange, columns=[station_id])
    ensemble_Q_predicted = {}
    ensemble_Q_predicted[station_id] = {}
    
    # Partition the date range into yearly subsets (this can be improved, to allow non-complete years)
    N_YEARS = int(np.floor(len(max_daterange)/365))
    starts = [f'{start_year+i}-01-01' for i in range(N_YEARS)]
    ends = [f'{start_year+i}-12-31' for i in range(N_YEARS)]
    daterange_subsets = np.vstack([starts, ends]).transpose()
    assert(pd.to_datetime(daterange_subsets[-1,-1]).date() <= Q_unmanaged.index.max().date()), 'The historic data must be more recent than QPPQ daterange.'

    ## QPPQ prediction
    # Generate 1 year at a time, to maximize the amount of data available for each years QPPQ
    for real in range(N_REALIZATIONS):
        print(f'Generating realization {real+1} of {N_REALIZATIONS} at site {station_id}.')
        for i, daterange_subset_i in enumerate(daterange_subsets):
            
            dates = [str(d) for d in daterange_subset_i]
                
            # Pull gauges that have flow during daterange
            Q_subset = Q_train.loc[dates[0]:dates[1], :].dropna(axis=1)
            subset_sites = [f'{i.split("-")[1]}' for i in Q_subset.columns]
            unmanaged_gauge_meta_subset = unmanaged_gauge_meta.loc[subset_sites, :]
            unmanaged_gauge_meta_subset.index = Q_subset.columns
            
            ## Initialize the model
            model = StreamflowGenerator(K= K,
                                        observed_flow = Q_subset, 
                                        full_observed_flow = Q_train,
                                        fdc_quantiles= fdc_quantiles,
                                        observation_locations=unmanaged_gauge_meta_subset,
                                        probabalistic_sample = probabalistic_QPPQ)            
            # Run QPPQ
            Q_pub = model.predict_streamflow(long_lat, fdc_prediction).values.flatten()                        
                
            # Store results
            Q_predicted.loc[dates[0]:dates[1], station_id] = Q_pub
            assert(Q_predicted.loc[dates[0]:dates[1], station_id].isna().sum() == 0), f'There are NAs in the reconstruction for gauge {station_id}'
        
        # store realization
        ensemble_Q_predicted[f'realization_{real}'] = Q_predicted.values.flatten()
    
    if N_REALIZATIONS > 1:
        return ensemble_Q_predicted
    else:
        return Q_predicted