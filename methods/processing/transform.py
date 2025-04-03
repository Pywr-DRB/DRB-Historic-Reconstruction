
import numpy as np
import pandas as pd


def transform_flow(data, transform = 'rolling', window = None, 
                   aggregation_type = 'mean', aggregation_length = None):
    """Transforms a single dataframe of flow data using pandas rolling or
    reample methods.
    """
    assert(transform in ['rolling', 'aggregation']), 'transform must be rolling or aggregation'
    assert(aggregation_type in ['mean', 'sum']), 'aggregation_type must be mean or sum'
    assert((window is not None) or (aggregation_length is not None)), 'window or aggregation_length must be specified'
    
    data = data.select_dtypes(include=[np.number])
    data = data.astype(float)
    
    if transform == 'rolling':
        if aggregation_type == 'mean':
            data = data.rolling(window=window).mean()
        elif aggregation_type == 'sum':
            data = data.rolling(window=window).sum()
    elif transform == 'aggregation':
        if aggregation_type == 'mean':
            data = data.resample(aggregation_length).mean()
        elif aggregation_type == 'sum':
            data = data.resample(aggregation_length).sum()
    return data


def transform_ensemble_flow(data, transform, window, 
                            aggregation_type, aggregation_length):
    """Transforms an ensemble of data one key/dataframe at a time.
    """
    new_data = {}
    for key in data.keys():
        vals = data[key]
        new_data[key] = transform_flow(vals, transform, window, 
                                   aggregation_type, aggregation_length)
    return new_data


def transform_results_dict_flow(data, transform, window, 
                                aggregation_type, aggregation_length):
    """Transforms a dictionary of results dataframes with flow data.
    """
    assert(type(data) == dict), 'data must be a dictionary'
    assert(transform in ['rolling', 'aggregation']), 'transform must be rolling or aggregation'
    new_data = {}
    for key in data.keys():
        print(f'Transforming {key} flow')
        if 'ensemble' in key:
            new_data[key] = transform_ensemble_flow(data[key], transform, window, 
                                                aggregation_type, aggregation_length)
        else:
            new_data[key] = transform_flow(data[key], transform, window, 
                                    aggregation_type, aggregation_length)
    return new_data



def streamflow_to_nonexceedance(Q, quantiles, 
                                log_fdc_interpolation=True,
                                Q_full=None):
    """Transforms flow timeseries to non-exceedance probability (NEP) timeseries.

    Args:
        Q (array): The flow to be transformed
        Qobs_full (array, optional): The full flow for the donor site, used to generated FDC. Defaults to None and uses Q.

    Returns:
        array: The non-exceedance timeseries for observed flow.
    """
    Q = Q if Q_full is None else Q_full
    
    # Get FDC from observed flow
    if log_fdc_interpolation:
        fdc = np.quantile(np.log(Q), quantiles)
        Q = np.log(Q)
    else:
        fdc = np.quantile(Q, quantiles)
        
    # Translate flow to NEP
    nep = np.interp(Q, fdc, quantiles, 
                    right = quantiles[-1], left = quantiles[0])
    return nep

def nonexceedance_to_streamflow(nep_timeseries,
                                quantiles,
                                fdc,
                                log_fdc_interpolation=True):
    
    # # The bound_percentage will determine how much (+/- %) random flow 
    # # is sampled when NEP > fdc_quantiles[-1] or NEP < fdc_quantiles[0] 
    bound_percentage = 0.01
    # high_flow_bound = np.random.uniform(self.predicted_fdc[-1], 
    #                                     self.predicted_fdc[-1] + bound_percentage*self.predicted_fdc[-1])
    # low_flow_bound = np.random.uniform(self.predicted_fdc[0] - bound_percentage*self.predicted_fdc[0], 
    #                                    self.predicted_fdc[0])
    
    if log_fdc_interpolation:
        fdc = np.log(fdc)
    
    Q = np.interp(nep_timeseries, quantiles, fdc, 
                right = fdc[-1], left = fdc[0])
    
    if log_fdc_interpolation:
        Q = np.exp(Q)
    return Q