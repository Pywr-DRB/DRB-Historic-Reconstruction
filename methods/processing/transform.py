
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
    
    if transform == 'rolling':
        if aggregation_type == 'mean':
            data = data.rolling(window=window).mean()
        elif aggregation_type == 'sum':
            data = data.rolling(window=window).sum()
        data = data.iloc[window-1:].dropna()
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
    for key in data.keys():
        data[key] = transform_flow(data[key], transform, window, 
                                   aggregation_type, aggregation_length)
    return data


def transform_results_dict_flow(data, transform, window, 
                                aggregation_type, aggregation_length):
    """Transforms a dictionary of results dataframes with flow data.
    """
    assert(type(data) == dict), 'data must be a dictionary'
    assert(transform in ['rolling', 'aggregation']), 'transform must be rolling or aggregation'
    
    for key in data.keys():
        print(f'Transforming {key} flow')
        if 'ensemble' in key:
            data[key] = transform_ensemble_flow(data[key], transform, window, 
                                                aggregation_type, aggregation_length)
        else:
            data[key] = transform_flow(data[key], transform, window, 
                                    aggregation_type, aggregation_length)
    return data
