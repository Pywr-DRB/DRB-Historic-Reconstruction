

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# import scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer

def get_timeseries_dtw(Q1, Q2, distance_metric=euclidean, radius=1):
    """
    Perform Dynamic Time Warping on two time series.

    Parameters:
    Q1, Q2 (array-like): The time series data.
    distance_metric (callable, optional): The distance metric for DTW.
    radius (int, optional): The radius parameter for FastDTW.

    Returns:
    list: The warping path for aligning Q1 to Q2.
    """
    distance, path = fastdtw(Q1, Q2, dist=distance_metric, radius=radius)
    return path

def align_timeseries_from_dtw(Q_up, Q_down, distance_metric=euclidean, radius=1):
    """
    Align one time series to another using DTW.

    Parameters:
    Q_up (array-like): The upstream flow time series data.
    Q_down (array-like): The downstream flow time series data.
    distance_metric (callable, optional): The distance metric for DTW.
    radius (int, optional): The radius parameter for FastDTW.

    Returns:
    array: The aligned upstream time series.
    """
    path = get_timeseries_dtw(Q_down, Q_up, distance_metric, radius)
    aligned_up = np.zeros_like(Q_down)
    for i, j in path:
        aligned_up[i] = Q_up[j]
    return aligned_up

def standardize_flow(Q, method='robust'):
    """
    Standardize the flow data using various methods.

    Parameters:
    Q (array-like): The time series data.
    method (str): The standardization method ('min_max', 'robust', 'log', etc.).

    Returns:
    array: The standardized time series.
    """

    if method == 'min_max':
        scaler = MinMaxScaler()
        return scaler.fit_transform(Q.reshape(-1, 1)).flatten(), scaler
    elif method == 'robust':
        scaler = RobustScaler()
        return scaler.fit_transform(Q.reshape(-1, 1)).flatten(), scaler
    elif method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(Q.reshape(-1, 1)).flatten(), scaler
    elif method == 'power':
        scaler = PowerTransformer()
        return scaler.fit_transform(Q.reshape(-1, 1)).flatten(), scaler
    else:
        raise ValueError("Unsupported standardization method")

def remove_upstream_flow(Q_down, Q_up, 
                         align_method='dtw', 
                         standardize=False, standardize_method='min_max', **kwargs):
    """
    Remove the upstream flow from the downstream flow.

    Parameters:
    Q_down, Q_up (array-like): The downstream and upstream flow data, respectively.
    align_method (str): The method for alignment ('dtw', etc.).
    standardize (bool): Whether to standardize the time series before alignment.
    standardize_method (str): The method for standardization if applicable.

    Returns:
    array: The downstream flow with the upstream flow removed.
    """
    Q_up = Q_up.copy()
    Q_down = Q_down.copy()
    if standardize:
        Q_down_std, scaler = standardize_flow(Q_down, standardize_method)
        Q_up_std, scaler = standardize_flow(Q_up, standardize_method)
    else:
        Q_down_std, Q_up_std = Q_down.flatten(), Q_up.flatten()

    if align_method == 'dtw':
        # Make arrays [N, 2] for fastdtw
        Q_down_std = np.stack([np.arange(len(Q_down_std)), Q_down_std], axis=1)
        Q_up_std = np.stack([np.arange(len(Q_up_std)), Q_up_std], axis=1)
        
        aligned_up = align_timeseries_from_dtw(Q_up_std, Q_down_std, **kwargs)
        aligned_Q_up = aligned_up[:, 1]
    elif align_method == 'lag':
        aligned_up = np.roll(Q_up_std, -kwargs['lag'])
    elif align_method is None:
        aligned_Q_up = Q_up_std
        
    # Reverse standardization
    if standardize:
        aligned_Q_up = scaler.inverse_transform(aligned_Q_up.reshape(-1, 1)).flatten()

    # print(aligned_Q_up.shape, Q_down.shape)
    adjusted_flow = Q_down - aligned_Q_up

    return adjusted_flow


        