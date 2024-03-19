import numpy as np

def calculate_lagged_correlation(x, y, lag):
    """
    

    Args:
        x (np.array): 1D array of values
        y (np.array): 1D array of values
        lag (int): Number of time steps to lag x by
    """
    # Calculate the correlation between x and y
    return np.corrcoef(x[:lag], y[lag:])[0, 1]

####################################################################################################

def calculate_lag_with_max_correlation(x, y, max_lag):
    """
    

    Args:
        x (np.array): 1D array of values
        y (np.array): 1D array of values
        max_lag (int): Maximum number of time steps to lag x by
    """
    # Get the correlation between x and y for each lag
    correlations = [calculate_lagged_correlation(x, y, lag) for lag in range(max_lag)]
    # Find the lag that gives the maximum correlation
    return np.argmax(correlations)

####################################################################################################

