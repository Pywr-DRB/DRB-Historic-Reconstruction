import numpy as np
import pandas as pd


bias_method_labels = {
    'method_1': 'Difference',
    'method_2': '% Difference',
    'method_3': 'Log Difference',
    'method_4': '% Log Difference',
    'method_5': 'Difference per Area',
    'method_6': '% Difference per Area',
    'method_7': 'Log Difference per Area',
    'method_8': '% Log Difference per Area'
}

def empirical_cdf(data, quantiles):
    """Applies the empirical CDF transformation to each column of the data."""
    N,M = data.shape
    for i in range(M):
        # drop na from column 
        y = data[:,i][~np.isnan(data[:,i])]
        y = y[y>0]
        cdf_y = np.quantile(y, quantiles).astype(float)
         
        if i == 0:
            cdf = cdf_y
        else:
            cdf = np.vstack((cdf, cdf_y))
    return cdf


def calculate_quantile_biases(X_observed, X_modeled, method, area=None,
                              precip_observation=False):
    """
    Calculate biases between observed and modeled data using specified method.
    :param X_observed: DataFrame of observed values with site IDs as index and quantiles as columns
    :param X_modeled: DataFrame of modeled values with same structure as X_observed
    :param method: Bias calculation method as an integer (1-11)
    :param area: DataFrame with 'area' column and site IDs as index, if applicable
    :return: DataFrame of calculated biases
    """
    if area is not None:
        assert 'area' in area.columns, "Area DataFrame must contain 'area' column"
        assert (X_observed.index == area.index).all(), "Indices of X_observed and area must match"
        A = area['area']
    else:
        A = 1

    # Multiply times area if precip_observation
    if precip_observation:
        X_observed = X_observed.multiply(A, axis=0)

    if method == 1:
        return X_modeled - X_observed
    elif method == 2:
        return (X_modeled - X_observed) / X_observed
    elif method == 3:
        return np.log(X_modeled) - np.log(X_observed)
    elif method == 4:
        return (np.log(X_modeled) - np.log(X_observed)) / np.log(X_observed)
    elif method == 5:
        diff = X_modeled - X_observed
        return diff.divide(A, axis=0)
    elif method == 6:
        return (X_modeled - X_observed) / (X_observed.multiply(A, axis=0))
    elif method == 7:
        return np.log(X_modeled.divide(A, axis=0)) - np.log(X_observed.divide(A, axis=0))
    elif method == 8:
        return (np.log(X_modeled.divide(A, axis=0)) - np.log(X_observed.divide(A, axis=0))) / np.log(X_observed.divide(A, axis=0))
    elif method == 9:
        return np.log(X_modeled - X_observed)
    elif method == 10:
        return np.log((X_modeled - X_observed) / X_observed)
    elif method == 11:
        return np.log((X_modeled - X_observed) / A)
    else:
        raise ValueError("Invalid method number")

def correct_quantile_biases(X_modeled, X_biases, method, area=None):
    """
    Adjust modeled values based on biases using the inverse of specified method.
    :param X_modeled: DataFrame of modeled values with site IDs as index and quantiles as columns
    :param X_biases: DataFrame of calculated biases with same structure as X_modeled
    :param method: Bias correction method as an integer (1-11)
    :param area: DataFrame with 'area' column and site IDs as index, if applicable
    :return: DataFrame of adjusted modeled values
    """
    if area is not None:
        assert 'area' in area.columns, "Area DataFrame must contain 'area' column"
        # assert (X_modeled.index == area.index).all(), "Indices of X_modeled and area must match"
        A = area['area'].values[0]
    else:
        A = 1

    if method == 1:
        return X_modeled - X_biases
    elif method == 2:
        return X_modeled / (1 + X_biases)
    elif method == 3:
        return np.exp(np.log(X_modeled) - X_biases)
    elif method == 4:
        return np.exp(np.log(X_modeled) - X_biases.multiply(np.log(X_modeled), axis=0))
    elif method == 5:
        return X_modeled - X_biases.multiply(A, axis=0)
    elif method == 6:
        return X_modeled / (1 + X_biases.multiply(A, axis=0))
    elif method == 7:
        return np.exp(np.log(X_modeled) - X_biases) * A
    elif method == 8:
        return np.exp(np.log(X_modeled) - X_biases.multiply(np.log(X_modeled.divide(A, axis=0)), axis=0)) * A
    elif method == 9:
        return np.exp(np.log(X_modeled) - X_biases) + X_modeled
    elif method == 10:
        return X_modeled / np.exp(-X_biases) + X_modeled
    elif method == 11:
        return np.exp(np.log(X_modeled) - X_biases) * A + X_modeled
    else:
        raise ValueError("Invalid method number")