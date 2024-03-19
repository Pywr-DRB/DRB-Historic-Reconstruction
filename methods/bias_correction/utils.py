import numpy as np

def broadcast_X(X, M):
    # Repeat x_area across the second dimension to match the shape of x_prcp
    # The new shape of x_area will be (N, M), where N is the original number of observations
    # and M is the number of quantiles for which we have the predictors.
    x_broadcasted = np.repeat(X, M, axis=1)
    return x_broadcasted
