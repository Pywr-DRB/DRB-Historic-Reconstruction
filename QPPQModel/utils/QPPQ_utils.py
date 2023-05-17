"""
Trevor Amestoy

Contains supplementary functions for the NN FDC generation process.

Includes:
1. A Nash-Sutcliffe Efficiency (NSE) calculation
2. A calculation of the FDC bins


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


################################################################################

def interpolate_FDC(nep, fdc, quants):
    """
    Performs linear interpolation of discrete FDC values to find flow at a NEP.

    Parameters
    ----------
    nep :: float
        Non-exceedance probability at a specific time.
    fdc :: array
        Array of discrete FDC points
    quants :: array
        Array of quantiles for discrete FDC.

    Returns
    -------
    flow :: float
        A single flow value given the NEP and FDC points.
    """
    tol = 0.0000001
    assert(len(fdc) == len(quants)), f'FDC and quants should be same length, but are {len(fdc)} and {len(quants)}.'
    if nep == 0:
        nep = np.array(tol)
    elif nep == 1.0:
        nep = np.array(0.999)
    sq_diff = (quants - nep)**2

    # Index of nearest discrete NEP
    ind = np.argmin(sq_diff)

    # Handle edge-cases
    if nep <= quants[0]:
        return fdc[0] - np.random.uniform(0, (fdc[1]-fdc[0]))
    elif nep >= quants[-1]:
        return fdc[-1] + np.random.uniform(0, (fdc[-1]-fdc[-2]))
    
    # Check which side of nearest quant the predicted NEP sits
    if quants[ind] <= nep:
        flow_range = fdc[ind:ind+2]
        nep_range = quants[ind:ind+2]
    else:
        flow_range = fdc[ind-1:ind+1]
        nep_range = quants[ind-1:ind+1]


    if len(flow_range) == 0:
        print(f'Error! NEP: {nep} and sq_dff {sq_diff} and ind {ind}')
        return None
    else:
        slope = (flow_range[1] - flow_range[0])/(nep_range[1] - nep_range[0])
        flow = flow_range[0] + slope*(nep_range[1] - nep)
    return flow

################################################################################
def innerproduct(X,Z=None):
    if Z is None: # case when there is only one input (X)
        Z=X;
    G = X@Z.T
    return G

################################################################################

def l2distance(X,Z=None):
    # Computes the Euclidean distance matrix.
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # FORMULA
    # $$D^2 = S - 2G + R$$

    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    G = innerproduct(X, Z = Z)

    # Make matrices S and R
    S = np.zeros((n, m))
    for row in range(n):
        S[row, :] = np.dot(X[row], X[row].T)

    R = np.zeros((n, m))
    for col in range(m):
        R[:,col] = np.dot(Z[col], Z[col].T)

    D_sqr = S - 2*G + R

    D_sqr = np.where(D_sqr<0.0, 0.0, D_sqr)

    D = np.sqrt(D_sqr)

    return D

################################################################################

def find_KNN(xTr,xTe,k):
    """
    Finds the k nearest neighbors of xTe in xTr.

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    # Calculate L2 norm between each
    alldists = l2distance(xTr, xTe)

    # Rank distance of features (columns)
    ranked_indices = np.argsort(alldists, axis = 0)
    ranked_dists = np.sort(alldists, axis = 0)

    # Return the k distances and indices of interest
    return ranked_indices[:k, :], ranked_dists[:k, :]



################################################################################
def find_continuous_FDC(data, quants):
    """
    Calculates FDC values as specified `quants` or non-exceedance probabilites.
    """
    flows = np.quantile(data, quants, axis = 1).T
    return flows


###############################################################################

def find_NEPs(hist_data, flow):
    """
    Finds the non-exceedance probability (NEP) for `flow` given hist_data
    timeseries.
    """

    quants = np.linspace(0,1,200)
    fdc_data = find_continuous_FDC(hist_data, quants)

    # Find nearest FDC value
    diff = fdc_data - flow

    small_diff = np.argsort(abs(diff), axis = 1)[:,0]
    nep = quants[small_diff]
    return nep

###############################################################################

def NSE_error(obs, pred):
    assert(len(obs) == len(pred)), "Predicted and observed must be same length."
    return 1- (np.sum((obs-pred)**2) / np.sum((obs-np.mean(obs))**2))

################################################################################


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    return x_vals, y_vals
