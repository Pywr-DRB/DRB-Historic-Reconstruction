import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
from scipy.stats import genextreme, pearson3
from scipy.optimize import curve_fit
from scipy.stats import genextreme as gev
from scipy.stats import pearson3
from sklearn.preprocessing import StandardScaler
import spei as si

def drought_metric_scatter_plot(drought_metrics):
    fig, ax = plt.subplots(figsize = (7,6))
    p = ax.scatter(drought_metrics['severity'], -drought_metrics['magnitude'],
            c= drought_metrics['duration'], cmap = 'viridis_r', s=100)

    plt.colorbar(p).set_label(label = 'Drought Duration (days)',size=15)
    plt.xlabel(r'Severity ($Minimum SSI_{6}$)', fontsize = 15)
    plt.ylabel(r'Magnitude (Acc. Deficit)', fontsize = 15)
    plt.title(f'Historic Droughts', fontsize = 16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    return



def plot_ssi(ssi, bound = 3.0, figsize= (8, 4), ax = None, gradient = False):
    """Plot the standardized index values as a time series. 
    Modified from https://github.com/martinvonk/SPEI/blob/main/src/spei/plot.py

    Parameters
    ----------
    si : pandas.Series
        Series of the standardized index
    bound : int, optional
        Maximum and minimum ylim of plot
    figsize : tuple, optional
        Figure size, by default (8, 4)
    ax : matplotlib.Axes, optional
        Axes handle, by default None which create a new axes

    Returns
    -------
    matplotlib.Axes
        Axes handle
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(ssi, color="k", lw = 1, label="SSI")
    ax.axhline(0, linestyle="--", color="k")

    nmin = -bound
    nmax = bound
    # Classify droughts
    in_drought = False
    in_critical_drought = False

    droughts = np.zeros_like(ssi.values)
    drought_days = []
    drought_counter = 0
    for ind in range(len(droughts)):
        if ssi.values[ind] < 0:
            in_drought = True
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
                droughts[drought_days] = 1
        else:
            in_drought = False
            in_critical_drought = False
            drought_days = [] 
            drought_counter = 0
        
    if gradient:
        droughts = ssi.to_numpy(dtype=float, copy=True)
        droughts[droughts > 0] = 0
        nodroughts = ssi.to_numpy(dtype=float, copy=True)
        nodroughts[nodroughts < 0] = 0
        x, y = np.meshgrid(ssi.index, np.linspace(nmin, nmax, 100))
        ax.contourf(
            x, y, y, cmap=plt.get_cmap("seismic_r"), levels=np.linspace(nmin, nmax, 100)
        )
        ax.fill_between(x=ssi.index, y1=droughts, y2=nmin, color="w")
        ax.fill_between(x=ssi.index, y1=nodroughts, y2=nmax, color="w")
    else:
        ax.fill_between(x=ssi.index, y1=nmax, y2=nmin, where=(droughts>0), color='red', alpha=0.5, interpolate=False, label = 'Drought Period')
    ax.set_ylim(nmin, nmax)
    return ax