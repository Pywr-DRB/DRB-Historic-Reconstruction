"""
TJA
"""

import numpy as np
import pandas as pd

import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PUB_Generator.utils.NN_utils import select_features

################################################################################

def pd_filter_nans(x):
    # first remove nan columns
    nan_column = x.columns[x.isna().any()].to_list()
    x = x.drop(nan_column, axis = 1)

    # remove nan observations
    nan_index = x.index[x.isna().any(axis = 1)].to_list()
    x = x.drop(nan_index, axis = 0)
    return x

################################################################################

def remove_outlier_gages(x, threshold = 10):
    scaler = StandardScaler()
    std_x = scaler.fit_transform(x)
    check = (std_x<threshold).any(axis = 1)
    x = x.loc[check==True,:]
    return x

################################################################################

def remove_managed_basins(x):
    """
    Removes managed basins. Inputs must be pd.DataFrame.
    """
    labels = ["CAT_RESERVOIR", "CAT_NORM_STORAGE2013", "CAT_NID_STORAGE2013", "CAT_NDAMS2013"]
    for label in labels:
        if label in x.columns:
            managed = x.index[x[label] > 0]
            x = x.drop(managed, axis = 0)
        else:
            pass
    return x



################################################################################

def filter_drb_sites(x, site_ids = None):
    sdir = 'C:/Users/tjame/Desktop/Research/DRB/DRB_water_management/DRB_spatial/DRB_shapefiles'
    crs = 4386
    if type(x) == np.ndarray:
        x = pd.DataFrame(x)

    if site_ids is not None:
        x.index = site_ids

    drb_boarder = gpd.read_file(f'{sdir}/drb_bnd_polygon.shp')
    drb_boarder = drb_boarder.to_crs(crs)
    x_all = gpd.GeoDataFrame(x, geometry = gpd.points_from_xy(x.long, x.lat, crs = crs))
    x_filtered = gpd.clip(x_all, drb_boarder)
    return x_filtered

################################################################################

def prepare_data(x, y, xP, Q,
                use_features = 'strong',
                remove_managed = True,
                remove_outliers = True,
                outlier_threshold = 6,
                normalize = False,
                standardize = False):

    # Remove nans
    x, y, Q = pd_filter_nans(x, y, Q)

    # Remove managed basins
    if remove_managed:
        x, y, Q = remove_managed_basins(x, y, Q)

    x, xP = select_features(x, use_features, xPr = xP)

    # Filter strong inputs by removing outlier locations
    if remove_outliers:
        x, y, Q = remove_outlier_gages(x, y, Q, threshold = outlier_threshold)

    if normalize:
        norm_mod = MinMaxScaler().fit(x)
        norm_x = norm_mod.transform(x)
        norm_xP = norm_mod.transform(xP)
        return norm_x, x, y, Q
    elif standardize:
        pass
    else:
        return
