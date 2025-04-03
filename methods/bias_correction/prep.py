import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression

from config import NHM_FDC_PERCENT_BIAS_FILE, NHM_FDC_BIAS_FILE
from config import PYWRDRB_CATCHMENT_FEATURE_FILE, PYWRDRB_SCALED_CATCHMENT_FEATURE_FILE
from config import USGS_CATCHMENT_FEATURE_FILE, USGS_SCALED_CATCHMENT_FEATURE_FILE

def load_bias_correction_training_outputs(percent_bias=True):
    df = pd.read_csv(NHM_FDC_PERCENT_BIAS_FILE if percent_bias else NHM_FDC_BIAS_FILE, 
                     index_col=0, dtype={'site_no':str})
    df.columns = df.columns.astype(float)
    return df    


def load_bias_correction_inputs(scaled=True, pywrdrb=False):
    if pywrdrb:
        file = PYWRDRB_SCALED_CATCHMENT_FEATURE_FILE if scaled else PYWRDRB_CATCHMENT_FEATURE_FILE
        index_name = 'name'
    else:
        file = USGS_SCALED_CATCHMENT_FEATURE_FILE if scaled else USGS_CATCHMENT_FEATURE_FILE
        index_name = 'site_no'
    df = pd.read_csv(file, index_col=0, dtype={index_name:str})
    return df



def select_features_using_mutual_information(x, y):
    mi_scores = np.zeros((x.shape[1], y.shape[1]))
    for i,q in enumerate(y.columns):
        mi = mutual_info_regression(x, y[q].values)
        mi_scores[:, i] = mi
    
    mi_avg_scores = pd.DataFrame(mi_scores.sum(axis=1), 
                                  index=x.columns, 
                                  columns = ['score'])
    sorted_mi_scores = mi_avg_scores.sort_values(ascending=False, by='score')
    
    selected_features = sorted_mi_scores 
    return selected_features

def calculate_fdc(df: pd.DataFrame, quantiles: list) -> pd.DataFrame:
    """
    Calculate Flow Duration Curves (FDCs) for a given streamflow dataframe.
    
    Parameters:
    - df: pd.DataFrame
        Each column represents a different streamflow site, each row is a point in the timeseries.
    - quantiles: list
        A list of quantile values (between 0 and 1) to compute FDCs.
        
    Returns:
    - pd.DataFrame
        FDCs with sites as the index and quantile values as columns.
    """
    fdc_values = {
        site: np.quantile(df[site].dropna(), quantiles)
        for site in df.columns
    }

    fdc_df = pd.DataFrame(fdc_values, index=quantiles).T
    
    return fdc_df


def calculate_quantile_biases(obs, 
                              mod,
                              precip_observation=False,
                              area = None,
                              percentage=True):
    """
    Calculate biases between observed and modeled data using specified method.
    :param X_observed: DataFrame of observed values with site IDs as index and quantiles as columns
    :param X_modeled: DataFrame of modeled values with same structure as X_observed
    :param method: Bias calculation method as an integer (1-11)
    :param area: DataFrame with 'area' column and site IDs as index, if applicable
    :return: DataFrame of calculated biases
    """
    if area is not None:
        # assert 'area' in area.columns, "Area DataFrame must contain 'area' column"
        # assert (obs.index == area.index).all(), "Indices of X_observed and area must match"
        A = area['area_km2']
    else:
        A = 1

    # Multiply times area if precip_observation
    # Convert mm/day to MGD
    if precip_observation:
        obs = obs.multiply(A, axis=0) * 0.264172 
        
    if percentage:
        return (mod - obs) / obs *100
    else:
        return (mod - obs)
    
    
    
def calculate_flow_metrics(Q_df):
    """
    Calculate flow metrics from a flow dataframe.
    """
    ### Get flow metrics of interest
    flow_metrics = ['mean', 'std', 'skew', 'kurt', 
                    'cv', 'high_q', 'low_q', 'min', 'max']

    sites= Q_df.columns
    flow_metric_vals = pd.DataFrame(index=sites, columns=flow_metrics)

    for site in sites:
        Q_site = Q_df.loc[:,site].dropna()
        Q_site = Q_site.replace(0, np.nan)
        Q_site = Q_site.dropna()
                    
        Q_site = np.log(Q_site.astype(float))
        flow_metric_vals.loc[site, 'mean'] = Q_site.mean()
        flow_metric_vals.loc[site, 'std'] = Q_site.std()
        flow_metric_vals.loc[site, 'skew'] = Q_site.skew()
        flow_metric_vals.loc[site, 'kurt'] = Q_site.kurt()
        flow_metric_vals.loc[site, 'high_q'] = Q_site.quantile(0.9)
        flow_metric_vals.loc[site, 'low_q'] = Q_site.quantile(0.1)
        flow_metric_vals.loc[site, 'cv'] = Q_site.std()/Q_site.mean()
        flow_metric_vals.loc[site, 'min'] = Q_site.min()
        flow_metric_vals.loc[site, 'max'] = Q_site.max()    
    return flow_metric_vals