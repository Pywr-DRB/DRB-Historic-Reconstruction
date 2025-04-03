
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, DATA_DIR, FIG_DIR, FDC_QUANTILES
from config import BART_PREDICTION_QUANTILES
from config import FEATURE_MUTUAL_INFORMATION_FILE
from config import PYWRDRB_CATCHMENT_FEATURE_FILE, PYWRDRB_SCALED_CATCHMENT_FEATURE_FILE
from config import USGS_CATCHMENT_FEATURE_FILE, USGS_SCALED_CATCHMENT_FEATURE_FILE
from config import NHM_FDC_BIAS_FILE, NHM_FDC_PERCENT_BIAS_FILE

from methods.bias_correction.prep import calculate_fdc
from methods.bias_correction.prep import calculate_quantile_biases
from methods.bias_correction.prep import calculate_flow_metrics
from methods.bias_correction.prep import select_features_using_mutual_information

from methods.load.data_loader import Data



### Prep bias correction inputs
def load_all_potential_bias_correction_inputs(sitetype):
    data_loader = Data()
    
    
    ### Re-load important info for the sitetype
    
    station_catchments = data_loader.load(datatype='catchments', sitetype=sitetype)
    catchment_nldi = data_loader.load(datatype='nldi', sitetype=sitetype)
    
    if sitetype == 'pywrdrb':
        catchment_metadata = data_loader.load(datatype='metadata', sitetype='pywrdrb')
        site_ids = catchment_metadata['name'].values
        Q_nhm = data_loader.load(datatype='streamflow', sitetype='pywrdrb', flowtype='nhm')
        Q_nhm = Q_nhm.loc[:, site_ids]
        
    elif sitetype == 'usgs':
        catchment_metadata = data_loader.load(datatype='metadata', sitetype='diagnostic')
        site_ids = catchment_metadata['site_no'].values
        catchment_metadata.set_index('site_no', inplace=True)
        Q_nhm = data_loader.load(datatype='streamflow', sitetype='diagnostic', flowtype='nhm')
        Q_nhm = Q_nhm.loc[:, site_ids]
    
    
    ### Calc flow duration curves
    fdc = {}
    fdc['nhm'] = data_loader.load(datatype='fdc', 
                                  sitetype=sitetype,
                                  flowtype='nhm', 
                                  timescale='daily')
    
    monthly_fdc = {}
    monthly_fdc['nhm'] = data_loader.load(datatype='fdc', 
                                  sitetype=sitetype,
                                  flowtype='nhm', 
                                  timescale='monthly')
    
    
    ### Get difference between daymet prcp FDCs and NHM streamflow FDCs
    daymet_mean = {}
    
    daymet_mean['monthly'] = data_loader.load(datatype='daymet', 
                                  sitetype=sitetype, 
                                  timescale='monthly')
    daymet_mean['monthly'] = daymet_mean['monthly'].loc[:, Q_nhm.columns]
    
    monthly_fdc['P_mean'] = calculate_fdc(daymet_mean['monthly'], np.array(FDC_QUANTILES))
    monthly_fdc['P_mean'].columns = monthly_fdc['P_mean'].columns.astype(str)
    
    monthly_mean_prcp_nhm_diff = calculate_quantile_biases(monthly_fdc['P_mean'], 
                                                                        monthly_fdc['nhm'], 
                                                                        area=station_catchments[['area_km2']],
                                                                        precip_observation=True)
    
    
    
    # calculate flow stat metrics
    nhm_flow_metrics = calculate_flow_metrics(Q_nhm)
    
    ### All potential inputs
    x_fdc = fdc['nhm'].loc[site_ids, ['0.1', '0.5', '0.9']].copy()
    x_fdc.columns = x_fdc.columns + '_nhm_fdc'
    
    x_full = pd.concat([x_fdc, 
                        catchment_nldi.loc[site_ids,:],
                        catchment_metadata.loc[site_ids, ['long', 'lat']],
                        ], axis=1)
    
    x_prcp_diff = monthly_mean_prcp_nhm_diff.loc[site_ids, :].copy()
    x_prcp_diff.columns = [f'{c}_monthly_prcp_diff' for c in x_prcp_diff.columns]
    x_prcp_diff = x_prcp_diff.loc[:, [f'{q}_monthly_prcp_diff' for q in [0.1, 0.5, 0.9]]]
    x_full = pd.concat([x_full, x_prcp_diff], axis=1)
    
    # flow metrics
    x_full = pd.concat([x_full, nhm_flow_metrics.loc[site_ids,:]], axis=1)
    
    # set all columns names as str
    x_full.columns = x_full.columns.astype(str)
    x_full = x_full.dropna(axis=1)
    return x_full





if __name__ == "__main__":

    X = {}
    y = {}
    
    for sitetype in ['pywrdrb', 'usgs']:
        X[sitetype] = load_all_potential_bias_correction_inputs(sitetype)
    
    
    ## PCA
    x_train_full = X['usgs'].copy()
    x_test_full = X['pywrdrb'].copy()
    
    scaler = StandardScaler()
    scaler.fit(x_train_full)
    x_train_scaled = scaler.transform(x_train_full)
    x_train_scaled = pd.DataFrame(x_train_scaled, index=x_train_full.index, columns=x_train_full.columns)

    pca = PCA()
    pca.fit(x_train_scaled)
    x_train_pca = pca.transform(x_train_scaled)
    x_train_pca = pd.DataFrame(x_train_pca, index=x_train_full.index)
    x_train_pca.columns = [f'PC{i+1}' for i in range(x_train_pca.shape[1])]
    x_train_pca = x_train_pca.loc[:, [f'PC{i+1}' for i in range(10)]]

    x_train_full = pd.concat([x_train_full, x_train_pca], axis=1)
    x_train_full.index.name = 'site_no'
    
    x_train_scaled = pd.concat([x_train_scaled, x_train_pca], axis=1)
    x_train_scaled.index.name = 'site_no'

    # transform pywrdrb node (test) data
    x_test_scaled = scaler.transform(x_test_full)
    x_test_scaled = pd.DataFrame(x_test_scaled, index=x_test_full.index, columns=x_test_full.columns)
    
    x_test_pca = pca.transform(x_test_scaled)
    x_test_pca = pd.DataFrame(x_test_pca, index=x_test_full.index)
    x_test_pca.columns = [f'PC{i+1}' for i in range(x_test_pca.shape[1])]
    x_test_pca = x_test_pca.loc[:, [f'PC{i+1}' for i in range(10)]]


    x_test_full = pd.concat([x_test_full, x_test_pca], axis=1)
    x_test_full.index.name = 'name'
    x_test_scaled = pd.concat([x_test_scaled, x_test_pca], axis=1)
    x_test_scaled.index.name = 'name'
    
    x_train_full.to_csv(USGS_CATCHMENT_FEATURE_FILE)
    x_train_scaled.to_csv(USGS_SCALED_CATCHMENT_FEATURE_FILE)
    x_test_full.to_csv(PYWRDRB_CATCHMENT_FEATURE_FILE)
    x_test_scaled.to_csv(PYWRDRB_SCALED_CATCHMENT_FEATURE_FILE)
    
    
    # load bias data (y)    
    bias = pd.read_csv(NHM_FDC_BIAS_FILE, index_col=0, dtype={'site_no':str})
    pbias = pd.read_csv(NHM_FDC_PERCENT_BIAS_FILE, index_col=0, dtype={'site_no':str})
    pbias.columns = pbias.columns.astype(float)
    
    # scale y
    y_train = pbias.loc[x_train_full.index, BART_PREDICTION_QUANTILES]
    

    feature_mi_scores = select_features_using_mutual_information(x_train_full, y_train)

    # Save mutual information scores
    feature_mi_scores.to_csv(FEATURE_MUTUAL_INFORMATION_FILE)
    