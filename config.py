"""
Contains specifications/configurations for methods used in reconstructions.
"""
"""
Contains specifications/configurations for methods used in reconstructions.
"""
import numpy as np

from methods.utils.directories import DATA_DIR, PYWRDRB_DIR, FIG_DIR, OUTPUT_DIR


### General
SEED = 711
START_YEAR = 1945
END_YEAR = 2023
DATES = (f'{START_YEAR}-01-01', 
         f'{END_YEAR}-12-31')

GEO_CRS = 4326
CARTESIAN_CRS = 3857

# Discrete FDC quantiles to use throughout methods
FDC_QUANTILES = [0.0003, 0.005, 0.01, 
                 0.05, 0.1, 0.2, 
                 0.3, 0.4, 0.5, 
                 0.6, 0.7, 0.8, 
                 0.9, 0.93, 0.95, 
                 0.97, 0.995, 0.9997]
fdc_quantiles = FDC_QUANTILES



# USGS gauge filtering criteria
MIN_YEARS = 20
MIN_ALLOWABLE_STORAGE = 500000 # Storage in acre-feet which is allowed in a catchment 
USE_MARGINAL_UNMANAGED_FLOWS = False

# Restrict to DRB or broader region
BBOX = (-77.8, 37.5, -74.0, 44.0)
FILTER_DRB = True
BOUNDARY = 'drb' if FILTER_DRB else 'regional'


filter_drb = FILTER_DRB
boundary = BOUNDARY


## Probabilistic QPPQ parameter ranges to test 
N_ENSEMBLE = 1000

K_ENSEMBLE = 5
K_AGGREGATE = 3


AGG_K_MIN = 1
AGG_K_MAX = 8
ENSEMBLE_K_MIN = 3
ENSEMBLE_K_MAX = 8


### Bias correction relevant
predict_quantiles = [0.1, 0.5, 0.9]
BART_PREDICTION_QUANTILES = predict_quantiles

load_selected_features = True
BART_USE_X_SCALED = True
BART_N_FEATURES = 15

BART_PARAMS = {'m': 50,
    'target_accept': 0.99,
    'n_chains': 4, 
    'n_cores': 1,
    'response': 'constant',
    'beta':1.8,
    'alpha':0.95,
    'separate_trees': True,
    'sigma_beta': 0.2,}


BART_REGRESSION_WEIGHTS = np.array([[0.0]*len(FDC_QUANTILES)]*3)
BART_REGRESSION_WEIGHTS[0, 0:6] = 1.0
BART_REGRESSION_WEIGHTS[0, 6] = 0.5
BART_REGRESSION_WEIGHTS[1, 6] = 0.5
BART_REGRESSION_WEIGHTS[1, 7:11] = 1.0
BART_REGRESSION_WEIGHTS[1, 11] = 0.5
BART_REGRESSION_WEIGHTS[2, 11] = 0.5
BART_REGRESSION_WEIGHTS[2, 12:] = 1.0


## MCMC params
MCMC_N_TUNE = 2000
MCMC_N_SAMPLE = 2000
MCMC_TARGET_ACCEPT = 0.92
MCMC_N_CHAINS = 4

## BART parameter sample ranges for grid search
m_range = [20, 150]
beta_range = [1.0, 2.0]
n_grid_samples = 10




## Relevant paths
path_to_nhm_data = '../Input-Data-Retrieval/datasets/NHMv10/'



### Filenames
PREDICTION_LOCATIONS_FILE = f'{DATA_DIR}/prediction_locations.csv'
USGS_GAGE_METADATA_FILE = f'{DATA_DIR}/USGS/{boundary}_usgs_metadata.csv'
USGS_GAGE_CATCHMENT_FILE = f'{DATA_DIR}/NHD/{boundary}_station_catchments.shp'
PYWRDRB_NODE_CATCHMENT_FILE = f'{DATA_DIR}/NHD/{boundary}_pywrdrb_node_catchments.shp'
PYWRDRB_NODE_METADATA_FILE = f'{DATA_DIR}/NHD/{boundary}_pywrdrb_node_metadata.csv'
PYWRDRB_NODE_METADATA_GEO_FILE = f'{DATA_DIR}/NHD/{boundary}_pywrdrb_node_metadata_with_geometry.shp'

ALL_SITE_METADATA_FILE = f'{DATA_DIR}/all_site_metadata.csv'
DIAGNOSTIC_SITE_METADATA_FILE = f'{DATA_DIR}/diagnostic_site_metadata.csv'
UNMANAGED_GAGE_METADATA_FILE = f'{DATA_DIR}/USGS/{boundary}_unmanaged_usgs_metadata.csv'



USGS_NLDI_CHARACTERISTICS_FILE = f'{DATA_DIR}/NLDI/{boundary}_usgs_nldi_catchment_characteristics.csv'
PYWRDRB_NLDI_CHARACTERISTICS_FILE = f'{DATA_DIR}/NLDI/pywrdrb_node_nldi_catchment_characteristics.csv'


PYWRDRB_DRB_CATCHMENT_BOUNDARY_FILE = f'{PYWRDRB_DIR}/DRB_spatial/DRB_shapefiles/drb_bnd_polygon.shp'
DRB_CATCHMENT_BOUNDARY_FILE = f'{DATA_DIR}/NHD/{boundary}_boundary.shp'


COMID_UPSTREAM_GAGE_FILE = f'{DATA_DIR}/NHD/comid_upstream_gauges.json'
STATION_UPSTREAM_GAGE_FILE = f'{DATA_DIR}/NHD/station_upstream_gauges.json'
IMMEDIATE_UPSTREAM_GAGE_FILE = f'{DATA_DIR}/NHD/immediate_upstream_gauges.json'

NID_METADATA_FILE = f'{DATA_DIR}/NID/{boundary}_dam_metadata.shp'
NID_SUMMARY_FILE = f'{DATA_DIR}/NID/{boundary}_catchment_dam_summary.csv'

ALL_USGS_DAILY_FLOW_FILE = f'{DATA_DIR}/USGS/{boundary}_streamflow_daily_usgs_cms.csv'
PYWRDRB_USGS_DAILY_FLOW_FILE = f'{OUTPUT_DIR}/streamflow_daily_usgs_{DATES[0][:4]}_{DATES[1][:4]}_cms.csv'


OBS_DIAGNOSTIC_STREAMFLOW_FILE = f'{OUTPUT_DIR}/obs_diagnostic_gauge_streamflow.csv'
NHM_DIAGNOSTIC_STREAMFLOW_FILE = f'{OUTPUT_DIR}/nhm_diagnostic_gauge_streamflow.csv'

NHM_FDC_BIAS_FILE = f'{OUTPUT_DIR}/nhm_daily_fdc_bias.csv'
NHM_FDC_PERCENT_BIAS_FILE = f'{OUTPUT_DIR}/nhm_daily_fdc_percentage_bias.csv'


PYWRDRB_CATCHMENT_FEATURE_FILE = f'{DATA_DIR}/pywrdrb_all_catchment_features.csv'
PYWRDRB_SCALED_CATCHMENT_FEATURE_FILE = f'{DATA_DIR}/pywrdrb_all_catchment_features_scaled.csv'

USGS_CATCHMENT_FEATURE_FILE = f'{DATA_DIR}/usgs_all_catchment_features.csv'
USGS_SCALED_CATCHMENT_FEATURE_FILE = f'{DATA_DIR}/usgs_all_catchment_features_scaled.csv'

FEATURE_MUTUAL_INFORMATION_FILE = f'{OUTPUT_DIR}/catchment_feature_bias_mutual_information_scores.csv'

PYWRDRB_INPUT_DIR = f'{OUTPUT_DIR}/../pywrdrb_inputs/'
PYWRDRB_OUTPUT_DIR = f'{OUTPUT_DIR}/../pywrdrb_outputs/'


DIAGNOSTIC_BIAS_PREDICTION_FILE = f'{OUTPUT_DIR}/LOO/loo_bias_correction/nhmv10_bias_posterior_samples.hdf5'
DIAGNOSTIC_BIAS_PREDICTION_FILTERED_FILE = f'{OUTPUT_DIR}/LOO/loo_bias_correction/nhmv10_bias_posterior_samples_filtered.hdf5'
DIAGNOSTIC_BIAS_CORRECTED_FDC_FILE = f'{OUTPUT_DIR}/LOO/loo_bias_correction/nhmv10_fdc_bias_corrected_samples.hdf5'

PYWRDRB_BIAS_PREDICTION_FILE = f'{OUTPUT_DIR}/nhmv10_pywrdrb_bias_posterior_samples.hdf5'
PYWRDRB_BIAS_PREDICTION_FILTERED_FILE = f'{OUTPUT_DIR}/nhmv10_pywrdrb_bias_posterior_samples_filtered.hdf5'
PYWRDRB_BIAS_CORRECTED_FDC_FILE = f'{OUTPUT_DIR}/nhmv10_pywrdrb_fdc_bias_corrected_samples.hdf5'
