import pandas as pd
import json
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from methods.utils.directories import DATA_DIR, OUTPUT_DIR
from methods.processing.load import load_model_segment_flows, load_gauge_matches
from methods.utils.constants import cms_to_mgd
from methods.processing.prep_loo import get_leave_one_out_sites
from methods.processing.catchments import load_station_catchments
from methods.bias_correction.transformations import calculate_quantile_biases, empirical_cdf

from methods.utils.RandomForestUtils import iterative_rfe

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


####################################################################################################
### Specifications ###
####################################################################################################

MAx_ALLOWABLE_STORAGE = 1000 # acre-feet of total storage in catchment
boundary = 'drb'


####################################################################################################
### Load data ###
####################################################################################################

### usgs gauge data
Q = pd.read_csv(f'{DATA_DIR}/USGS/drb_streamflow_daily_usgs_cms.csv',
                index_col=0, parse_dates=True)*cms_to_mgd
Q = Q.loc['01-01-1980':,:]

# usgs gauge metadata
gauge_meta = pd.read_csv(f'{DATA_DIR}/USGS/drb_usgs_metadata.csv', 
                         index_col=0, dtype={'site_no': str, 'comid': str})


# Gauge matches have columns for both station number and comid or feature id
gauge_matches = load_gauge_matches()

# NHM 
drb_nhm_segment_flows = load_model_segment_flows('nhmv10', station_number_columns=True)
drb_nwm_segment_flows = load_model_segment_flows('nwmv21', station_number_columns=True)

## Get unmanaged catchments
unmanaged_gauge_meta = gauge_meta[gauge_meta['total_catchment_storage'] < MAx_ALLOWABLE_STORAGE]
unmanaged_marginal_gauge_meta = gauge_meta[gauge_meta['marginal_catchment_storage'] < MAx_ALLOWABLE_STORAGE]


## NLDI features
nldi_data = pd.read_csv(f'{DATA_DIR}/NLDI/drb_usgs_nldi_catchment_characteristics.csv', index_col=0)

# convert index from comid to station_id using gauge_meta
nldi_data['site_no'] = np.array([np.nan]*len(nldi_data), dtype=object)
for cid in nldi_data.index.values:
    if str(cid) in gauge_meta['comid'].values:
        s_id = gauge_meta[gauge_meta['comid'] == str(cid)].index.values[0]
        nldi_data.loc[cid, 'site_no'] = s_id
    else:
        nldi_data.drop(cid, inplace=True, axis=1)
nldi_data = nldi_data.set_index('site_no')



nldi_chars = pd.read_csv(f'{DATA_DIR}/NLDI/final_features.csv', index_col=0)
nldi_chars = nldi_chars.index.values

## Get loo sites
loo_sites = get_leave_one_out_sites(Q, unmanaged_gauge_meta.index.values,
                                        gauge_matches['nwmv21']['site_no'].values,
                                        gauge_matches['nhmv10']['site_no'].values)

# get union with NLDI site data
nldi_data = nldi_data.loc[nldi_data.index.isin(loo_sites)]
nldi_data = nldi_data.loc[~nldi_data.index.duplicated(keep='first')]

loo_sites = nldi_data.index.values

# catchment geometry
station_catchments = load_station_catchments()
station_catchments['area'] = station_catchments['area']*1e3
station_catchments = station_catchments.loc[station_catchments.index.isin(loo_sites)]

print(f'{len(loo_sites)} leave-one-out sites')


### DayMet data ###
# Load daymet data
aggregate = 'daily'
catchment_prcp = pd.read_csv(f'{DATA_DIR}/Daymet/drb_catchment_avg_prcp_{aggregate}.csv',
                                    index_col=0, parse_dates=True).dropna(axis=1)
catchment_prcp_monthly = pd.read_csv(f'{DATA_DIR}/Daymet/drb_catchment_avg_prcp_monthly.csv', 
                                     index_col=0, parse_dates=True).dropna(axis=1)

# Drop the first 3 years
catchment_prcp = catchment_prcp.loc[(3*365):]
catchment_prcp_monthly = catchment_prcp_monthly.iloc[(3*12):]

####################################################################################################
### Preprocessing ###
####################################################################################################

### Get empirical FDCs
fdc_quantiles = [0.0003, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.995, 0.9997]

fdc = {}
monthly_fdc = {}
for model in ['obs', 'nhmv10', 'nwmv21']:
    if model == 'obs':
        Q_model = Q
    elif model == 'nhmv10':
        Q_model = drb_nhm_segment_flows
    elif model == 'nwmv21':
        Q_model = drb_nwm_segment_flows
    
    fdc[model]= empirical_cdf(Q_model.loc[:,loo_sites].values, fdc_quantiles)
    fdc[model] = pd.DataFrame(fdc[model], index=loo_sites, columns=fdc_quantiles)
    
    monthly_fdc[model] = empirical_cdf(Q_model.loc[:,loo_sites].resample('ME').sum().replace(0.0, np.nan).values, fdc_quantiles)
    monthly_fdc[model] = pd.DataFrame(monthly_fdc[model], index=loo_sites, columns=fdc_quantiles)
    

## DayMet data "FDCs"
if aggregate == 'daily':
    n_years = 41-3
    daymet_start_year = 1983
    # # set index to day of year 
    day_of_year_index = np.tile(np.arange(1,366), n_years)
    catchment_prcp.index = day_of_year_index

    catchment_prcp = catchment_prcp.replace(0, np.nan)
    prcp_fdc = catchment_prcp.groupby(catchment_prcp.index).median().quantile(fdc_quantiles, axis=0).T
    # prcp_fdc = catchment_prcp.dropna().quantile(fdc_quantiles, axis=0).T
        

prcp_fdc = prcp_fdc.loc[loo_sites]
assert((prcp_fdc.index == fdc['obs'].index).all()), 'prcp and obs indices do not match'
fdc['P'] = prcp_fdc.copy()
catchment_prcp.index = pd.date_range(f'{daymet_start_year}-01-01', periods=len(catchment_prcp), freq='D')
station_catchments = station_catchments.loc[prcp_fdc.index]

monthly_fdc['P'] = empirical_cdf(catchment_prcp_monthly.loc[:,loo_sites].values, fdc_quantiles)
monthly_fdc['P'] = pd.DataFrame(monthly_fdc['P'], index=loo_sites, columns=fdc_quantiles)

### Get differences between various FDC curves
obs_model_diff = {}
prcp_model_diff = {}
monthly_prcp_model_diff = {}
for i in range(1,9):
    obs_model_diff[f'method_{i}'] = {}
    prcp_model_diff[f'method_{i}'] = {}
    monthly_prcp_model_diff[f'method_{i}'] = {}
    for model in ['obs', 'nhmv10', 'nwmv21']:
        prcp_model_diff[f'method_{i}'][model] = calculate_quantile_biases(fdc['P'], 
                                                                          fdc[model], 
                                                                          method=i, 
                                                                          area=station_catchments[['area']],
                                                                          precip_observation=True)
        monthly_prcp_model_diff[f'method_{i}'][model] = calculate_quantile_biases(monthly_fdc['P'], 
                                                                          monthly_fdc[model], 
                                                                          method=i, 
                                                                          area=station_catchments[['area']],
                                                                          precip_observation=True)
                                                                                  
        if model != 'obs':
            obs_model_diff[f'method_{i}'][model] = calculate_quantile_biases(fdc['obs'],
                                                                            fdc[model], 
                                                                            method=i, 
                                                                            area=station_catchments[['area']])  
            
            
### Get flow metrics of interest
flow_metrics = ['mean', 'std', 'skew', 'kurt', 
                'cv', 'high_q', 'low_q', 'min', 'max']

flow_metric_vals = {}
flow_metric_vals['nhmv10'] = pd.DataFrame(index=loo_sites, columns=flow_metrics)
flow_metric_vals['nwmv21'] = pd.DataFrame(index=loo_sites, columns=flow_metrics)
flow_metric_vals['P'] = pd.DataFrame(index=loo_sites, columns=flow_metrics)

for model in ['nhmv10', 'nwmv21', 'P']:
    for site in loo_sites:
        if model == 'nhmv10':
            Q_site = drb_nhm_segment_flows.loc[:,site].dropna()
        elif model == 'nwmv21':
            Q_site = drb_nwm_segment_flows.loc[:,site].dropna()
        elif model == 'P':
            Q_site = catchment_prcp.loc[:,site].dropna()
        Q_site = Q_site.replace(0, np.nan)
        Q_site = Q_site.dropna()
        
        
        Q_site = np.log(Q_site.astype(float))
        flow_metric_vals[model].loc[site, 'mean'] = Q_site.mean()
        flow_metric_vals[model].loc[site, 'std'] = Q_site.std()
        flow_metric_vals[model].loc[site, 'skew'] = Q_site.skew()
        flow_metric_vals[model].loc[site, 'kurt'] = Q_site.kurt()
        flow_metric_vals[model].loc[site, 'high_q'] = Q_site.quantile(0.9)
        flow_metric_vals[model].loc[site, 'low_q'] = Q_site.quantile(0.1)
        flow_metric_vals[model].loc[site, 'cv'] = Q_site.std()/Q_site.mean()
        flow_metric_vals[model].loc[site, 'min'] = Q_site.min()
        flow_metric_vals[model].loc[site, 'max'] = Q_site.max()        
        
# All potential inputs
test_fdc = 'nhmv10'

x = prcp_model_diff['method_3'][test_fdc]
x_prcp_diff = x.copy()
x_prcp_diff.columns = x_prcp_diff.columns.astype(str)
x_prcp_diff.columns = x_prcp_diff.columns + '_prcp_diff'

x_fdc = fdc[test_fdc]
x_fdc.columns = x_fdc.columns.astype(str)
x_fdc.columns = x_fdc.columns + '_fdc'

x_full = pd.concat([x_prcp_diff, 
                    x_fdc, 
                    nldi_data,
                    flow_metric_vals[test_fdc],
                    gauge_meta.loc[x.index, ['long', 'lat', 'total_catchment_storage']]], axis=1)

x_monthly_fdc_diff = monthly_prcp_model_diff['method_3'][test_fdc]
x_monthly_fdc_diff.columns = x_monthly_fdc_diff.columns.astype(str)
x_monthly_fdc_diff.columns = x_monthly_fdc_diff.columns + '_monthly_prcp_diff'
x_full = pd.concat([x_full, x_monthly_fdc_diff], axis=1)

# set all columns names as str
x_full.columns = x_full.columns.astype(str)
x_full = x_full.dropna(axis=1)
x_full = x_full.loc[:,~x_full.columns.str.contains('CAT_')]

## perform PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_full)
x_scaled = pd.DataFrame(x_scaled, index=x_full.index, columns=x_full.columns)

pca = PCA()
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
x_pca = pd.DataFrame(x_pca, index=x_full.index)
x_pca.columns = [f'PC{i+1}' for i in range(x_pca.shape[1])]

x_full = pd.concat([x_full, x_pca], axis=1)



bias_method = 3
y = obs_model_diff[f'method_{bias_method}'][test_fdc]

### Feature selection

rfe_cv_folds = [10, 10, 10]
rfe_n_estimators = [200, 100, 50]
n_features_to_select = [150, 100, 30]  # Example dynamic feature selection
n_iter = len(rfe_cv_folds)

# Distribute tasks
if rank == 0:
    selected_features = iterative_rfe(x_full, y[[0.5]].values.flatten(), n_iter, 
                                      rfe_cv_folds, rfe_n_estimators, n_features_to_select)
    selected_features = selected_features.to_list()
elif rank == 1:
    selected_features = iterative_rfe(x_full, y[[0.1]].values.flatten(), n_iter, 
                                      rfe_cv_folds, rfe_n_estimators, n_features_to_select)
    selected_features = selected_features.to_list()
elif rank == 2:
    selected_features = iterative_rfe(x_full, y[[0.9]].values.flatten(), n_iter, 
                                      rfe_cv_folds, rfe_n_estimators, n_features_to_select)
    selected_features = selected_features.to_list()

# Gather results at root process
selected_features_all = comm.gather(selected_features, root=0)

# combine all selected features
if rank == 0:
    # Process and combine all selected features from all processes
    combined_features = {0.5: selected_features_all[0], 
                         0.1: selected_features_all[1], 
                         0.9: selected_features_all[2]}
    # save combined_features to a file or further processin
    with open(f'{OUTPUT_DIR}/all_rfe_features.json', 'w') as f:
        json.dump(combined_features, f)
