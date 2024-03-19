"""
This script makes the final figures!

This includes:
- Map
- Diagnostic data summary
- Leave-one-out diagnostic comparison
- Drought characteristics
- Simulated performance during FFMP period
- Simulated performance during 1960s drought


Supplemental:
- Inflow scaling regression
- Errors across K-range for KNN-QPPQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import geopandas as gpd
import sys

sys.path.insert(0, '../')
from methods.processing.load import load_gauge_matches, load_leave_one_out_datasets
from methods.processing.load import load_unmanaged_gauge_metadata
from methods.processing.load import load_historic_datasets
from methods.processing import extract_loo_results_from_hdf5, get_upstream_gauges
from methods.processing.prep_loo import get_basin_catchment_area
from methods.diagnostics import get_error_summary

from methods.plotting.diagnostic_plots import plot_grid_metric_map, plot_Nx3_error_slope_cdf
from methods.plotting.diagnostic_plots import plot_error_cdf_subplot
from methods.plotting.diagnostic_plots import plot_Nx1_binyear_boxplots
from methods.plotting.data_plots import plot_data_summary
from methods.utils.constants import cms_to_mgd, crs
from methods.utils.lists import model_datasets, pub_model_datasets
from methods.diagnostics.metrics import error_metrics
from methods.generator.inflow_scaling_regression import plot_inflow_scaling_regression
from methods.utils.directories import pywrdrb_dir, path_to_nhm_data, path_to_nwm_data, fig_dir
from methods.utils.directories import data_dir, output_dir


plt.ioff()

error_datasets = ['nhmv10', 'nwmv21', 
                  'obs_pub_nhmv10', 'obs_pub_nwmv21', 
                  'obs_pub_nhmv10_ensemble', 'obs_pub_nwmv21_ensemble'] 

recalculate_error_metrics = False

#####################
### Load Data #######
#####################

# Comparison dates
start_date = '1983-10-01'
end_date = '2016-12-31'


### Spatial
drb_boundary = gpd.read_file(f'{pywrdrb_dir}DRB_spatial/DRB_shapefiles/drb_bnd_polygon.shp').to_crs(crs)
drb_mainstem = gpd.read_file(f'{pywrdrb_dir}DRB_spatial/DRB_shapefiles/delawareriver.shp').to_crs(crs)
prediction_locations = pd.read_csv(f'{data_dir}prediction_locations.csv', sep = ',', index_col=0)
prediction_locations = gpd.GeoDataFrame(prediction_locations, geometry=gpd.points_from_xy(prediction_locations.long, prediction_locations.lat))
    
## Streamflows
Q_observed = pd.read_csv(f'{data_dir}USGS/drb_historic_unmanaged_streamflow_cms.csv', index_col=0, parse_dates=True)
Q_loo_datasets = load_leave_one_out_datasets()
Q_reconstructions = load_historic_datasets(model_datasets)
unmanaged_gauge_meta = load_unmanaged_gauge_metadata()
gauge_matches = load_gauge_matches()

loo_sites = Q_loo_datasets['nhmv10'].columns.to_list()

print('Finding upstream gauges from LOO sites')
subcatchment_gauges = get_upstream_gauges(loo_sites, unmanaged_gauge_meta,
                                          simplify=True)

## Basin area geom
# Get catchment areas
# print('Getting catchment areas...')
# basin_areas = np.zeros(len(loo_sites))
# for i, site in enumerate(loo_sites):
#     basin_areas[i] = get_basin_catchment_area(feature_id=site,
#                                               feature_source='nwissite')
    
print(f'Ready to analyze {len(loo_sites)} LOO sites')


################
### Errors #####
################
### Get Error Metrics
if recalculate_error_metrics:
    error_summary_annual = get_error_summary(Q_loo_datasets, 
                                             error_datasets, 
                                         loo_sites, 
                                         by_year=True, 
                                         start_date='1945-01-01', end_date='2016-12-31')
    error_summary_annual.to_csv(f'{output_dir}LOO/loo_error_summary_annual.csv')
    
    error_summary = get_error_summary(Q_loo_datasets, 
                                      error_datasets, 
                                         loo_sites,  
                                         start_date='1945-01-01', end_date='2016-12-31')
    error_summary.to_csv(f'{output_dir}LOO/loo_error_summary.csv')

else:
    error_summary_annual = pd.read_csv(f'{output_dir}LOO/loo_error_summary_annual.csv', dtype={'site':str})
    error_summary = pd.read_csv(f'{output_dir}LOO/loo_error_summary.csv', dtype={'site':str})


################
### Main #######
################

### Diagnostic data summary
print('Making data summary plot...')
plot_data_summary(Q_observed, 
                       unmanaged_gauge_meta, 
                       loo_sites, 
                       prediction_locations, 
                       drb_boundary, 
                       sort_by='lat')

plot_data_summary(Q_observed, 
                       unmanaged_gauge_meta, 
                       loo_sites, 
                       prediction_locations, 
                       drb_boundary, 
                       sort_by='record_length')


### Leave-one-out diagnostics

## CDF comparisons of every metric
for metric in error_metrics:
    is_max = False if 'Q' in metric else True
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_error_cdf_subplot(ax, error_datasets, metric, error_summary, legend=True,
                           plot_ensemble_range=True, maximize=is_max)
    plt.savefig(f'./figures/diagnostics/{metric}_cdf_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    

## Nx3 Grid Slope & CDF comparisons of multiple metric
print('Making LOO Nx3 grid slope plots...')
use_metrics = ['nse', 'log_nse', 'r', 'AbsQ0.1pbias']

plot_Nx3_error_slope_cdf(error_summary, error_datasets, 
                         use_metrics, loo_sites,
                         plot_ensemble=False)


## Nx1 Errors Over Time
plot_Nx1_binyear_boxplots(error_summary_annual,
                          use_metrics,  
                          bin_size=10, plot_ensembles=False)

plt.ioff()

### Flow & Drought characteristics









################
# Supplemental #
################

## Inflow scaling regression scatter plots
for model in ['nhmv10', 'nwmv21']:
    plot_inflow_scaling_regression(model, roll_window=3)
    
## Errors across K-range for KNN-QPPQ
