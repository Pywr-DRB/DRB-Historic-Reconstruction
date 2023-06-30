# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import datetime as dt
import seaborn as sns
import h5py

# Import the QPPQ model
from QPPQModel import StreamflowGenerator

from data_processing import export_dict_ensemble_to_hdf5, extract_realization_from_hdf5
from inflow_scaling_regression import train_inflow_scale_regression_models, predict_inflow_scaling
from inflow_scaling_regression import get_quarter

# Directory to pywrdrb project
pywrdrb_directory = '../Pywr-DRB/'
sys.path.append(pywrdrb_directory)

from pywrdrb.pywr_drb_node_data import obs_pub_site_matches


# Model generation specifications
full_date_range = ('1945-01-01', '2022-12-31')
max_daterange = pd.date_range('1945-01-01', '2022-12-31')
N_REALIZATIONS = 30
K = 5
donor_fdc = 'nhmv10'
hru_scaled = False
remove_mainstem_gauges = True
regression_nhm_inflow_scaling = False

# Constants
cms_to_mgd = 22.82
fdc_quantiles = [0.0003, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.995, 0.9997]

### Load data 
Q = pd.read_csv(f'./data/historic_unmanaged_streamflow_1900_2023_cms.csv', sep = ',', index_col = 0, parse_dates=True)*cms_to_mgd
nhm_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nhmv10.csv', sep =',',  index_col = 0, parse_dates=True)
nwm_flow = pd.read_csv(f'{pywrdrb_directory}/input_data/gage_flow_nwmv21.csv', sep =',',  index_col = 0, parse_dates=True)
nyc_nhm_inflows = pd.read_csv('./data/nyc_inflow_nhm_streamflow.csv', sep = ',', index_col = 0, parse_dates=True)


prediction_locations = pd.read_csv(f'./data/prediction_locations.csv', sep = ',', index_col=0)
gauge_meta = pd.read_csv(f'./data/drb_unmanaged_usgs_metadata.csv', sep = ',', dtype = {'site_no':str})
gauge_meta.set_index('site_no', inplace=True)

# Some gauge data is faulty
gauge_meta.loc['01414000', 'begin_date'] = '1996-12-05'
gauge_meta.loc['0142400103', 'begin_date'] = '1996-12-05'

# Get estiamtes of FDCs at all nodes; to be used for QPPQ when no data is available
node_fdcs = pd.DataFrame(index = prediction_locations.index, columns=fdc_quantiles)
if donor_fdc == 'nhmv10':
    fdc_donor_flow = nhm_flow
elif donor_fdc == 'nwmv21':
    fdc_donor_flow = nwm_flow
else:
    print('Invalid donor_fdc specification. Options: nhmv10, nwmv21')

for i, node in enumerate(prediction_locations.index):
    node_fdcs.loc[node, :] = np.quantile(fdc_donor_flow.loc[:,node], fdc_quantiles)
    
# Remove outflow gauges from flow data
for node, sites in obs_pub_site_matches.items():
    if f'USGS-{node}' in Q.columns:
        print(f'Removing {node} from data.')
        Q = Q.drop(f'USGS-{node}', axis=1)

# Remove Trenton mainstem gauge
if remove_mainstem_gauges:
    if f'USGS-{obs_pub_site_matches["delTrenton"]}' in Q.columns:
        print(f'Removing Trenton gauge from data.')
        Q = Q.drop(f'USGS-{obs_pub_site_matches["delTrenton"]}', axis=1)
    if f'USGS-{obs_pub_site_matches["delMontague"]}' in Q.columns:
        print(f'Removing Montague gauge from data.')
        Q = Q.drop(f'USGS-{obs_pub_site_matches["delMontague"]}', axis=1)

    obs_pub_site_matches['delTrenton'] = None
    obs_pub_site_matches['delDRCanal'] = None
    obs_pub_site_matches['delMontague'] = None

# Make sure other inflow gauges are in the dataset
missing = 0
for node, sites in obs_pub_site_matches.items():
    if sites is not None:
        for s in sites:
            if f'USGS-{s}' not in Q.columns:
                print(f'Site {s} for node {node} is not available')
#assert(missing == 0), 'Atleast one of the inflow gauge timeseries if not available in the data.'

############
### QPPQ ###
############

# Set-up QPPQ
reconstructed_sites = []
for node, sites in obs_pub_site_matches.items():
    if node == 'delDRCanal':
        pass
    elif sites is None:
        reconstructed_sites.append(node)
    else:
        for s in sites:
            reconstructed_sites.append(s)

# Intialize storage
max_annual_NA_fill = 20
Q_reconstructed = pd.DataFrame(index=max_daterange, columns = reconstructed_sites)
ensemble_Q_reconstructed = {}

N_YEARS = int(np.floor(len(max_daterange)/365))

starts = [f'{1945+i}-01-01' for i in range(N_YEARS)]
ends = [f'{1945+i}-12-31' for i in range(N_YEARS)]
daterange_subsets = np.vstack([starts, ends]).transpose()
assert(pd.to_datetime(daterange_subsets[-1,-1]).date() <= Q.index.max().date()), 'The historic data must be more recent than QPPQ daterange.'


if regression_nhm_inflow_scaling:
    linear_models = {}
    linear_results = {}

    for reservoir in ['cannonsville', 'pepacton']:
        linear_models[reservoir], linear_results[reservoir] = train_inflow_scale_regression_models(reservoir, 
                                                                                                   nyc_nhm_inflows)
        
## QPPQ prediction
# Generate 1 year at a time, to maximize the amount of data available for each years QPPQ
for real in range(N_REALIZATIONS):
    print(f'Generation realization {real} of {N_REALIZATIONS}.')
    for i, dates in enumerate(daterange_subsets):
        # Run predictions one location at a time
        for node, sites in obs_pub_site_matches.items():
            
            # Pull gauges that have flow during daterange
            Q_subset = Q.loc[dates[0]:dates[1], :].dropna(axis=1)
            subset_sites = [f'{i.split("-")[1]}' for i in Q_subset.columns]
            gauge_meta_subset = gauge_meta.loc[subset_sites, :]
            gauge_meta_subset.index = Q_subset.columns
            
            # Initialize the model
            model = StreamflowGenerator(K= K,
                                        observed_flow = Q_subset, 
                                        observation_locations=gauge_meta_subset,
                                        probabalistic_sample = True)

            # Handle sites with historic data
            if sites is not None:
                for s in sites:
                    # First, use observation data if available
                    number_of_nas = Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna().sum()
                    if (number_of_nas == 0):
                        Q_reconstructed.loc[dates[0]:dates[1], s] = Q.loc[dates[0]:dates[1], f'USGS-{s}'].values
                    elif (number_of_nas <= max_annual_NA_fill):
                        # print(f'Filling {number_of_nas} NAs for site {s} using median.')
                        Q_reconstructed.loc[dates[0]:dates[1], s] = Q.loc[dates[0]:dates[1], f'USGS-{s}'].values
                        
                        # Fill NA using median                    
                        na_indices = Q.loc[dates[0]:dates[1],:].loc[Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), :].index
                        median_flow = np.median(Q.loc[dates[0]:dates[1], :].loc[~Q.loc[dates[0]:dates[1], f'USGS-{s}'].isna(), f'USGS-{s}'])
                        Q_reconstructed.loc[na_indices.date, s] = median_flow
                            
                    # If flow data isn't available, use historic observation to generate FDC and make PUB predictions
                    else:
                        # print(f'Using partial record for {s} during {dates}')
                        location = gauge_meta.loc[s, ['long', 'lat']].values
                        incomplete_site_flows = Q.loc[:, f'USGS-{s}'].dropna(axis=0)
                        
                        # Only use site flows for FDC if longer than 10-year record
                        if len(incomplete_site_flows)/365 >= 10:
                            fdc = np.quantile(incomplete_site_flows.values, fdc_quantiles)
                        else:
                            fdc = node_fdcs.loc[node, :].astype('float').values            
            
                        Q_reconstructed.loc[dates[0]:dates[1], s] = model.predict_streamflow(location, fdc).values.flatten()
            else:
                # print(f'Full PUB for {s} during {dates}')
                location = prediction_locations.loc[node, ['long', 'lat']].values
                fdc = node_fdcs.loc[node, :].astype('float').values

                Q_reconstructed.loc[dates[0]:dates[1], node] = model.predict_streamflow(location, fdc).values.flatten()
                
            # Apply NHM scaling at Cannonsville and Pepacton
            if (node in ['cannonsville', 'pepacton']) and regression_nhm_inflow_scaling:
                unscaled_inflows = Q_reconstructed.loc[dates[0]:dates[1], obs_pub_site_matches[node]].sum(axis=1).values
                scaling_coef = np.zeros_like(unscaled_inflows)
                
                # Use linear regression to find inflow scaling coefficient
                for i, day_month in enumerate(Q_reconstructed.loc[dates[0]:dates[1], :].index.month):
                    quarter = get_quarter(day_month)
                    
                    scaling_coef[i] = predict_inflow_scaling(linear_models[node][quarter], 
                                               linear_results[node][quarter], 
                                                log_flow= np.log(unscaled_inflows[i]),
                                                method = 'random')
                                        
                # Apply daily scaling equally across gauges
                for site in obs_pub_site_matches[node]:
                    Q_reconstructed.loc[dates[0]:dates[1], site] = np.multiply(Q_reconstructed.loc[dates[0]:dates[1], site].values,
                                                                               scaling_coef)
                    
    assert(Q_reconstructed.isna().sum().sum() == 0), ' There are NAs in the reconstruction'
    
    ensemble_Q_reconstructed[f'realization_{real}'] = Q_reconstructed.copy()


##############
### Export ###
##############

def export_dict_ensemble_to_hdf5(dict, output_file):
    N = len(dict)
    T, M = dict[f'realization_0'].shape
    column_labels = dict[f'realization_0'].columns.to_list()
    
    with h5py.File(output_file, 'w') as f:
        for i in range(N):
            data = dict[f'realization_{i}']
            datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
            
            grp = f.create_group(f"realization_{i+1}")
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels

            # Create dataset for dates
            grp.create_dataset('date', data=datetime)
            
            # Create datasets for each location's timeseries
            for j in range(M):
                dataset = grp.create_dataset(column_labels[j], data=data[column_labels[j]].to_list())


if regression_nhm_inflow_scaling:
    output_filename = f'./outputs/ensembles/historic_reconstruction_daily_ensemble_{N_REALIZATIONS}_scaled_NYC.hdf5'
else:
    output_filename = f'./outputs/ensembles/historic_reconstruction_daily_ensemble_{N_REALIZATIONS}.hdf5'

print(f'Exporting {N_REALIZATIONS} to {output_filename}')

export_dict_ensemble_to_hdf5(ensemble_Q_reconstructed, output_filename)
