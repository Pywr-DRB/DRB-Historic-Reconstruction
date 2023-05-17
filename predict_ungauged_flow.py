"""
Trevor Amestoy

Uses neural-net predicted FDCs and flow data from gages in the surrounding
region to predict streamflow at ungaged locations in the Delaware River Basin.

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import datetime as dt
import PUB_Generator

# Load a predefined set of neural net hyperparameters
from PUB_Generator.NN.hyperparameters import best_config

#%%############################################################################
### Step 0: Data, constants, and parameters of interest
###############################################################################
## Constants and specifications
date_start = dt.datetime(1999,1,1) 
date_end = dt.datetime(2016,12,31) 
feature_subset = 'my_features'
cms_to_mgd = 22.82

use_nhm_fdcs = True

### Step 0.1: Load data - already processed
yTr = pd.read_csv(f'./data/training_outputs.csv', sep = ',', index_col = 0)*cms_to_mgd
xTr = pd.read_csv(f'./data/training_inputs.csv', sep = ',', index_col = 0)
Q = pd.read_csv('./data/historic_observed_flow.csv', sep = ',', index_col = 0)*cms_to_mgd
xPr = pd.read_csv('./data/prediction_inputs.csv', sep =',',  index_col = 0)
nhm_fdc = pd.read_csv('./data/nhmv10_node_fdc.csv', sep =',',  index_col = 0)

prediction_locations = pd.read_csv(f'./data/prediction_locations.csv', sep = ',')



Q.index = pd.to_datetime(Q.index)
Q = Q.loc[date_start:date_end,:]
Q = Q.dropna(axis=1)
print(f'Q is size {Q.shape}')

qs = [0.0003, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.995, 0.9997]
basin_area_ind = np.where(xTr.columns.values == 'CAT_BASIN_AREA')[0][0]
n_k = 20
n_ensemble = 10
NN_parameters = {
    'lr_initial': 0.005,
    'lr_minimum': 0.00001,
    'lr_momentum': 0.05,
    'dropout': 0.005,
    'batch_fraction': 0.5,
    'max_epochs': 800,
    'n_hiddenlayers': 2,
    'n_nodes': [100, 50],
    'print_frequency': 290
}

# Initialize storage
predicted_fdcs = pd.DataFrame(index = prediction_locations.name, columns = qs)
predicted_flows = pd.DataFrame(index = Q.index, columns = prediction_locations.name)
error_metrics = pd.DataFrame(index = prediction_locations.name, columns = ['model_error', 'mono_violations'])
knn = pd.DataFrame(index = prediction_locations.name, columns = np.arange(n_k))

### Step 1: PUB Prediction

model = PUB_Generator.PUBModel(xTr, yTr, xPr, Q, NN_parameters,
                               basin_area_name = "CAT_BASIN_AREA",
                               standardize = True,
                               normalize = False,
                               pca = False,
                               filter_drb = False,
                               remove_managed = True,
                               remove_outliers = False,
                               log_runoff = True,
                               use_features = "all",
                               debugging = True)

model._preprocess()

n_repeats = 1
fdc_predictions = np.zeros((len(prediction_locations), len(qs), n_repeats))
for i in range(n_repeats):
    model.train()
    print(f'Biases: {model.model.net.lfcs[3].bias}')
    model.predict_fdc()
    fdc_predictions[:,:, i] = model.fdc_prediction

mean_fdc_predictions = fdc_predictions.mean(axis = 2)
pd.DataFrame(mean_fdc_predictions, index = model.xPr.index, columns = qs)
all_xPr = model.xPr.copy()
all_test_x = model.test_x.copy()

if use_nhm_fdcs:
    fdc_predictions = nhm_fdc.loc[model.xPr.index,:].to_numpy()
else:
    fdc_predictions = mean_fdc_predictions

np.savetxt(f'./output/all_drb_predicted_fdcs_{feature_subset}.csv', fdc_predictions, delimiter =',')
# Plot FDCs before continuing
colormap = iter(plt.cm.rainbow(np.linspace(0, 1, all_xPr.shape[0])))

quants = np.linspace(0,1,200)
all_fdc = np.quantile(model.Q.T, quants, axis = 1)
nhm_fdc = nhm_fdc.loc[model.xPr.index, :]
nhm_logrunoff = np.log(nhm_fdc.divide(model.xPr.CAT_BASIN_AREA, axis=0)) 
fig, ax = plt.subplots(figsize = (7,3), dpi =250)
for node in nhm_logrunoff.index:
    c = next(colormap)
    ax.plot(qs, nhm_logrunoff.loc[node,:], alpha= 0.5, color = c, linestyle = '--', linewidth = 1)
colormap = iter(plt.cm.rainbow(np.linspace(0, 1, all_xPr.shape[0])))
for i in range(len(fdc_predictions)):
    c = next(colormap)
    ax.plot(qs, fdc_predictions[i], alpha= 1, color = c, linewidth = 1, label = xPr.index[i])
#plt.yscale('log')
#plt.yscale('log')
plt.legend(fontsize = 5, bbox_to_anchor = (1.0,1.0), loc = 'upper left', ncol = 2)
plt.tight_layout()
plt.show()


clean_sites = model.xTr.index.intersection(model.Q.columns)
model.xTr = model.xTr.loc[clean_sites, :]
model.Q = model.Q.loc[:,clean_sites]
print(f'Q is size {model.Q.shape}')


for pi in range(xPr.shape[0]):
    print(f'Generating prediction for site {pi} of {xPr.shape[0]}.')
    
    name = prediction_locations.name[pi]
    x_predict = pd.DataFrame(all_xPr.iloc[pi, :]).T
   
    
    model.xPr = x_predict
    model.test_x = all_test_x[pi,:]
    model.fdc_prediction = fdc_predictions[pi,:].flatten()

    model.predict_flow()

    # Store
    predicted_fdcs.loc[name,:] = model.fdc_prediction
    predicted_flows.loc[:,name] = model.predicted_flow.flatten()
    error_metrics.loc[name, 'model_error'] = model.model_error
    error_metrics.loc[name, 'mono_violations'] = model.fdc_monotonic_violations
    knn.loc[name,:] = model.knn_distances

    # Periodic save
    if pi%5 == 0:
        predicted_fdcs.to_csv(f'./output/drb_pub_predicted_fdcs_{feature_subset}.csv', sep =',')
        predicted_flows.to_csv(f'./output/drb_pub_predicted_flows_{feature_subset}.csv', sep =',')
        knn.to_csv(f'./output/drb_pub_knn_distances_{feature_subset}.csv', sep = ',')

#%%#############################################################################
### EXPORT RESULTS
################################################################################

predicted_fdcs.to_csv(f'./output/drb_pub_predicted_fdcs_{feature_subset}.csv', sep =',')
predicted_flows.to_csv(f'./output/drb_pub_predicted_flows_{feature_subset}.csv', sep =',')
knn.to_csv(f'./output/drb_pub_knn_distances_{feature_subset}.csv', sep = ',')

print('Done!  See the output folder.')

ts = np.arange(predicted_flows.shape[0])
ts = predicted_flows.index
fig, ax = plt.subplots(figsize = (10,2), dpi =250)
for i in range(10):
    ax.plot(ts, predicted_flows.iloc[:,i], alpha= 0.3, linewidth = 0.5)
plt.yscale('log')
plt.show()


ts = model.Q.index
fig, ax = plt.subplots(figsize = (10,2), dpi =250)
for i in range(10):
    ax.plot(ts, model.Q.iloc[:,i], alpha= 0.3, linewidth = 0.5)
plt.yscale('log')
plt.show()

ts = Q.index
fig, ax = plt.subplots(figsize = (10,2), dpi =250)
for i in range(20):
    ax.plot(ts, model.QPPQ_model.observed_NEPs[i,:], alpha= 0.3, color = 'grey', linewidth = 0.5)
ax.plot(ts, model.QPPQ_model.predicted_NEPs.flatten(), alpha= 1.0, color = 'darkblue', linewidth = 1)
plt.show()

quants = np.linspace(0,1,200)
fig, ax = plt.subplots(figsize = (5,5), dpi =250)
knn_fdc = np.quantile(model.QPPQ_model.KNN_Qobs, quants, axis = 1)
for i in range(knn_fdc.shape[1]):
    ax.plot(quants, knn_fdc[:,i], alpha= 0.5, color = 'grey', linewidth = 0.5)
ax.plot(qs, fdc_predictions[-1], color = "darkblue", alpha= 1, linewidth = 1)
plt.yscale('log')
plt.yscale('log')
plt.show()

fig, ax = plt.subplots(figsize = (5,5), dpi =250)
all_fdc = np.quantile(model.Q.T, quants, axis = 1)

for i in range(all_fdc.shape[1]):
    ax.plot(quants, all_fdc[:,i], alpha= 0.5, color = 'grey', linewidth = 0.5)
for i in range(len(fdc_predictions)):
    ax.plot(qs, fdc_predictions[i], alpha= 1, linewidth = 1, label = xPr.index[i])
plt.yscale('log')
plt.yscale('log')
#plt.legend()
plt.show()

