AGG_K_MIN = 1
AGG_K_MAX = 8
ENSEMBLE_K_MIN = 2
ENSEMBLE_K_MAX = 10
loo_models = []
ensemble_methods = [1,2]
### Get lists of filenames
for nxm in ['nhmv10', 'nwmv21']:
    
    ## Different QPPQ aggregate models
    for k in range(AGG_K_MIN, AGG_K_MAX):
        loo_models.append(f'obs_pub_{nxm}_K{k}')
        
    ## Different QPPQ ensemble models
    for k in range(ENSEMBLE_K_MIN, ENSEMBLE_K_MAX):
        for e in ensemble_methods:
            loo_models.append(f'obs_pub_{nxm}_K{k}_ensemble_m{e}')


# Random
fig_dpi = 300
ax_tick_label_fsize = 10
ax_label_fsize = 12
sup_title_fsize = 12
legend_fsize = 9


# Aesthetics for plotting
# Aesthetics for plotting
model_colors = {
    'obs': 'black',
    'nhmv10' : '#925736', 
    'nwmv21' : '#385723',
    'obs_pub_nhmv10' : '#F27300',
    'obs_pub_nhmv10_ObsScaled': '#F27300', 
    'obs_pub_nhmv10_ensemble' : '#F9B572', 
    'obs_pub_nhmv10_ObsScaled_ensemble' : '#F9B572', 
    'obs_pub_nhmv10_BC_ObsScaled_ensemble' : '#F9B572', 
    'obs_pub_nwmv21' : '#0174BE', 
    'obs_pub_nwmv21_ensemble': '#9CD2F6',
    'obs_pub_nwmv21_ObsScaled' : '#0174BE', 
    'obs_pub_nwmv21_BC_ObsScaled_ensemble': '#9CD2F6',
    'obs_pub_nwmv21_ObsScaled_ensemble': '#9CD2F6'
    }

model_labels = {
    'obs': 'Observed',
    'nhmv10' : 'NHMv1.0',
    'nwmv21' : 'NWMv2.1',
    'obs_pub_nhmv10' : 'PUB-NHM',
    'obs_pub_nhmv10_ensemble' : 'PUB-NHM Ensemble',
    'obs_pub_nwmv21' : 'PUB-NWM',
    'obs_pub_nwmv21_ensemble':'PUB-NWM Ensemble',
    'obs_pub_nhmv10_ObsScaled' : 'PUB-NHM',
    'obs_pub_nhmv10_ObsScaled_ensemble' : 'PUB-NHM Ensemble',
    'obs_pub_nhmv10_BC_ObsScaled_ensemble' : 'Ensemble',
    'obs_pub_nwmv21_ObsScaled' : 'PUB-NWM',
    'obs_pub_nwmv21_ObsScaled_ensemble':'PUB-NWM Ensemble',
    'obs_pub_nwmv21_BC_ObsScaled_ensemble':'PUB-BC-NWM Ensemble'
    }


for model in loo_models:
    if model not in model_colors.keys():
        model_colors[model] = 'black'
    if model not in model_labels.keys():
        model_labels[model] = model
    if 'm2' in model:
        model_colors[model] = 'blue'
        model_labels[model] = model_labels[model] + ' (Method 2)'
    if 'm1' in model:
        model_labels[model] = model_labels[model] + ' (Method 1)'
        
        
model_linewidths = {
    'obs': 2.5,
    'nhmv10' : 2,
    'nwmv21' : 2,
    'obs_pub_nhmv10' : 2,
    'obs_pub_nhmv10_ensemble' : 1,
    'obs_pub_nwmv21' : 2,
    'obs_pub_nwmv21_ensemble':1
    }

model_linstyles = {
    'obs': '--',
    'nhmv10' : '-',
    'nwmv21' : '-',
    'obs_pub_nhmv10' : '-',
    'obs_pub_nhmv10_ensemble' : '-',
    'obs_pub_nwmv21' : '-',
    'obs_pub_nwmv21_ensemble':'-'
    }


### Axis information

ideal_metric_scores = {'nse':1, 'kge':1, 'r':1, 'alpha':1, 'beta':1,
                       'log_nse':1, 'log_kge':1, 'log_r':1, 'log_alpha':1, 'log_beta':1, 
                       'Q0.1pbias':0.0, 'Q0.2pbias':0.0, 'Q0.3pbias':0.0, 
                       'AbsQ0.1pbias':0.0, 'AbsQ0.2pbias':0.0, 'AbsQ0.3pbias':0.0}
                
lower_bound_metric_scores = {'nse':-1, 'kge':-1, 'r':0.0, 'alpha':0, 'beta':0,
                       'log_nse':-1, 'log_kge':0, 'log_r':0.0, 'log_alpha':-1, 'log_beta':-1, 
                       'Q0.1pbias':-1, 'Q0.2pbias':-1, 'Q0.3pbias':-1, 
                       'AbsQ0.1pbias':0, 'AbsQ0.2pbias':0, 'AbsQ0.3pbias':0}

upper_bound_metric_scores = {'nse':1, 'kge':1, 'r':1, 'alpha':2, 'beta':2,
                       'log_nse':1, 'log_kge':1, 'log_r':1, 'log_alpha':2, 'log_beta':2, 
                       'Q0.1pbias':1, 'Q0.2pbias':1, 'Q0.3pbias':1, 
                       'AbsQ0.1pbias':3, 'AbsQ0.2pbias':3, 'AbsQ0.3pbias':3}


# Metric labels
metric_labels = {
    'r' : 'Pearson Correlation',
    'nse' : 'NSE',
    'kge' : 'KGE',
    'alpha' : 'Relative Variability (KGE Alpha)',
    'beta' : 'Relative Bias (KGE Beta)',
    "Q0.1pbias" : 'Percent Bias (Q0.1)',
    "Q0.2pbias" : 'Percent Bias (Q0.2)',
    "Q0.3pbias" : 'Percent Bias (Q0.3)',
    "AbsQ0.1pbias" : 'Absolute Percent Bias (Q0.1)',
    "AbsQ0.2pbias" : 'Absolute Percent Bias (Q0.2)',
    "AbsQ0.3pbias" : 'Absolute Percent Bias (Q0.3)',
}

for k in list(metric_labels.keys()):
    metric_labels[f'log_{k}'] = f'Log Flow {metric_labels[k]}'

metric_limits = {
    'r' : [0.5, 1.0],
    'nse' : [0.0, 1.0],
    'kge' : [0.0, 1.0],
    'alpha' : [0.0, 2.0],
    'beta' : [0.0, 2.0],
    'Q0.1pbias' : [-100, 100],
    'Q0.2pbias' : [-100, 100],
    'Q0.3pbias' : [-100, 100],
    'AbsQ0.1pbias' : [0, 100],
    'AbsQ0.2pbias' : [0, 100],
    'AbsQ0.3pbias' : [0, 100],
}
for k, v in list(metric_limits.items()):
    metric_limits[f'log_{k}'] = v

metric_ideal = {
    'r' : 1.0,
    'nse' : 1.0,
    'kge' : 1.0,
    'alpha' : 1.0,
    'beta' : 1.0,
    'Q0.1pbias' : 0.0,
    'Q0.2pbias' : 0.0,
    'Q0.3pbias' : 0.0,
    'AbsQ0.1pbias' : 0.0,
    'AbsQ0.2pbias' : 0.0,
    'AbsQ0.3pbias' : 0.0,
}

for k, v in list(metric_ideal.items()):
    metric_ideal[f'log_{k}'] = v




model_colors = {
    'obs': 'black',
    'nhmv10' : 'cornflowerblue', 
    'nwmv21' : '#385723',
    'obs_pub_nhmv10' : '#F27300',
    'obs_pub_nhmv10_ObsScaled': '#F27300', 
    'obs_pub_nhmv10_ensemble' : '#F9B572', 
    'obs_pub_nhmv10_ObsScaled_ensemble' : '#F9B572', 
    'obs_pub_nwmv21' : '#0174BE', 
    'obs_pub_nwmv21_ensemble': '#9CD2F6',
    'obs_pub_nwmv21_ObsScaled' : '#0174BE', 
    'obs_pub_nwmv21_ObsScaled_ensemble': '#9CD2F6'
    }

for K in range(2, 11):
    for BC in ['_BC', '']:
        model_colors[f'obs_pub_nhmv10{BC}_K{K}_ensemble'] = '#F9B572'
        model_labels[f'obs_pub_nhmv10{BC}_K{K}_ensemble'] = f'PUB-NHM K{K} Ensemble'
