
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
    'obs_pub_nwmv21' : '#0174BE', 
    'obs_pub_nwmv21_ensemble': '#9CD2F6',
    'obs_pub_nwmv21_ObsScaled' : '#0174BE', 
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
    'obs_pub_nwmv21_ObsScaled' : 'PUB-NWM',
    'obs_pub_nwmv21_ObsScaled_ensemble':'PUB-NWM Ensemble'
    }

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
metric_labels = {'nse': 'NSE', 'kge': 'KGE', 'r': 'Pearson Correlation', 'alpha': 'Alpha', 'beta': 'Beta',
                 'log_nse': 'Log-NSE', 'log_kge': 'Log-KGE', 'log_r': 'Log Pearson Correlation', 
                 'log_alpha': 'Log-Alpha', 'log_beta': 'Log-Beta', 
                 'Q0.1pbias': '0.1% Bias', 'Q0.2pbias': '0.2% Bias', 'Q0.3pbias': '0.3% Bias', 
                 'AbsQ0.1pbias': '0.1% Abs. Bias', 'AbsQ0.2pbias': '0.2% Abs. Bias', 
                 'AbsQ0.3pbias': '0.3% Abs. Bias'}
