import pandas as pd
import os
from config import OUTPUT_DIR, FIG_DIR
from methods.plotting.error_box_plot import plot_diagnostic_error_box_plot
from methods.plotting.diagnostic_site_aggregate_flow import plot_aggregate_weekly_flow_diagnostic_sites, aggregate_sites_weekly_flow

from methods.diagnostics.metrics import error_metrics
from methods.diagnostics.metrics import get_leave_one_out_filenames

from methods.processing.load import load_leave_one_out_datasets

from methods.load.data_loader import Data
from methods.utils.constants import mg_to_mcm

plot_metrics = error_metrics

FIG_DIR = f"{FIG_DIR}diagnostics/"

if __name__ == '__main__':
    ############################
    ### Loading
    ############################
    
    data_loader = Data()
    
    ### Streamflow
    # Load observed and NHM streamflow
    Q_nhm = data_loader.load(datatype='streamflow', 
                                flowtype='nhm', sitetype='diagnostic')
    Q_obs = data_loader.load(datatype='streamflow', 
                                flowtype='obs', sitetype='diagnostic')
    Q_obs = Q_obs.loc["1983-10-01":"2016-12-31", :]

    print(f"Q obs loaded with shape {Q_obs.shape}")


    # Get list of leave-one-out ensemble filenames
    loo_filenames, loo_models = get_leave_one_out_filenames()
    agg_models = [
                  'obs_pub_nhmv10_BC_K5_ensemble', 
                  'obs_pub_nhmv10_K5_ensemble',
                  ]
    agg_model_filenames = [] 
    for agg_model in agg_models:
        for f in loo_filenames:
            if agg_model.split("pub")[1] in f:
                agg_model_filenames.append(f)

    loo_sites = list(Q_obs.columns)
    
    Q = load_leave_one_out_datasets(agg_model_filenames, agg_models)
    Q['obs'] = Q_obs
    Q['nhmv10'] = Q_nhm
    
    # leave-one-out error metric summary
    err_summary_file = f"{OUTPUT_DIR}LOO/loo_error_summary.csv"
    error_summary = pd.read_csv(err_summary_file, index_col=0, dtype={'site': str})
    models = error_summary['model'].unique()
    
    ############################
    ### Plotting
    ############################
    
    ### Plot aggregate weekly flow
    print('Plotting aggregate weekly flow')
    agg_models += ['obs', 'nhmv10']
    
    # convert Q to MCM
    all_keys = Q.keys()
    Q_mcm = {}
    for key in all_keys:
        
        # apply to each realization in the ensemble
        if 'ensemble' in key:
            Q_mcm[key] = {}
            for site in Q_mcm['obs'].columns:
                Q_mcm[key][site] = Q[key][site] * mg_to_mcm
        else:
            Q_mcm[key] = Q[key] * mg_to_mcm

    # get basin agg weekly flow, using Q with MCM units
    Q_weekly = aggregate_sites_weekly_flow(Q_mcm, agg_models)
    
    # Get the NHM bias during week with lowest flow
    low_flow_idx = Q_weekly['obs_week_sum'].idxmin()
    nhm_low_flow = Q_weekly['nhmv10_week_sum'].loc[low_flow_idx]
    obs_low_flow = Q_weekly['obs_week_sum'].loc[low_flow_idx]
    nhm_weekly_bias = (nhm_low_flow - obs_low_flow) / obs_low_flow * 100
    print(f"STAT: NHM bias during week with lowest flow: {nhm_weekly_bias:.2f}%")
    
    plot_aggregate_weekly_flow_diagnostic_sites(Q_weekly, f'{FIG_DIR}loo_aggregate_weekly_flow.svg')
    
    ### Plot error box plot for each model
    plot_models = models

    for model in plot_models:
        
        if model == "nhmv10":
            continue
        
        fig_dir = FIG_DIR + f"{model}/"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            
        for metric in plot_metrics:

            plot_diagnostic_error_box_plot(error_summary, metric, model, 
                                           f'{fig_dir}loo_error_{metric}.svg')
    
    
    

    print('Done')