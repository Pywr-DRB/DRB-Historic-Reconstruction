import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.plotting.styles import model_colors, model_labels

import os
from config import OUTPUT_DIR, FIG_DIR

def aggregate_sites_weekly_flow(Q, models):
        
    Q_weekly = {}
    agg_sites = Q["obs"].columns.tolist()
    
    for model in models:
        new_label = model + '_week_sum'
        
        if 'obs_pub' in model:
            weekly_sums = None
            weekly_means = None
            for site in agg_sites:
                t_subset = Q['obs'].index.intersection(Q[model][site].index)
                Q_subset = Q[model][site].loc[t_subset, :]
                
                if weekly_sums is None:
                    weekly_sums = Q_subset.resample('W-MON').sum()
                    weekly_means = weekly_sums.groupby(weekly_sums.index.isocalendar().week,
                                                         as_index=False).median()
                else:
                    weekly_sums += Q_subset.resample('W-MON').sum()
                    weekly_means = weekly_sums.groupby(weekly_sums.index.isocalendar().week, 
                                                   as_index=False).median()
            Q_weekly[new_label] = weekly_means
        else:
            t_subset = Q['obs'].index.intersection(Q[model].index)
            Q_subset = Q[model].loc[t_subset, agg_sites]
            
            Q_subset = Q_subset.sum(axis=1)
            
            weekly_sums = Q_subset.resample('W-MON').sum()
            weekly_means = weekly_sums.groupby(weekly_sums.index.isocalendar().week).median()
            Q_weekly[new_label] = weekly_means
    return Q_weekly

def plot_aggregate_weekly_flow_diagnostic_sites(Q, fname,
                                                units='MCM'):

    ### Plot hydrograph of aggregate flow across all diagnostic sites
    # Shows the range of aggregate flow for ensemble
    # Shows single aggregate flow for NHM and Obs

    xs = np.arange(53)

    fig, ax = plt.subplots(figsize=(12,4))


    ax.plot(xs, Q['obs_week_sum'], c='k', ls='--', lw=3.0, label='Observed Flows', zorder=10)
    ax.plot(xs, Q['nhmv10_week_sum'], c=model_colors["nhmv10"], label='NHM Model Flows', zorder=8)

    y_lo= np.array(Q['obs_pub_nhmv10_BC_K5_ensemble_week_sum'].min(axis=1).values)
    y_hi= np.array(Q['obs_pub_nhmv10_BC_K5_ensemble_week_sum'].max(axis=1).values)
    y_lo = y_lo.astype(float)
    y_hi = y_hi.astype(float)

    ax.fill_between(xs, y_hi, y_lo, 
                    color='darkorange', label='Bias-Corrected Reconstruction Range')


    y_lo= np.array(Q['obs_pub_nhmv10_K5_ensemble_week_sum'].min(axis=1).values)
    y_hi= np.array(Q['obs_pub_nhmv10_K5_ensemble_week_sum'].max(axis=1).values)
    y_lo = y_lo.astype(float)
    y_hi = y_hi.astype(float)

    ax.fill_between(xs, y_hi, y_lo, color='gold', label='Non-Bias-Corrected Reconstruction Range')


    
    # ax.set_xlabel('Week of Year')
    ax.set_xlim([0,52])
    ax.set_yscale('log')
    
    # get ylim
    y_min, y_max = ax.get_ylim()
    # round y_min and y_max to nearest 10^x
    y_min = np.floor(np.log10(y_min))
    y_max = np.ceil(np.log10(y_max))
    
    # # set ylim
    # ax.set_ylim([10**y_min, 10**y_max])
    ax.set_ylabel(f'Median Weekly Flow ({units})\nSum Across All Diagnostic Sites', fontsize=14)
    
    # Re-format x tick labels
    x_ticks = ax.get_xticks()
    weeks = [0, 4, 8, 13, 17, 22, 26, 31, 35, 40, 44, 48, 52]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

    ax.set_xticks(weeks, months)
    ax.set_xticks(np.arange(52), ['']*52, minor=True)


    plt.legend()
    
    plt.savefig(fname, dpi=250)
    plt.savefig(fname.replace('.png', '.svg'))
    return

    