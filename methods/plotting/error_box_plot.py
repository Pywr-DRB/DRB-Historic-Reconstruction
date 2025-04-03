import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from methods.plotting.styles import model_colors, model_labels, metric_labels, metric_limits, metric_ideal

def plot_diagnostic_error_box_plot(error_summary, 
                                   metric, 
                                   model,
                                   fname):
    ### Site-to-site metric change comparison
    # Sites are ranked based on a specific 'metric' and corresponding 'value'
    # The CDF of site metric values is plotted for the obs-pub data as scatter points

    figsize = (4.5,4)

    pt_size = 20

    nxm_model = 'nhmv10'

    fig, ax = plt.subplots(figsize=figsize)

    # sort sites from highest to lowest median value across ensemble realizations
    pub_df = error_summary[error_summary['model'].isin([model])]
    pub_df = pub_df[pub_df['metric'] == metric]
    pub_medians = pub_df.drop(['metric', 'model'], axis=1).groupby(['site']).median()

    nxm_df = error_summary[error_summary['model'].isin([nxm_model])]
    nxm_df = nxm_df[nxm_df['metric'] == metric]
    nxm_medians = nxm_df.drop(['metric', 'model'], axis=1).groupby('site').median()
    nxm_medians = nxm_medians.loc[pub_medians.index, :]

    sites_sorted = pub_medians.sort_values(by='value', ascending=False).index


    xs = np.arange(len(sites_sorted))

    for site in sites_sorted:
        
        ys = pub_df[pub_df['site']==site]['value'].values    
        ax.boxplot(ys, positions=[xs[sites_sorted.get_loc(site)]], 
                widths=1, 
                showfliers=False,
                zorder=1,
                patch_artist=True,
                boxprops= dict(linewidth=0.5, 
                                facecolor='darkorange',
                                color='black'), 
                whiskerprops=dict(linestyle='-',linewidth=0.5, color='darkorange'),
                medianprops=dict(color='black'))
            
    
        # add NHM error    
        ys = nxm_medians.loc[site]['value']
        ax.scatter(xs[sites_sorted.get_loc(site)], ys, 
                color=model_colors[nxm_model], s=pt_size, 
                lw=0.5, zorder=10,
                edgecolors='k', label='NHM Model Estimate')

    # add ideal value
    ax.hlines(metric_ideal[metric], xmin=0, xmax=len(xs), 
              color='k', lw=3, ls='--', label='Ideal Value')
    ax.set_ylim(metric_limits[metric])
    ax.set_ylabel(metric_labels[metric], fontsize=14)

    ax.set_xlabel('Site Rank', fontsize=14)
    ax.set_xticks(range(1, len(xs), 10))
    ax.set_xticklabels(range(1, len(xs), 10), fontsize=12)
    ax.set_xlim([-1.1, len(xs)+0.1])
    
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()
    return