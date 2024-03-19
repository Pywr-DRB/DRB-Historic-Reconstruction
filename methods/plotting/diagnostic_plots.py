
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from methods.utils.directories import fig_dir
from methods.diagnostics.metrics import group_metric_data_yearbin
from .styles import model_colors, model_labels, metric_labels
from .styles import lower_bound_metric_scores, upper_bound_metric_scores, ideal_metric_scores
from .styles import ax_tick_label_fsize, ax_label_fsize, sup_title_fsize, legend_fsize
from .styles import fig_dpi

def plot_error_slope_subplot(ax, model, metric, error_summary, sites,
                       plot_ensemble = False, 
                       maximize=True):
    """
    Makes a scatter/slope-change plot for 1 error metric 
    for both pub and non-pub variants of a model on a subplot ax.
    """
    ## Specs

    scatter_pt_size = 85
    scatter_alpha = 0.5
    line_alpha = 0.9

    YMIN = lower_bound_metric_scores[metric]
    YMAX = upper_bound_metric_scores[metric]

    # cmap
    cmap = mpl.colormaps.get_cmap('coolwarm_r')
    VMIN = -0.5
    VMAX = 0.5
    
    # Pull data
    metric_error_summary = error_summary.loc[error_summary['metric']==metric, :].copy()
    model_errors = metric_error_summary.loc[metric_error_summary['model']==model, :].copy()
    pub_model_errors = metric_error_summary.loc[metric_error_summary['model']==f'obs_pub_{model}', :].copy()
    pub_ensemble_model_errors = metric_error_summary.loc[metric_error_summary['model']==f'obs_pub_{model}_ensemble', :].copy()
    
    N = len(sites)
    
    # vertical line
    ax.vlines(x=[0, 1], ymin=YMIN, ymax=YMAX, color='grey', linestyle='--')

    ax.scatter([0]*N, model_errors['value'], 
               color=model_colors[model], 
               label=model, 
               s=scatter_pt_size, zorder=5, alpha = scatter_alpha)
    ax.scatter([1]*N, pub_model_errors['value'], 
               color=model_colors[f'obs_pub_{model}'], 
               label=f'obs_pub_{model}',
               s=scatter_pt_size, zorder=5, alpha = scatter_alpha)

    # lines
    for s in sites:
        mod_site_error = model_errors.loc[model_errors['site']==s, 'value'].values[0]
        
        pub_mod_site_error = pub_model_errors.loc[pub_model_errors['site']==s, 'value'].values[0]
        slope = (pub_mod_site_error - mod_site_error) if maximize else (mod_site_error - pub_mod_site_error)
        norm_slope = (slope - VMIN) / (VMAX - VMIN)
        c = cmap(norm_slope)
        ax.plot([0, 1], [mod_site_error, pub_mod_site_error], 
                color=c, alpha=line_alpha, lw=2, zorder = 4)
        
        if plot_ensemble:
            for real in pub_ensemble_model_errors['realization'].unique():
                pub_mod_site_error = pub_ensemble_model_errors.loc[(pub_ensemble_model_errors['site']==s) & 
                                                                    (pub_ensemble_model_errors['realization']==real), 
                                                                    'value'].values[0]
                slope = (pub_mod_site_error - mod_site_error) if maximize else (mod_site_error - pub_mod_site_error)
                norm_slope = ((slope - VMIN) / (VMAX - VMIN))
                c = cmap(norm_slope)
                ax.plot([0, 1], [mod_site_error, pub_mod_site_error], 
                        color=c, alpha=line_alpha-0.3, lw=0.5, 
                        zorder=0)
 
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlim(-0.2, 1.2)
    
    # Convert x ticks to model labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    
    # Reduce yticks to only 0 and 1
    ax.set_yticks([YMIN, YMAX])
    
    if not maximize:
        ax.invert_yaxis()
    
    return ax



def plot_error_cdf_subplot(ax, models, metric, error_summary, 
                   axis_labels = True, legend=False, maximize=True, 
                   plot_median=True, plot_ensemble_range=False):
    """
    Makes a CDF plot of the error metric for each model on a subplot ax.
    """
    # Pull all data for different models
    data = {}
    for model in models:
        mod_data = error_summary.loc[error_summary['model'] == model, :]
        mod_metric_data = mod_data.loc[mod_data['metric'] == metric, :]
        
        if 'ensemble' in model:
            for real in mod_metric_data['realization'].unique():
                data[f'{model}_{real}'] = mod_metric_data.loc[mod_metric_data['realization']==real, 'value'].values
             
        else:
            data[model] = mod_metric_data['value'].values
    
    # Arrange data in rank order
    data_rank = {}
    for m in data.keys():
        if maximize:
            data_rank[m] = np.sort(data[m])
        else:
            data_rank[m] = np.sort(data[m])[::-1]
        
    # Plot CDF
    for model in models:
        if 'ensemble' in model:
            if plot_ensemble_range:
                # Get range of realization values
                realizations = [int(m.split('_')[-1]) for m in data_rank.keys() if model in m]
                
                ensemble_ranks = np.zeros((len(realizations), len(data_rank[f'{model}_0'])))
                for i, real in enumerate(realizations):
                    ensemble_ranks[i, :] = data_rank[f'{model}_{real}']
                
                ensemble_min = np.min(ensemble_ranks, axis=0)
                ensemble_max = np.max(ensemble_ranks, axis=0)
                ax.fill_between(np.linspace(0, 1, len(data_rank[f'{model}_0'])), ensemble_min, ensemble_max, 
                                color=model_colors[model], alpha=0.3, zorder=2)
        
        else:
            ax.plot(np.linspace(0, 1, len(data_rank[model])), data_rank[model],    
                    color=model_colors[model], 
                    label=model_labels[model],
                    lw=3, zorder=3)
        if plot_median:
            if 'ensemble' not in model:
                
                median_value= np.median(data_rank[model])
                ax.scatter(0.5, median_value, color=model_colors[model], 
                           edgecolor='black', linewidth=1,
                           marker="^", s=100, zorder = 5)                
    
    if legend:
        ax.legend()
    if axis_labels:
        ax.set_ylabel(metric_labels[metric], fontsize=ax_label_fsize)
        ax.set_xlabel('Rank')


    ax.set_ylim(lower_bound_metric_scores[metric], 
                upper_bound_metric_scores[metric])
    if not maximize:
        ax.invert_yaxis()
        
    # ax.set_yticklabels([lower_bound_metric_scores[metric], upper_bound_metric_scores[metric]])
    ax.set_yticks([lower_bound_metric_scores[metric], upper_bound_metric_scores[metric]])
    ax.set_yticklabels([lower_bound_metric_scores[metric], upper_bound_metric_scores[metric]], 
                       fontsize= ax_tick_label_fsize)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])

    return ax



def plot_Nx3_error_slope_cdf(error_summary, models, metrics, sites,
                             plot_ensemble=False, fig_dir = fig_dir):
    
    fig, axs = plt.subplots(nrows=len(metrics), ncols= 3, 
                            sharex='col', sharey='row',
                            figsize = (len(metrics)*2.5, len(metrics)*3))

    for i,m in enumerate(metrics):
        
        is_max = False if 'Q' in m else True
        
        axs[i,0] = plot_error_slope_subplot(axs[i,0], 'nhmv10', m, error_summary, sites=sites, 
                                    plot_ensemble=plot_ensemble,
                                    maximize=is_max)
        axs[i,1] = plot_error_slope_subplot(axs[i,1], 'nwmv21', m, error_summary, sites=sites,
                                    plot_ensemble = plot_ensemble, 
                                    maximize=is_max)
        
        axs[i, 2] = plot_error_cdf_subplot(axs[i, 2], models, m, error_summary, 
                                           legend=False, 
                                           axis_labels=False, 
                                           maximize=is_max, plot_median=True, 
                                           plot_ensemble_range=plot_ensemble)
        axs[i,0].set_ylabel(metric_labels[m], fontsize=ax_label_fsize)

    axs[-1, 0].set_xticklabels(['NHMv1.0', 'PUB-NHM'], fontsize=12)
    axs[-1, 1].set_xticklabels(['NWMv2.1', 'PUB-NWM'], fontsize=12)
    axs[-1, 2].set_xlabel('Rank', fontsize=12)
    axs[-1, 2].set_xticklabels(['Worst', 'Best'], fontsize=12)
    
    # Add a colorbar for slope improvement plots
    
    # Get mappable from ax
    im = axs[0, 0].get_children()[0]
    # Make a new axis for the colorbar
    cax = fig.add_axes([0.0, -0.25, 0.5, 0.2])
    # Add colorbar to new axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('PUB Improvement', rotation=270, labelpad=12)
    cbar.set_ticks([-0.5, 0, 0.5])
        
    # plt.suptitle('Difference in metric performance between\nNHM/NWM and corresponding PUB-based predictions', fontsize=14, y=0.95)
    plt.tight_layout()
    fig.align_ylabels(axs[:, 0])
    # Save
    plt.savefig(f'{fig_dir}/diagnostics/Nx3_LeaveOneOut_{metrics}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    return















def plot_binyear_boxplot_subplot(error_summary_annual, 
                                 ax, 
                                 metric='nse', 
                                 model_colors=model_colors, 
                                 bin_size=10,
                                 plot_ensembles= False,
                                 ideal_metric_scores=ideal_metric_scores,
                                 lower_bound_metric_scores=lower_bound_metric_scores, 
                                 upper_bound_metric_scores=upper_bound_metric_scores):
    """Make a seaborn box plot of errors by year bin and model
    """

    metric_data = group_metric_data_yearbin(error_summary_annual, 
                                                    bin_size=bin_size, metric=metric)
    
    
    if plot_ensembles:
        model_hue_order = ['nhmv10', 'obs_pub_nhmv10_ensemble', 'obs_pub_nhmv10',  
                           'nwmv21', 'obs_pub_nwmv21_ensemble', 'obs_pub_nwmv21']
    else:
        model_hue_order = ['nhmv10', 'obs_pub_nhmv10', 
                           'nwmv21', 'obs_pub_nwmv21']
    flierprops = dict(marker='o', markerfacecolor='k', markersize=0.5,
                    linestyle='none')
    
    sns.boxplot(y="value", x="year_bin", hue="model", 
                hue_order = model_hue_order,
                data=metric_data, 
                palette=model_colors,
                saturation=1,
                linewidth=0.7,
                flierprops = flierprops,
                ax=ax)

    # add horizontal line across full boxplot area
    ax.axhline(ideal_metric_scores[metric], 
               ls='-', color='black', linewidth=1, 
               alpha=0.5, 
               zorder=0)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel('')

    ax.legend().set_visible(False)
    ax.set_ylim([lower_bound_metric_scores[metric],
                 upper_bound_metric_scores[metric]])
    # Turn off axis boarder box
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    
    # ax.grid(which='major', axis='y', linestyle='-', alpha=0.5)
    
    return ax




# Test
def plot_Nx1_binyear_boxplots(error_summary_annual, 
                              metrics, 
                              model_colors=model_colors,
                              ideal_metric_scores=ideal_metric_scores,
                              lower_bound_metric_scores=lower_bound_metric_scores,
                              upper_bound_metric_scores=upper_bound_metric_scores,
                              bin_size=10, plot_ensembles= True,
                              fig_dir=fig_dir):

    NROWS = len(metrics)
    # Set up the matplotlib figure and aesthetics
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.ioff()
    
    fig, axs = plt.subplots(nrows=NROWS, ncols=1, 
                            figsize=(8, (NROWS*1.75)), 
                            dpi = 300, sharex=True)
    for i, metric in enumerate(metrics):
        ax = axs[i]
        plot_binyear_boxplot_subplot(error_summary_annual, ax, 
                                     metric, 
                                     model_colors=model_colors,
                                    ideal_metric_scores=ideal_metric_scores, 
                                    lower_bound_metric_scores=lower_bound_metric_scores, 
                                    upper_bound_metric_scores=upper_bound_metric_scores,
                                    bin_size=bin_size,
                                    plot_ensembles=plot_ensembles)
    
    # Make a single legend for the whole plot
    handles, labels = ax.get_legend_handles_labels()
    # Fix labels
    labels = [model_labels[label] for label in labels]
    
    fig.legend(handles, labels, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.1), frameon=False)
    
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/diagnostics/Nx1_binyear_boxplots_{metrics}.svg', 
                dpi = fig_dpi, bbox_inches='tight')
    # plt.show()
    return 





def plot_grid_metric_map(error_summary, donor_model, site_catchment_areas=None):
    VMAX = 0.5
    VMIN = -0.5
    
    # Pull data
    sites = error_summary['site'].unique()
    metrics = error_summary['metric'].unique()
    ideal_scores = 1.0 
    
    # Sort the sites by catchment area
    if site_catchment_areas is not None:
        sites = sites[np.argsort(site_catchment_areas)[::-1]]
    
    # Dimensions of the grid
    n_sites = len(sites)
    n_metrics = len(metrics)
    
    # Make a colormap
    model_colors = ['lightgrey', 'darkgreen']
    # cmap = ListedColormap(model_colors)
    cmap = mpl.colormaps.get_cmap('RdBu')
    
    # Create a NumPy array to store the best model information
    grid = np.zeros((n_sites, n_metrics))
    
    for i, site in enumerate(sites):
        site_errors = error_summary.loc[error_summary['site'] == site, :].copy()

        site_errors['abs_diff'] = abs(site_errors['value'] - 1.0)

        for j, metric in enumerate(metrics):
            site_metric_errors = site_errors.loc[site_errors['metric']==metric, :]
            donor_model_diff = site_metric_errors.loc[site_metric_errors['model']==donor_model, 'abs_diff'].values[0]
            pub_diff = site_metric_errors.loc[site_metric_errors['model']==f'obs_pub_{donor_model}', 'abs_diff'].values[0]
            
            grid[i, j] = donor_model_diff - pub_diff

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid, aspect='auto',
              cmap=cmap, vmin=VMIN, vmax=VMAX)

    # Add grid lines
    for x in range(n_metrics + 1):
        ax.axvline(x - 0.5, color='white', lw=2)
    for y in range(n_sites + 1):
        ax.axhline(y - 0.5, color='white', lw=2)
    
    # Add metric labels
    ax.set_xticks(np.arange(0, n_metrics))
    ax.set_xticklabels(metrics, rotation=90)
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f'Metric Improvement Following PUB Estimation', 
                       rotation=-90, fontsize=16, va='bottom')
    # cbar.yticklabels(fontsize=14)
    plt.savefig(f'{fig_dir}/diagnostics/grid_metric_map_{donor_model}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return
