
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from methods.utils.directories import fig_dir
from .styles import model_colors, model_labels
from .styles import lower_bound_metric_scores, upper_bound_metric_scores
from .styles import ax_tick_label_fsize, ax_label_fsize, sup_title_fsize, legend_fsize
from .styles import fig_dpi

def plot_data_summary(Q_obs_unmanaged,
                           unmanaged_gauge_meta, 
                           diagnostic_site_list, 
                           pywrdrb_node_metadata, 
                           drb_boundary,
                           sort_by = 'lat'):
    
    diagnostic_site_ax = False
    date_min = 1945
    date_max = 2022
    n_years = date_max - date_min
    
    marker_types = {'gauge': '.',
                    'diagnostic_site': '.', 
                    'pywrdrb_node': '^'}
    marker_colors = {'gauge': 'darkblue',
                    'diagnostic_site': 'darkorange',
                    'pywrdrb_node': 'cornflowerblue'}
    marker_sizes = {'gauge': 50,
                    'diagnostic_site': 60,
                    'pywrdrb_node': 35}

    # Define the custom colormap
    colors = ["white", "darkblue"]  # White for 0, blue for 1
    cmap_name = "custom_cmap"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    NCOLS = 2
    width_ratios = [1, 1.5]
    fig, axs = plt.subplots(nrows=1, ncols=NCOLS, figsize=(7, 5), 
                                   dpi = fig_dpi,
                                   gridspec_kw={'width_ratios': width_ratios,
                                                'height_ratios': [1]})
    
    ax1 = axs[0]
    ax2 = axs[1]
        
    ### Subplot 1: Map of sites
    drb_boundary.plot(ax=ax1, color='white', edgecolor='dimgrey', lw=0.5,
                      label = 'DRB Boundary', zorder=0)
    # drb_mainstem.plot(ax=ax, color='black', edgecolor='black', lw=2)
    
    # Plot gauges
    ax1.scatter(unmanaged_gauge_meta.loc[:, 'long'], unmanaged_gauge_meta.loc[:, 'lat'], 
            c=marker_colors['gauge'],
            marker=marker_types['gauge'],
            edgecolor=marker_colors['gauge'],
            s=marker_sizes['gauge'], 
            lw=1, zorder=4, alpha = 0.75, 
            label ='Unmanaged USGS Gauges')
    
    # Plot diagnostic sites
    ax1.scatter(unmanaged_gauge_meta.loc[diagnostic_site_list, 'long'], 
                unmanaged_gauge_meta.loc[diagnostic_site_list, 'lat'],
               c= marker_colors['gauge'],
               edgecolor = marker_colors['diagnostic_site'], 
               s=marker_sizes['diagnostic_site'], 
               marker=marker_types['diagnostic_site'],
               lw=1, 
               zorder=5, alpha = 0.85,
               label='Diagnostic Gauge Prediction Locations')
    
    # Plot pywrdrb nodes
    ax1.scatter(pywrdrb_node_metadata.loc[:, 'long'], pywrdrb_node_metadata.loc[:, 'lat'],
               c=marker_colors['pywrdrb_node'], 
               s=marker_sizes['pywrdrb_node'], 
               marker=marker_types['pywrdrb_node'],
               zorder=2, alpha = 0.85,
               label='Pywr-DRB Prediction Locations')
    
    # Improve ticks and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ### Subplot 2: Length of records
    Q_obs_unmanaged = Q_obs_unmanaged.loc[f'{date_min}-01-01':f'{date_max}-12-31', :]
    record_lengths =pd.to_datetime(unmanaged_gauge_meta['end_date']) -  pd.to_datetime(unmanaged_gauge_meta['begin_date'])

    # Get NA count
    site_na_counts = Q_obs_unmanaged.isna().sum()
    total_length = len(Q_obs_unmanaged)
    na_fraction = site_na_counts/total_length
    na_fraction.index = [s.split('-')[1] for s in na_fraction.index]
    na_fraction.sort_values(ascending=True, inplace=True)
    bad_sites = []
    for station in na_fraction.index:
        if na_fraction[station] > 0.85:
            bad_sites.append(station)
            
    # Sort
    sorted_lat = unmanaged_gauge_meta.loc[:, 'lat'].sort_index(ascending=False)
    sorted_record_lengths = record_lengths.sort_values(ascending=False)
    sorted_flows = Q_obs_unmanaged.loc[:, [f'USGS-{s}' for s in sorted_record_lengths.index]]
    if sort_by == 'record_length':
        sorted_index = sorted_record_lengths.index.drop(bad_sites)
    elif sort_by == 'lat':
        sorted_index = sorted_lat.index.drop(bad_sites)
    elif sort_by == 'na_fraction':
        sorted_index = na_fraction.index.drop(bad_sites)

    # Make grid of 0,1 for data availability
    grid = np.zeros((len(sorted_index), n_years))
    for i, site in enumerate(sorted_index):
        for j in range(n_years):
            annual_flow = Q_obs_unmanaged.loc[f'{date_min+j}-01-01':f'{date_min+j}-12-31', f'USGS-{site}']
            na_count = annual_flow.isna().sum()            
            if na_count < 20:
                grid[i, j] = 1.0

    ## Plotting timeseries grid
    ax2.imshow(grid, cmap=custom_cmap, aspect='auto', norm=mcolors.Normalize(vmin=0, vmax=1))

    ## Format
    # Set axis ticks as years
    tick_spacing = 15
    ax2.set_xticks(range(0, n_years, tick_spacing))
    ax2.set_xticklabels(range(date_min, date_max, tick_spacing), fontsize=ax_tick_label_fsize)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # ax2.set_xlabel('Year', fontsize=ax_label_fsize)
    # ax2.set_ylabel(f'USGS Gauge Stations (N = {len(sorted_index)})', fontsize=12)

    # Create custom patches for the legend
    data_available_patch = mpatches.Patch(color='blue', label='Data available')
    no_data_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='No data')
    
    # Collect legend handles and labels from ax1
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()

    # Combine the handles and labels from both axes
    handles = handles_ax1 + [data_available_patch, no_data_patch]
    labels = labels_ax1 + ['Data available', 'No data']

    # Create a single legend for the figure
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=2, 
               fontsize=legend_fsize)
    
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(wspace=0.01)
    fig.align_labels()
    plt.savefig(f'{fig_dir}/data_summary_sortby_{sort_by}.svg', 
                bbox_inches='tight', dpi=fig_dpi)
    plt.show()
    return