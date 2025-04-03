
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import MultiLineString
from shapely import ops


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from methods.utils.directories import fig_dir
from methods.plotting.styles import ax_tick_label_fsize, ax_label_fsize, legend_fsize
from methods.plotting.styles import fig_dpi

from config import GEO_CRS
from config import DATA_DIR

marker_types = {'gauge': '.',
                'diagnostic_site': '.', 
                'pywrdrb_node': '^'}
marker_colors = {'gauge': 'darkblue',
                'diagnostic_site': 'darkorange',
                'pywrdrb_node': 'darkorange'}
marker_sizes = {'gauge': 100,
                'diagnostic_site': 95,
                'pywrdrb_node': 100}


def load_nhd_data(spatial_data_dir = f"{DATA_DIR}/DRB_spatial",
                  crs = GEO_CRS):
    nhd = gpd.read_file(f"{spatial_data_dir}/NHD_0204/Shape/NHDFlowline.shp").to_crs(
        crs
    )
    return nhd

def get_mainstem_geometry_from_nhd(nhd, simplify=True):
    ### get river network from NHD
    mainstem = nhd.loc[nhd["gnis_name"] == "Delaware River"]
    ## for river/stream objects, merge rows into single geometry to avoid white space on plot
    multi_linestring = MultiLineString([ls for ls in mainstem["geometry"].values])
    merged_linestring = ops.linemerge(multi_linestring)
    mainstem = gpd.GeoDataFrame({"geometry": [merged_linestring]})
    
    if simplify:
        mainstem = mainstem.simplify(0.01)
        
    return mainstem


def get_tributary_geometry_from_nhd(nhd, simplify=True):
        ### get all other tributary streams containing a Pywr-DRB reservoir, or downstream of one. Note that 2 different regions have Tulpehocken Creek - use only the correct one.
    trib_names = [
        "West Branch Delaware River",
        "East Branch Delaware River",
        "Neversink River",
        "Mongaup River",
        "Lackawaxen River",
        "West Branch Lackawaxen River",
        "Wallenpaupack Creek",
        "Lehigh River",
        "Shohola Creek",
        "Pohopoco Creek",
        "Merrill Creek",
        "Musconetcong River",
        "Pohatcong Creek",
        "Tohickon Creek",
        "Assunpink Creek",
        "Schuylkill River",
        "Maiden Creek",
        "Tulpehocken Creek",
        "Still Creek",
        "Little Schuylkill River",
        "Perkiomen Creek",
    ]
    tribs = []
    for trib_name in trib_names:
        trib = nhd.loc[
            [
                (n == trib_name) and ((n != "Tulpehocken Creek") or ("02040203" in c))
                for n, c in zip(nhd["gnis_name"], nhd["reachcode"])
            ]
        ]
        multi_linestring = MultiLineString([ls for ls in trib["geometry"].values])
        merged_linestring = ops.linemerge(multi_linestring)
        trib = gpd.GeoDataFrame({"geometry": [merged_linestring], "name": trib_name})
        
        if simplify:
            trib = trib.simplify(0.01)
            
        tribs.append(trib)
    return tribs



def plot_spatial_data_distribution(ax,
                                   unmanaged_gauge_meta,
                                   pywrdrb_node_metadata,
                                   drb_boundary,
                                   diagnostic_site_list=None,
                                   marker_colors=marker_colors,
                                   marker_sizes=marker_sizes,
                                   marker_types=marker_types,
                                   plot_mainstem = True,
                                   plot_tributaries = True,
                                   highlight_diagnostic_sites = False,
                                   ):
    
    
    drb_boundary.plot(ax=ax, 
                      color='white', edgecolor='dimgrey', 
                      lw=0.6,
                      label = 'DRB Boundary', zorder=0)
    if plot_mainstem or plot_tributaries:
        nhd = load_nhd_data()
    if plot_mainstem:
        drb_mainstem = get_mainstem_geometry_from_nhd(nhd)
        drb_mainstem.plot(ax=ax, color='black', edgecolor='black', lw=1.0)
    if plot_tributaries:
        drb_tribs = get_tributary_geometry_from_nhd(nhd)
        for trib in drb_tribs:
            trib.plot(ax=ax, color='black', edgecolor='black', lw=1.0)
    
    # Plot gauges
    ax.scatter(unmanaged_gauge_meta.loc[:, 'long'], 
               unmanaged_gauge_meta.loc[:, 'lat'], 
            c=marker_colors['gauge'],
            marker=marker_types['gauge'],
            edgecolor=marker_colors['gauge'],
            s=marker_sizes['gauge'], 
            lw=1, zorder=4, alpha = 0.75, 
            label ='USGS Gauge')
    
    # Plot diagnostic sites
    if highlight_diagnostic_sites and diagnostic_site_list is not None:
        ax.scatter(unmanaged_gauge_meta.loc[diagnostic_site_list, 'long'], 
                    unmanaged_gauge_meta.loc[diagnostic_site_list, 'lat'],
                c= marker_colors['diagnostic_site'],
                edgecolor = marker_colors['gauge'], 
                s=marker_sizes['diagnostic_site'], 
                marker=marker_types['diagnostic_site'],
                lw=1.5, 
                zorder=5, alpha = 1,
                label='USGS Gauge - Diagnostic Comparison Site')
    
    # Plot pywrdrb nodes
    ax.scatter(pywrdrb_node_metadata.loc[:, 'long'],
                pywrdrb_node_metadata.loc[:, 'lat'],
               c=marker_colors['pywrdrb_node'], 
               s=marker_sizes['pywrdrb_node'], 
               marker=marker_types['pywrdrb_node'],
               zorder=10, alpha = 0.85,
               label='Pywr-DRB Prediction Locations')
    
    # Improve ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax


def plot_temporal_data_distribution(ax, Q_obs,
                                    unmanaged_gauge_meta,
                                    cmap = 'Blues',
                                    sort_by = 'lat',
                                    date_min = 1945,
                                    date_max = 2022,):
    
    n_years = date_max - date_min
    
    Q_obs_unmanaged = Q_obs.loc[:, unmanaged_gauge_meta['site_no']]
    Q_obs_unmanaged = Q_obs_unmanaged.loc[f'{date_min}-01-01':f'{date_max}-12-31', :]
    record_lengths =pd.to_datetime(unmanaged_gauge_meta['end_date']) -  pd.to_datetime(unmanaged_gauge_meta['begin_date'])

    # Get NA count
    site_na_counts = Q_obs_unmanaged.isna().sum()
    total_length = len(Q_obs_unmanaged)
    na_fraction = site_na_counts/total_length
    if '-' in na_fraction.index[0]:
        na_fraction.index = [s.split('-')[1] for s in na_fraction.index]
    na_fraction.sort_values(ascending=True, inplace=True)
    bad_sites = []
    for station in na_fraction.index:
        if na_fraction[station] > 0.85:
            bad_sites.append(station)
            
    # Sort
    sorted_lat = unmanaged_gauge_meta.loc[:, 'lat'].sort_index(ascending=False)
    sorted_record_lengths = record_lengths.sort_values(ascending=False)

    if sort_by == 'record_length':
        sorted_index = sorted_record_lengths.index
    elif sort_by == 'lat':
        sorted_index = sorted_lat.index
    elif sort_by == 'na_fraction':
        sorted_index = na_fraction.index
        
    for st in bad_sites:
        if st in sorted_index:
            sorted_index = sorted_index.drop(st)

    # Make grid of 0,1 for data availability
    grid = np.zeros((len(sorted_index), n_years))
    for i, site in enumerate(sorted_index):
        for j in range(n_years):
            annual_flow = Q_obs_unmanaged.loc[f'{date_min+j}-01-01':f'{date_min+j}-12-31', site]
            na_count = annual_flow.isna().sum()            
            if na_count < 20:
                grid[i, j] = 1.0

    ## Plotting timeseries grid
    ax.imshow(grid, cmap=cmap, aspect='auto', norm=mcolors.Normalize(vmin=0, vmax=1))

    ## Format
    # Set axis ticks as years
    tick_spacing = 15
    ax.set_xticks(range(0, n_years, tick_spacing))
    ax.set_xticklabels(range(date_min, date_max, tick_spacing), fontsize=ax_tick_label_fsize)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.set_xlabel('Year', fontsize=ax_label_fsize)
    ax.set_ylabel(f'USGS Gauge Stations (N = {len(sorted_index)})', fontsize=ax_label_fsize)
    return ax    


def plot_data_summary(Q_obs_unmanaged,
                           unmanaged_gauge_meta, 
                           diagnostic_site_list, 
                           pywrdrb_node_metadata, 
                           drb_boundary,
                           plot_mainstem = True,
                           plot_tributaries = True,
                           sort_by = 'lat',
                           highlight_diagnostic_sites = True):
    
    ### Create figure
    NCOLS = 2
    width_ratios = [1, 1.5]
    fig, axs = plt.subplots(nrows=1, ncols=NCOLS, figsize=(10, 7), 
                                   dpi = fig_dpi,
                                   gridspec_kw={'width_ratios': width_ratios,
                                                'height_ratios': [1]})
    
    ax1 = axs[0]
    ax2 = axs[1]
        
    ### Subplot 1: Map of sites
    ax1 = plot_spatial_data_distribution(ax1,
                                            unmanaged_gauge_meta=unmanaged_gauge_meta,
                                            pywrdrb_node_metadata=pywrdrb_node_metadata,
                                            drb_boundary=drb_boundary,
                                            diagnostic_site_list=diagnostic_site_list,
                                            plot_mainstem = plot_mainstem,
                                            plot_tributaries = plot_tributaries,
                                            highlight_diagnostic_sites = highlight_diagnostic_sites)
    
    ### Subplot 2: Length of records
    # Define the custom colormap
    colors = ["white", "darkblue"]  # White for 0, blue for 1
    cmap_name = "custom_cmap"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    date_min = 1945
    date_max = 2022
    ax2 = plot_temporal_data_distribution(ax2, Q_obs_unmanaged,
                                          unmanaged_gauge_meta,
                                          cmap = custom_cmap,
                                          sort_by = sort_by,
                                          date_min = date_min,
                                          date_max = date_max)

    # Create custom patches for the legend
    data_available_patch = mpatches.Patch(color="darkblue", label='Data Available')
    no_data_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='No Data')
    
    # Collect legend handles and labels from ax1
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()

    # Combine the handles and labels from both axes
    handles = handles_ax1 + [data_available_patch, no_data_patch]
    labels = labels_ax1 + ['Data Available', 'No Data']

    # Create a single legend for the figure
    fig.legend(handles, labels, 
               loc='upper center',
               bbox_to_anchor=(0.5, -0.05), 
               ncol=2, 
               fontsize=legend_fsize)

    # Set bounding boxes around axes as dark grey
    spine_color = 'black'
    for ax in axs:
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color(spine_color)
            
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(wspace=0.01)
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/data_summary_sortby_{sort_by}.svg', 
                bbox_inches='tight', dpi=fig_dpi)
    plt.savefig(f'{fig_dir}/data_summary_sortby_{sort_by}.png', 
            bbox_inches='tight', dpi=fig_dpi)
    plt.show()
    return