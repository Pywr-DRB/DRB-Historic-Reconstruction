"""
Contains functions for plotting PywrDRB ensemble results.

Includes:
- plot_ensemble_nyc_storage

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

from methods.utils.constants import mg_to_mcm
from sklearn.preprocessing import MinMaxScaler

from pywrdrb.pywr_drb_node_data import upstream_nodes_dict, downstream_node_lags, immediate_downstream_nodes_dict

from pywrdrb.post.ensemble_metrics import ensemble_mean
from pywrdrb.plotting.styles import model_label_dict, model_colors_historic_reconstruction
from pywrdrb.plotting.styles import base_model_colors
from pywrdrb.plotting.styles import model_colors_historic_reconstruction

from pywrdrb.utils.constants import delTrenton_target, delMontague_target
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, drbc_lower_basin_reservoirs
from pywrdrb.utils.directories import input_dir, fig_dir 
from pywrdrb.utils.reservoir_data import get_reservoir_capacity
from pywrdrb.utils.timeseries import subset_timeseries

input_dir = input_dir.split("site-packages")[0]

colors = ['#2166ac', '#4393c3', '#92c5de', 
            '#d1e5f0', '#f6e8c3', '#dfc27d', 
            '#bf812d', '#8c510a', '#F8F8F8']

flow_colors = {
    "NYC Flood Release" : '#2166ac',
    "NYC Spill" : '#4393c3',
    "NYC Minimum Daily Release" : '#92c5de',
    "NYC Directed Montague Release" : '#d1e5f0',
    "NYC Trenton Equiv. Flow Release" : '#BF92D2', #'#F8F8F8',
    "Uncontrolled Flow" : '#f6e8c3',
    "Non-NYC Normal Release" : '#dfc27d',
    "Non-NYC Trenton Equiv. Flow Release" : '#bf812d',
}

def get_subplot_handles_and_labels(axs):
    # Gather legend handles and labels from all subplots and combine into single legend
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # get only unique handles and labels
    handles, labels = np.array(handles), np.array(labels)
    idx = np.unique(labels, return_index=True)[1]
    handles, labels = handles[idx], labels[idx]
    return handles, labels


def create_mirrored_cmap(cmap_name):
    original_cmap = plt.cm.get_cmap(cmap_name)
    reversed_cmap = original_cmap.reversed()
    combined_colors = np.vstack((original_cmap(np.linspace(0, 1, 128)),
                                 reversed_cmap(np.linspace(0, 1, 128))))
    mirrored_cmap = mcolors.LinearSegmentedColormap.from_list('mirrored_' + cmap_name, combined_colors)
    return mirrored_cmap


def clean_xtick_labels(axes, start_date, end_date, 
                       fontsize=10, date_format='%Y', 
                       max_ticks=10, rotate_labels=False):
    """
    Clean up x-axis tick labels for time series data.
    """
    try:
        start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        if start_date >= end_date:
            raise ValueError(f"Start date must be before end date. Start: {start_date}, End: {end_date}")

        total_days = (end_date - start_date).days

        if total_days <= 30:
            date_format = '%Y-%m-%d'
            tick_spacing = 'D'
        elif total_days <= 365 * 2:
            date_format = '%Y-%m'
            tick_spacing = 'MS'
        elif total_days <= 365 *6:
            date_format = '%Y'
            tick_spacing = '1YS'
        elif total_days <= 365 * 10:
            date_format = '%Y'
            tick_spacing = '2YS'            
        elif total_days <= 365 * 20:
            # Space every 5 years
            date_format = '%Y'
            tick_spacing = '5YS'
        else:
            # Space every 10 years
            date_format = '%Y'
            tick_spacing = '10YS'

        use_ticks = pd.date_range(start_date, end_date, freq=tick_spacing)
        tick_labels = [t.strftime(date_format) for t in use_ticks]

        for i in range(len(axes)):
            ax=axes[i]
            ax.set_xticks(use_ticks)
            ax.set_xticklabels(tick_labels, 
                               rotation=45 if rotate_labels else 0, 
                               fontsize=fontsize, ha='center')
            ax.tick_params(axis='x', which='minor', length=0)
            ax.xaxis.set_minor_locator(plt.NullLocator())

            # Adjust layout to ensure labels are not cut off
            ax.figure.tight_layout()

    except Exception as e:
        print(f"Error in setting tick labels: {e}")

    return axes


def plot_ensemble_percentile_cmap(ensemble_df, model, ax, 
                                  q_upper_bound, q_lower_bound, 
                                  alpha=1, zorder=4):
    mirror_cmap = create_mirrored_cmap('Oranges') if 'nhm' in model else create_mirrored_cmap('Blues')
    norm = Normalize(vmin=-0.2, vmax=1.2)

    percentiles = np.linspace(q_lower_bound, 0.5, 50) 
    delta_percentile = percentiles[1] - percentiles[0]
    
    # Add a line at the median value
    median = ensemble_df.median(axis=1) 
    ax.plot(ensemble_df.index, median, 
            color=mirror_cmap(norm(0.5)), 
            ls='-', zorder=zorder-1, lw=0.3)
    
    for i in range(len(percentiles)-1):
        lower = ensemble_df.quantile(percentiles[i], axis=1, interpolation='linear')
        upper = ensemble_df.quantile(1-percentiles[i], axis=1, interpolation='linear')
        
        
        
        if (i != len(percentiles)-1) or (i !=0):
            if not np.isclose(lower.values, upper.values).all():            
                ax.fill_between(ensemble_df.index,
                                lower,
                                upper,
                                color= mirror_cmap(norm(percentiles[i])),
                                interpolate=False, 
                                edgecolor = mirror_cmap(norm(percentiles[i])),
                                alpha=alpha, zorder=zorder, lw=0.05)
            else:
                print(f'Skipping {percentiles[i]}')
        else:
            ax.fill_between(ensemble_df.index,
                            lower,
                            upper,
                            color= mirror_cmap(norm(percentiles[i])),
                            interpolate=False, 
                            edgecolor = mirror_cmap(norm(percentiles[i])),
                            alpha=alpha, zorder=zorder, lw=1,
                            label=model_label_dict[model])
    return ax




def plot_ensemble_nyc_storage(storages, 
                              ffmp_level_boundaries,
                              models, 
                              colordict = model_colors_historic_reconstruction,
                              start_date = '1999-10-01', end_date = '2010-05-31', 
                              fig_dir=fig_dir, 
                              percentile_cmap=False,
                              q_lower_bound = 0.05,
                              q_upper_bound = 0.95,
                              fill_ffmp_levels = True, 
                              plot_observed = True,
                              ax = None,
                              legend=True,
                              ensemble_fill_alpha = 0.8,
                              smoothing_window=1,
                              fontsize=10,
                              ffmp_fill_alpha = 0.3,
                              dpi=200,
                              units='MGD'):    
    """
    """

    if ax is None:
        fig, ax = plt.subplots(1,1,
                               figsize=(8, 3.5),
                               dpi = dpi)
        is_subplot = False
    else:
        is_subplot = True
    

    ### get reservoir storage capacities
    if units == 'MGD':
        capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    elif units in ['MCM', 'MCM/d']:
        capacities = {r: get_reservoir_capacity(r)*mg_to_mcm for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    ffmp_level_boundaries['level1a'] = 100.


    ### First plot FFMP levels as background color
    levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
    level_labels = {
        'level1a': 'Flood A',
        'level1b': 'Flood B',
        'level1c': 'Flood C',
        'level2': 'Normal',
        'level3': 'Drought Warning',
        'level4': 'Drought Watch',
        'level5': 'Drought Emergency'
    }
    level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
                    ['papayawhip'] +\
                    [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
    level_alpha = [0.5]*3 + [0.5] + [0.5]*3
    x = ffmp_level_boundaries.index
    
    if fill_ffmp_levels:    
        for i in range(len(levels)):
            y0 = ffmp_level_boundaries[levels[i]]
            if i == len(levels)-1:
                y1 = 0.
            else:
                y1 = ffmp_level_boundaries[levels[i+1]]
            ax.fill_between(x, y0, y1, 
                            color=level_colors[i], 
                            lw=0.2, edgecolor='k',
                            alpha=level_alpha[i], zorder=1, 
                            label=level_labels[levels[i]])
    
    # Or just do thr drought emergency level 5
    else:
        y = ffmp_level_boundaries['level5']
        drought_color = 'maroon' # cm.get_cmap('Reds')(0.3)
        
        # Fill with hatch
        ax.fill_between(x, [0.0]*len(y), y,  
                        facecolor='none',
                        edgecolor=drought_color,
                        linewidth=1.0,
                        hatch='XX',
                        alpha=0.2, zorder=1,
                        label='Drought Emergency')
        
    # Observed storage
    if plot_observed:
        assert 'obs' in list(storages.keys()), 'Must include key "obs" in models to plot observed storage.'
        historic_storage = subset_timeseries(storages['obs'][0]["Total"], 
                                             start_date, end_date)

        historic_storage = historic_storage.rolling(smoothing_window, center=True).mean()
        ax.plot(historic_storage, 
                color= colordict['obs'], 
                ls = '--', 
                lw = 1.,
                label=model_label_dict['obs'], 
                zorder=10)
        
    line_colors = [colordict[m] for m in models]
    
    # Loop through models
    for m,c in zip(models,line_colors):
        
        if 'ensemble' in m:
            # Get realization numbers
            realization_numbers = list(storages[m].keys())
            for i, real in enumerate(realization_numbers):
                modeled_storage = subset_timeseries(storages[m][real][reservoir_list_nyc], start_date, end_date).sum(axis=1)
                modeled_storage *= 100/capacities['combined']
                if i == 0:
                    ensemble_modeled_storage = pd.DataFrame(modeled_storage, columns=[real], 
                                                            index=modeled_storage.index)
                else:
                    ensemble_modeled_storage = pd.concat((ensemble_modeled_storage, 
                                                          pd.DataFrame(modeled_storage, 
                                                                       columns=[real], 
                                                                       index=modeled_storage.index)), axis=1)                
                
            ensemble_modeled_storage = ensemble_modeled_storage.rolling(smoothing_window, center=True).mean()
            # Plot quantiles            
            if percentile_cmap:
                ax = plot_ensemble_percentile_cmap(ensemble_modeled_storage, 
                                                   m,
                                                   ax, 
                                                   q_lower_bound=q_lower_bound, q_upper_bound=q_upper_bound, 
                                                   alpha=ensemble_fill_alpha, zorder=2)

            else:
                ax.fill_between(ensemble_modeled_storage.index,
                    ensemble_modeled_storage.quantile(q_lower_bound, axis=1),
                    ensemble_modeled_storage.quantile(q_upper_bound, axis=1),
                    color=c, alpha=ensemble_fill_alpha, 
                    zorder=2, lw=1.6,
                    label = model_label_dict[m])

    ### clean up figure
    ax.set_xlim([start_date, end_date])
    ax.set_ylabel('Combined NYC Storage (%)', fontsize=fontsize)
    ax.set_ylim([0,100])
    
    if not is_subplot:
        ax = clean_xtick_labels([ax], start_date, end_date, fontsize=fontsize)[0]
        ax.set_xlabel('Year', fontsize=fontsize)
        ax.legend(frameon=False, fontsize=fontsize, loc='upper left', 
              bbox_to_anchor=(0.0, -0.2), ncols=3)
        
        plt.savefig(f'{fig_dir}ensemble_nyc_storage_{start_date.strftime("%Y")}_{end_date.strftime("%Y")}.png',
                    dpi=dpi, 
                    bbox_inches='tight')        
    else:
        return ax


####################################################################

def plot_ensemble_NYC_release_contributions(model,
                                     nyc_release_components,
                                     reservoir_releases,
                                     reservoir_downstream_gages,
                                     all_mrf,
                                     colordict= model_colors_historic_reconstruction,
                                     plot_observed=True,
                                     percentile_cmap=False,
                                     start_date=None, end_date=None, 
                                     fig_dpi=200, 
                                     fig_dir=fig_dir, 
                                     fontsize=10, 
                                     use_log=False,
                                     q_lower_bound = 0.05,
                                     q_upper_bound = 0.95,
                                     smoothing_window=1,
                                     ensemble_fill_alpha = 1,
                                     contribution_fill_alpha= 0.9,
                                     ax=None,
                                     units='MGD'):
    
    if 'ensemble' in model:
        use_contribution_model = model + '_mean'
        colordict[use_contribution_model] = colordict[model]
        model_label_dict[use_contribution_model] = model_label_dict[model] + 'Mean'
        
        # get ensemble mean for each dataset
        nyc_release_components[use_contribution_model] = ensemble_mean(nyc_release_components[model])
        reservoir_releases[use_contribution_model] = ensemble_mean(reservoir_releases[model])
        reservoir_downstream_gages[use_contribution_model] = ensemble_mean(reservoir_downstream_gages[model])
        all_mrf[use_contribution_model] = ensemble_mean(all_mrf[model])
    else:
        use_contribution_model = model
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(7,3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True
    release_total = subset_timeseries(reservoir_releases[use_contribution_model][reservoir_list_nyc], 
                                      start_date, end_date).sum(axis=1)

    # Handle when mrf requirements are greater than release 
    ## TODO: Fix this!

    total_flow_released = release_total.copy()
    for c in ['mrf_target_individual', 'mrf_montagueTrenton', 'flood_release', 'spill']:
        for r in reservoir_list_nyc:
            nyc_release_components[use_contribution_model][f'{c}_{r}'] = nyc_release_components[use_contribution_model][f'{c}_{r}'].clip(lower=0, 
                                                                                                                       upper=total_flow_released)
            total_flow_released -= nyc_release_components[use_contribution_model][f'{c}_{r}']
    deficit = release_total - total_flow_released
    
    if np.isnan(release_total).any():
        print('Warning: NaNs in release_total.')
    
    x = release_total.index
    downstream_gage_pywr = subset_timeseries(reservoir_downstream_gages[use_contribution_model]['NYCAgg'], 
                                             start_date, end_date)
    if np.isnan(downstream_gage_pywr).any():
        print('Warning: NaNs in downstream_gage_pywr.')
        print(f'downstream_gage_pywr: {downstream_gage_pywr}')
    downstream_uncontrolled_pywr = downstream_gage_pywr - release_total
    
    if 'ensemble' in model:
        realizations = list(reservoir_downstream_gages[model].keys())
        for i,real in enumerate(realizations):
            realization_downstream_gage_pywr = subset_timeseries(reservoir_downstream_gages[model][real][reservoir_list_nyc], 
                                              start_date, end_date).sum(axis=1)
            if i == 0:
                ensemble_downstream_gage_pywr = pd.DataFrame(realization_downstream_gage_pywr, columns=[real], index=release_total.index)
            else:
                df = pd.DataFrame(realization_downstream_gage_pywr, 
                                  columns=[real], 
                                  index=release_total.index)
                ensemble_downstream_gage_pywr = pd.concat((ensemble_downstream_gage_pywr, df), axis=1)
                
        ensemble_downstream_gage_pywr = ensemble_downstream_gage_pywr.rolling(smoothing_window, center=True).mean()

        # Fill between quantiles
        if percentile_cmap:
            ax = plot_ensemble_percentile_cmap(ensemble_downstream_gage_pywr, model, ax,
                                                  q_lower_bound=q_lower_bound, q_upper_bound=q_upper_bound,
                                                  alpha=ensemble_fill_alpha, zorder=2)
        else:
            ax.fill_between(ensemble_downstream_gage_pywr.index,
                        ensemble_downstream_gage_pywr.quantile(q_lower_bound, axis=1),
                        ensemble_downstream_gage_pywr.quantile(q_upper_bound, axis=1),
                        color=colordict[model], alpha=0.85, zorder=4, lw=0.0,
                        label=model_label_dict[model])
    
    if plot_observed:
        downstream_gage_obs = subset_timeseries(reservoir_downstream_gages['obs'][0]['NYCAgg'], 
                                                start_date, end_date)
        downstream_gage_obs = downstream_gage_obs.rolling(smoothing_window, center=True).mean()
    
        if len(downstream_gage_obs) > 0:
            ax.plot(downstream_gage_obs, color='k', ls='--', lw=1, 
                    label='Observed Flow', zorder=10)
            
    if not is_subplot:
        ax.legend(frameon=False, loc='upper center', 
                bbox_to_anchor=(0.94, -1.25), ncols=1, fontsize=fontsize)

    ax.set_xlim([x[0], x[-1]])
    ax_twin = ax.twinx()
    ax_twin.set_ylim([0,100])

    if use_log:
        ax.semilogy()
        ymax = downstream_gage_pywr.max()
        ymin = downstream_gage_pywr.min()
        if plot_observed:
            ymax = max(ymax, downstream_gage_obs.max())
            ymin = max(ymin, downstream_gage_obs.min())
        for i in range(10):
            if ymin < 10 **i:
                ymin = 10 **(i-1)
                break
        for i in range(10):
            if ymax < 10 **i:
                ymax = 10 **(i)
                break
        ax.set_ylim([ymin, ymax])
    else:
        ax.set_ylim([0, ax.get_ylim()[1]]) 

    release_components_full = subset_timeseries(nyc_release_components[use_contribution_model], start_date, end_date)
    
    release_types = ['mrf_target_individual', 'flood_release', 'spill']
    release_components = pd.DataFrame({release_type: release_components_full[[c for c in release_components_full.columns
                                                                              if release_type in c]].sum(axis=1)
                                       for release_type in release_types})
    release_components['uncontrolled'] = downstream_uncontrolled_pywr

    release_components['mrf_montagueTrenton'] = release_components_full[[c for c in release_components_full.columns
                                                                                if 'mrf_montagueTrenton' in c]].sum(axis=1)
    
    release_components['mrf_montague'] = all_mrf[use_contribution_model][[c for c in all_mrf[use_contribution_model].columns
                                                               if 'release_needed_mrf_montague_' in c]].sum(axis=1)
    release_components['mrf_trenton'] = release_components['mrf_montagueTrenton'] - release_components['mrf_montague']
    release_components['mrf_trenton'] = release_components['mrf_trenton'].clip(lower=0.0)

    release_components = release_components.divide(downstream_gage_pywr, axis=0) * 100
    release_components.fillna(0, inplace=True)
    
    release_components = release_components.rolling(smoothing_window, center=True, axis=0).mean()
    release_components.fillna(0, inplace=True)
    for c in release_types:
        if np.isnan(release_components[c]).any():
            print(f'Warning: NaNs in release components for {c} after divide and rolling.')
            print(f'downstream_gage_pywr: {downstream_gage_pywr}')
            
    y1 = np.zeros(len(release_components['uncontrolled'].values))
    y2 = y1 + release_components['uncontrolled'].values
    y3 = y2 + release_components['mrf_montague'].values
    y33 = y3 + release_components['mrf_trenton'].values
    y4 = y33 + release_components['mrf_target_individual'].values
    y5 = y4 + release_components['flood_release'].values
    y6 = y5 + release_components['spill'].values
    
    y1 = y1 * 100/y6
    y2 = y2 * 100/y6
    y3 = y3 * 100/y6
    y33 = y33 * 100/y6
    y4 = y4 * 100/y6
    y5 = y5 * 100/y6
    y6 = y6 * 100/y6
    
    for i,y in enumerate([y2, y3, y4, y5, y6]):        
        if sum(np.isnan(y)) > 0:
            print(f'Warning: NaNs in release components for y{i+1}')
    print(f'Max NYC contribution perc: {y6.max()}')
    ax_twin.fill_between(x, y5, y6, label='NYC Spill', color=flow_colors['NYC Spill'], alpha=contribution_fill_alpha, lw=0, zorder=1)
    ax_twin.fill_between(x, y4, y5, label='NYC Flood Release', color=flow_colors["NYC Flood Release"], alpha=contribution_fill_alpha, lw=0, zorder=1)
    ax_twin.fill_between(x, y33, y4, label='NYC Minimum Daily Release', color=flow_colors["NYC Minimum Daily Release"], alpha=contribution_fill_alpha, lw=0, zorder=1)
    ax_twin.fill_between(x, y3, y33, label='NYC Trenton Equiv. Flow Release', color=flow_colors['NYC Trenton Equiv. Flow Release'], alpha=contribution_fill_alpha, lw=0, zorder=1)
    ax_twin.fill_between(x, y2, y3, label='NYC Directed Montague Release', color=flow_colors['NYC Directed Montague Release'], alpha=contribution_fill_alpha, lw=0, zorder=1)
    ax_twin.fill_between(x, y1, y2, label='Uncontrolled Flow', color=flow_colors['Uncontrolled Flow'], alpha=contribution_fill_alpha, lw=0, zorder=1)

    ax.set_ylabel(f'NYC Release ({units})', fontsize=fontsize)
    ax_twin.set_ylabel('Flow Contribution (%)', fontsize=fontsize)

    ax_twin.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax_twin.set_yticks(ax_twin.get_yticks(), 
                       ax_twin.get_yticklabels(), fontsize=fontsize)
    return (ax, ax_twin)


def plot_ensemble_node_flow_contributions(model,
                                          node,
                                          major_flows,
                                     nyc_release_components,
                                     lower_basin_mrf_contributions,
                                     reservoir_releases,
                                     inflows, 
                                     consumptions,
                                     diversions, 
                                     all_mrf=None,
                                     colordict= model_colors_historic_reconstruction,
                                     plot_observed=True,
                                     plot_flow_target=False,
                                     percentile_cmap = False,
                                     ensemble_fill_alpha = 1,
                                     contribution_fill_alpha= 0.9,
                                     start_date=None, end_date=None, 
                                     fig_dpi=200, fig_dir=fig_dir, 
                                     fontsize=10, use_log=False,
                                     q_lower_bound = 0.05,
                                     q_upper_bound = 0.95,
                                     smoothing_window=1,
                                     ax=None,
                                     units='MGD'):
    if plot_flow_target:
        assert all_mrf is not None, 'Must provide all_mrf if plot_flow_target is True.'
    
    if 'ensemble' in model:
        use_contribution_model = model + '_mean'
        colordict[use_contribution_model] = colordict[model]
        model_label_dict[use_contribution_model] = model_label_dict[model] + 'Mean'
        
        # get ensemble mean for each dataset
        major_flows[use_contribution_model] = ensemble_mean(major_flows[model])
        nyc_release_components[use_contribution_model] = ensemble_mean(nyc_release_components[model])
        lower_basin_mrf_contributions[use_contribution_model] = ensemble_mean(lower_basin_mrf_contributions[model])
        reservoir_releases[use_contribution_model] = ensemble_mean(reservoir_releases[model])
        inflows[use_contribution_model] = ensemble_mean(inflows[model])
        consumptions[use_contribution_model] = ensemble_mean(consumptions[model])
        diversions[use_contribution_model] = ensemble_mean(diversions[model])
    
        all_mrf[use_contribution_model] = ensemble_mean(all_mrf[model])
    
    else:
        use_contribution_model = model
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(7,3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True
        
    # Get total sim and obs flow    
    if 'ensemble' in model:
        realizations = list(major_flows[model].keys())
        # Make empty dataframe to store ensemble results
        ensemble_sim_node_flow = pd.DataFrame(columns=realizations, index=major_flows[model][realizations[0]][node].index).astype(float)
        
        
        for real in realizations:
            total_sim_node_flow = subset_timeseries(major_flows[model][real][node], start_date, end_date)
            
            ### for Trenton, add NJ diversion to simulated flow. also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
            if node == 'delTrenton':
                nj_diversion = subset_timeseries(diversions[model][real]['delivery_nj'], start_date, end_date)
                total_sim_node_flow += nj_diversion

                ### get drbc contributions from lower basin reservoirs
                realization_lower_basin_mrf_contributions = subset_timeseries(lower_basin_mrf_contributions[model][real], 
                                                                  start_date, end_date)
                realization_lower_basin_mrf_contributions.columns = [c.split('_')[-1] for c in realization_lower_basin_mrf_contributions.columns]

                # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
                if node == 'delTrenton':
                    for c in ['blueMarsh']:
                        lag = downstream_node_lags[c]
                        downstream_node = immediate_downstream_nodes_dict[c]
                        while downstream_node != 'output_del':
                            lag += downstream_node_lags[downstream_node]
                            downstream_node = immediate_downstream_nodes_dict[downstream_node]
                        if lag > 0:
                            idx = realization_lower_basin_mrf_contributions.index
                            realization_lower_basin_mrf_contributions.loc[idx[lag:], c] = realization_lower_basin_mrf_contributions.loc[:, c].shift(lag)
                total_sim_node_flow += realization_lower_basin_mrf_contributions['blueMarsh']
                total_sim_node_flow = pd.to_numeric(total_sim_node_flow, errors='coerce')
            
            # store
            ensemble_sim_node_flow[real] = total_sim_node_flow.copy()         
                
        ensemble_sim_node_flow = ensemble_sim_node_flow.copy()
        ensemble_sim_node_flow = ensemble_sim_node_flow.rolling(smoothing_window, center=True).mean()
        
        if percentile_cmap:
            ax = plot_ensemble_percentile_cmap(ensemble_sim_node_flow, model, ax, 
                                                q_lower_bound=q_lower_bound, q_upper_bound=q_upper_bound,
                                                alpha=ensemble_fill_alpha, zorder=10)
            
            ensemble_median = ensemble_sim_node_flow.median(axis=1)
            min_ensemble_median_val = ensemble_median.min()
            min_ensemble_median_day = ensemble_median.idxmin()
            print(f"STAT: Min ensemble median value: {min_ensemble_median_val} on {min_ensemble_median_day} for {node}")
            
            target = all_mrf[use_contribution_model].loc[start_date:end_date, f'mrf_target_{node}']
            ffmp_target_during_min = target.loc[min_ensemble_median_day]
            print(f"STAT: FFMP target during min ensemble median value: {ffmp_target_during_min} on {min_ensemble_median_day} for {node}")
        else:
            ax.fill_between(ensemble_sim_node_flow.index,
                            ensemble_sim_node_flow.quantile(q_lower_bound, axis=1),
                            ensemble_sim_node_flow.quantile(q_upper_bound, axis=1),
                            color=colordict[model], alpha=ensemble_fill_alpha, 
                            zorder=2, lw=1.6,
                            label = model_label_dict[model])
                
    # repeat for QPPQ aggregate version
    total_sim_node_flow = subset_timeseries(major_flows[use_contribution_model][node], start_date, end_date)
    if total_sim_node_flow.isna().any():
        print(f'WARNING: toal_sim_node_flow has NAs.')
    
    ### for Trenton, add NJ diversion to simulated flow. 
    # also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
    if node == 'delTrenton':
        nj_diversion = subset_timeseries(diversions[use_contribution_model]['delivery_nj'], start_date, end_date)
        total_sim_node_flow += nj_diversion

        ### get drbc contributions from lower basin reservoirs
        lower_basin_mrf_contributions = subset_timeseries(lower_basin_mrf_contributions[use_contribution_model], start_date, end_date)
        lower_basin_mrf_contributions.columns = [c.split('_')[-1] for c in lower_basin_mrf_contributions.columns]

        # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
        for c in ['blueMarsh']:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != 'output_del':
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = lower_basin_mrf_contributions.index
                lower_basin_mrf_contributions.loc[idx[lag:], c] = lower_basin_mrf_contributions.loc[:, c].shift(lag)
        total_sim_node_flow += lower_basin_mrf_contributions['blueMarsh']

    nyc_release_components= nyc_release_components.copy()
    
    for r in reservoir_list_nyc:
        total_release = reservoir_releases[use_contribution_model][r].copy()
        for c in ['mrf_target_individual', 'mrf_montagueTrenton', 'flood_release', 'spill']:
            
            # Getting error: "cannot reindex on an axis with duplicate labels"
            total_release = total_release.loc[~total_release.index.duplicated(keep='first')]
            nyc_res_release= nyc_release_components[use_contribution_model][f'{c}_{r}']
            nyc_res_release = nyc_res_release.loc[~nyc_res_release.index.duplicated(keep='first')]
            nyc_res_release = nyc_res_release.reindex(total_release.index, fill_value=0)
            
            mrf_shortfall = total_release - nyc_res_release 
            mrf_shortfall[mrf_shortfall >= 0] = 0
            total_release -= nyc_res_release
            total_release[total_release < 0] = 0
            
            nyc_release_components[use_contribution_model][f'{c}_{r}'] = nyc_release_components[use_contribution_model][f'{c}_{r}'] + mrf_shortfall
            

    # Plot observed flow
    if plot_observed:      
        total_obs_node_flow = subset_timeseries(major_flows['obs'][0][node], start_date, end_date)                
        if node == 'delTrenton':
            nj_diversion = subset_timeseries(diversions[use_contribution_model]['delivery_nj'], 
                                                    start_date, end_date)
            total_obs_node_flow += nj_diversion
        total_obs_node_flow = total_obs_node_flow.rolling(smoothing_window, center=True).mean()
  
        if len(total_obs_node_flow)>0:
            ax.plot(total_obs_node_flow, color='k', 
                    ls='--', lw=1, zorder = 10)

    # plot streamflow target
    if plot_flow_target:
        target = all_mrf[use_contribution_model].loc[start_date:end_date, f'mrf_target_{node}']
            
        target_label = "Min. Flow Target at Montague" if node == 'delMontague' else "Min. Flow Target at Trenton" 

        ax.plot(target, color='k', 
                ls=':', lw=1, zorder=5,
                label=target_label)

            
    ax_twin = ax.twinx()
    ax_twin.set_ylim([0,100])
    ax.set_xlim(start_date, end_date)
    if use_log:
        ax.semilogy()
        ymax = total_sim_node_flow.max()
        ymin = total_sim_node_flow.min()
        if plot_observed:
            ymax = max(ymax, total_obs_node_flow.max())
            ymin = max(ymin, total_obs_node_flow.min())
        for i in range(10):
            if ymin < 10 ** i:
                ymin = 10 ** (i - 1)
                break
        for i in range(10):
            if ymax < 10 ** i:
                ymax = 10 ** (i)
                break
        ax.set_ylim([ymin, ymax])
    else:
        ax.set_ylim([0, ax.get_ylim()[1]])

    ax.set_ylabel(f'Total Flow ({units})', fontsize=fontsize)
    ax_twin.set_ylabel('Flow Contribution (%)', fontsize=fontsize)



    # Get contributing flows
    contributing = upstream_nodes_dict[node]
    non_nyc_reservoirs = [i for i in contributing if (i in reservoir_list) and (i not in reservoir_list_nyc)]
    non_nyc_release_contributions = reservoir_releases[use_contribution_model][non_nyc_reservoirs]

    if node == 'delTrenton':
        ### subtract lower basin ffmp releases from their non-ffmp releases
        for r in drbc_lower_basin_reservoirs:
            if r != 'blueMarsh':
                non_nyc_release_contributions.loc[:, r] = np.maximum(non_nyc_release_contributions[r] -
                                                                lower_basin_mrf_contributions[r], 0)

        print(f'non_nyc_release_contributions.columns: {non_nyc_release_contributions.columns}')
        print(f'lower_basin_mrf_contributions.columns: {lower_basin_mrf_contributions.columns}')

    use_inflows = [i for i in contributing if (i in majorflow_list)]
    if node == 'delMontague':
        use_inflows.append('delMontague')
    inflow_contributions = inflows[use_contribution_model][use_inflows] - consumptions[use_contribution_model][use_inflows]
    mrf_target_individuals = nyc_release_components[use_contribution_model][[c for c in nyc_release_components[use_contribution_model].columns
                                                                 if 'mrf_target_individual' in c]]
    mrf_target_individuals.columns = [c.rsplit('_',1)[1] for c in mrf_target_individuals.columns]
    
    
    # Montague & Trenton releases needed to meet targets 
    mrf_needed_montague = all_mrf[use_contribution_model][[c for c in all_mrf[use_contribution_model].columns
                                                               if 'release_needed_mrf_montague_' in c]].sum(axis=1)

    # Combined Montague & Trenton releases        
    mrf_montagueTrenton_agg = nyc_release_components[use_contribution_model][[c for c in nyc_release_components[use_contribution_model].columns
                                                               if 'mrf_montagueTrenton' in c]].sum(axis=1)
    
    mrf_trenton_contribution = mrf_montagueTrenton_agg - mrf_needed_montague
    mrf_trenton_contribution = mrf_trenton_contribution.clip(lower=0.0)
    
    mrf_montague_contribution = mrf_montagueTrenton_agg - mrf_trenton_contribution
    mrf_montague_contribution = mrf_montague_contribution.clip(lower=0.0)
    
    # mrf_montagueTrenton_agg = all_mrf[use_contribution_model][['total_agg_mrf_montagueTrenton_step1',
    #                                                            'total_agg_mrf_montagueTrenton_step2']].sum(axis=1)

    
    flood_releases = nyc_release_components[use_contribution_model][[c for c in nyc_release_components[use_contribution_model].columns if
                                                         'flood_release' in c]]
    flood_releases.columns = [c.rsplit('_',1)[1] for c in flood_releases.columns]
    spills = nyc_release_components[use_contribution_model][[c for c in nyc_release_components[use_contribution_model].columns if 'spill' in c]]
    spills.columns = [c.rsplit('_',1)[1] for c in spills.columns]

    # Impose lag
    for c in upstream_nodes_dict[node][::-1]:
        if c in inflow_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = inflow_contributions.index
                inflow_contributions.loc[idx[lag:], c] = inflow_contributions.loc[:, c].shift(lag)
        elif c in non_nyc_release_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = non_nyc_release_contributions.index
                non_nyc_release_contributions.loc[idx[lag:], c] = non_nyc_release_contributions.loc[:, c].shift(lag)
                if node == 'delTrenton' and c in drbc_lower_basin_reservoirs:
                    idx = lower_basin_mrf_contributions.index
                    lower_basin_mrf_contributions.loc[idx[lag:], c] = lower_basin_mrf_contributions.loc[:, c].shift(lag)
                ### note: blue marsh lower_basin_mrf_contribution lagged above. 
                # It wont show up in upstream_nodes_dict here, so not double lagging.
        elif c in mrf_target_individuals.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = mrf_target_individuals.index
                mrf_target_individuals.loc[idx[lag:], c] = mrf_target_individuals.loc[:, c].shift(lag)
                flood_releases.loc[idx[lag:], c] = flood_releases.loc[:, c].shift(lag)
                spills.loc[idx[lag:], c] = spills.loc[:, c].shift(lag)
                # mrf_montagueTrentons.loc[idx[lag:], c] = mrf_montagueTrentons.loc[:, c].shift(lag)            


    inflow_contributions = subset_timeseries(inflow_contributions, start_date, end_date).sum(axis=1)
    non_nyc_release_contributions = subset_timeseries(non_nyc_release_contributions, start_date, end_date).sum(axis=1)
    if node == 'delTrenton':
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.sum(axis=1)
    mrf_trenton_contribution = subset_timeseries(mrf_trenton_contribution, start_date, end_date)
        
    idx = mrf_montague_contribution.index
    mrf_montague_contribution.loc[idx[4:]] = mrf_montague_contribution.loc[:].shift(4)
    mrf_montague_contribution = subset_timeseries(mrf_montague_contribution, start_date, end_date)
            
    mrf_target_individuals = subset_timeseries(mrf_target_individuals, start_date, end_date).sum(axis=1)
    
    
    flood_releases = subset_timeseries(flood_releases, start_date, end_date).sum(axis=1)
    spills = subset_timeseries(spills, start_date, end_date).sum(axis=1)

    inflow_contributions = inflow_contributions.divide(total_sim_node_flow) * 100
    non_nyc_release_contributions = non_nyc_release_contributions.divide(total_sim_node_flow) * 100
    if node == 'delTrenton':
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.divide(total_sim_node_flow) * 100
    mrf_montague_contribution = mrf_montague_contribution.divide(total_sim_node_flow) * 100
    mrf_trenton_contribution = mrf_trenton_contribution.divide(total_sim_node_flow) * 100
    
    mrf_target_individuals = mrf_target_individuals.divide(total_sim_node_flow) * 100
    
    flood_releases = flood_releases.divide(total_sim_node_flow) * 100
    spills = spills.divide(total_sim_node_flow) * 100

    # Apply rolling smooth across dfs
    inflow_contributions = inflow_contributions.rolling(smoothing_window, center=True).mean()
    non_nyc_release_contributions = non_nyc_release_contributions.rolling(smoothing_window, center=True).mean()
    
    if node == 'delTrenton':
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.rolling(smoothing_window, center=True).mean()
        
    mrf_target_individuals = mrf_target_individuals.rolling(smoothing_window, center=True).mean()
    mrf_montague_contribution = mrf_montague_contribution.rolling(smoothing_window, center=True).mean()
    mrf_trenton_contribution = mrf_trenton_contribution.rolling(smoothing_window, center=True).mean()
    
    flood_releases = flood_releases.rolling(smoothing_window, center=True).mean()
    spills = spills.rolling(smoothing_window, center=True).mean()
    

    x = total_sim_node_flow.index
    y1 = 0
    y2 = y1 + inflow_contributions.clip(lower=0.0)
    y3 = y2 + non_nyc_release_contributions.clip(lower=0.0)
    if node == 'delTrenton':
        y4 = y3 + lower_basin_mrf_contributions.clip(lower=0.0)
        y5 = y4 + mrf_montague_contribution.clip(lower=0.0)
        
    else:
        y5 = y3 + mrf_montague_contribution.clip(lower=0.0)
    y55 = y5 + mrf_trenton_contribution.clip(lower=0.0)
    y6 = y55 + mrf_target_individuals.clip(lower=0.0)
    y7 = y6 + flood_releases.clip(lower=0.0)
    y8 = y7 + spills.clip(lower=0.0)
    print(f'Total contribution mean:{y8.dropna().mean()}')
    print(f'Total contribution min:{y8.dropna().min()}')
    
    # scale y1 -> y8 so that they add to 100
    y8 = y8 * 100 / y8
    y7 = y7 * 100 / y8
    y6 = y6 * 100 / y8
    y55 = y55 * 100 / y8
    y5 = y5 * 100 / y8
    if node == 'delTrenton':
        y4 = y4 * 100 / y8
    y3 = y3 * 100 / y8
    y2 = y2 * 100 / y8
    y1 = y1 * 100 / y8
    

    ax_twin.fill_between(x, y7, y8, label='NYC Spill', color=flow_colors["NYC Spill"], alpha=contribution_fill_alpha, lw=0)
    ax_twin.fill_between(x, y6, y7, label='NYC Flood Release', color=flow_colors["NYC Flood Release"], alpha=contribution_fill_alpha, lw=0)
    ax_twin.fill_between(x, y5, y6, label='NYC Minimum Daily Release', color=flow_colors["NYC Minimum Daily Release"], alpha=contribution_fill_alpha, lw=0)
    if node == 'delTrenton':
        ax_twin.fill_between(x, y4, y5, label='NYC Directed Montague Release', color=flow_colors["NYC Directed Montague Release"], alpha=contribution_fill_alpha, lw=0)
        ax_twin.fill_between(x, y5, y55, label='NYC Trenton Equiv. Flow Release', color=flow_colors["NYC Trenton Equiv. Flow Release"], alpha=contribution_fill_alpha, lw=0)
        ax_twin.fill_between(x, y3, y4, label='Non-NYC Trenton Release', color=flow_colors["Non-NYC Trenton Equiv. Flow Release"], alpha=contribution_fill_alpha, lw=0)
    else:
        ax_twin.fill_between(x, y3, y5, label='NYC Directed Montague Release', color=flow_colors["NYC Directed Montague Release"], alpha=contribution_fill_alpha, lw=0)
        ax_twin.fill_between(x, y5, y55, label='NYC Trenton Equiv. Flow Release', color=flow_colors["NYC Trenton Equiv. Flow Release"], alpha=contribution_fill_alpha, lw=0)
    ax_twin.fill_between(x, y2, y3, label='Non-NYC Normal Release', color=flow_colors["Non-NYC Normal Release"], alpha=contribution_fill_alpha, lw=0)
    ax_twin.fill_between(x, y1, y2, label='Uncontrolled Flow', color=flow_colors["Uncontrolled Flow"], alpha=contribution_fill_alpha, lw=0)

    if not is_subplot:
        ax_twin.legend(frameon=False, 
                fontsize=fontsize, loc='upper center', 
                bbox_to_anchor=(0.37, -0.15), ncols=4)

    ax_twin.set_zorder(1)
    ax.set_zorder(10)
    ax.patch.set_visible(False)
    
    ax_twin.set_yticks(ax_twin.get_yticks(), 
                       ax_twin.get_yticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=fontsize)
    return (ax, ax_twin)


def plot_NYC_release_components_combined(storages, 
                                         ffmp_level_boundaries, 
                                         model,
                                         node,
                                         nyc_release_components,
                                         lower_basin_mrf_contributions, 
                                         reservoir_releases, 
                                         reservoir_downstream_gages,
                                         major_flows, 
                                         inflows, 
                                         diversions, 
                                         consumptions,
                                         all_mrf = None,
                                         colordict = base_model_colors, 
                                         start_date = None, end_date = None,
                                         use_log=False, 
                                         plot_flow_target=False,
                                         plot_observed=False, 
                                         fill_ffmp_levels=True,
                                         percentile_cmap = False,
                                         ensemble_fill_alpha = 1,
                                         contribution_fill_alpha= 0.9,
                                         q_lower_bound = 0.05,
                                         q_upper_bound = 0.95,
                                         smoothing_window=1,
                                         fig_dir=fig_dir, 
                                         fig_dpi=200,
                                         save_svg=False,
                                         units='MGD'):

    fig, axs = plt.subplots(3,1,figsize=(7,7), 
                            gridspec_kw={'hspace':0.1},
                            sharex=True)
    fontsize = 8
    labels = ['a)','b)','c)']

    ########################################################
    ### subplot a: Reservoir modeled storages
    ########################################################

    ax1 = axs[0]

    ### subplot a: Reservoir modeled storages
    ax1 = plot_ensemble_nyc_storage(storages,
                                ffmp_level_boundaries,
                                models = [model],
                                colordict = colordict,
                                start_date = start_date,
                                end_date = end_date,
                                fig_dir=fig_dir,
                                plot_observed = plot_observed,
                                ax=ax1,
                                fill_ffmp_levels=fill_ffmp_levels,
                                fontsize=fontsize,
                                percentile_cmap=percentile_cmap,
                                ensemble_fill_alpha=ensemble_fill_alpha,
                                smoothing_window=smoothing_window,
                                dpi=fig_dpi,
                                legend=False,
                                units=units)
    
    ax1.legend(frameon=False, loc='lower center', 
               bbox_to_anchor=(0.5,1.03), ncols=4, 
               fontsize=fontsize)
    ax1.annotate(labels[0], xy=(0.005, 0.975), 
                 xycoords='axes fraction', 
                 ha='left', va='top', 
                 weight='bold',
                 fontsize=fontsize)


    ########################################################
    # ### subfig b: first split up NYC releases into components
    ########################################################
    ax2 = axs[1]
    (ax1, ax2_twin) = plot_ensemble_NYC_release_contributions(model=model,
                                     nyc_release_components=nyc_release_components,
                                     reservoir_releases=reservoir_releases,
                                     reservoir_downstream_gages=reservoir_downstream_gages,
                                     all_mrf=all_mrf,
                                     colordict= colordict,
                                     plot_observed=plot_observed,
                                     percentile_cmap=percentile_cmap,
                                     start_date=start_date, end_date=end_date, 
                                     fig_dpi=fig_dpi, fig_dir=fig_dir, 
                                     fontsize=fontsize, use_log=use_log,
                                     q_lower_bound = q_lower_bound,
                                     q_upper_bound = q_upper_bound,
                                     ensemble_fill_alpha=ensemble_fill_alpha,
                                     contribution_fill_alpha=contribution_fill_alpha,
                                     smoothing_window=smoothing_window,
                                     ax=ax2,
                                     units=units)
    ax2.annotate(labels[1], xy=(0.005, 0.975), 
                 xycoords='axes fraction', 
                 ha='left', va='top', 
                 weight='bold',
                 fontsize=fontsize)


    ########################################################
    ### subfig c: split up montague/trenton flow into components
    ########################################################
    ax3 = axs[2]
    (ax3, ax3_twin) = plot_ensemble_node_flow_contributions(model, node, 
                                                            major_flows,
                                          nyc_release_components=nyc_release_components,
                                          lower_basin_mrf_contributions=lower_basin_mrf_contributions,
                                          reservoir_releases=reservoir_releases,
                                          inflows=inflows,
                                          consumptions=consumptions,
                                          diversions=diversions,
                                          colordict=colordict,
                                          plot_observed=plot_observed,
                                          plot_flow_target=plot_flow_target,
                                          all_mrf=all_mrf,
                                          percentile_cmap=percentile_cmap,
                                          ensemble_fill_alpha=ensemble_fill_alpha,
                                          contribution_fill_alpha=contribution_fill_alpha,
                                          start_date=start_date, end_date=end_date,
                                          fig_dpi=fig_dpi, fig_dir=fig_dir,
                                          fontsize=fontsize, use_log=use_log,
                                          q_lower_bound=q_lower_bound,
                                          q_upper_bound=q_upper_bound,
                                          smoothing_window=smoothing_window,
                                          ax=ax3,
                                          units=units)
    ax3.annotate(labels[2], xy=(0.005, 0.975), 
                 xycoords='axes fraction', ha='left', va='top', 
                 weight='bold',
                fontsize=fontsize)

    ### Make a single legend for subplots b and c
    combine_axs = [ax2, ax2_twin, ax3, ax3_twin]
    handles, labels = get_subplot_handles_and_labels(combine_axs)
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), 
               fontsize=fontsize, ncol=3, frameon=False)


    ### Clean up figure
    plt.xlim(start_date, end_date)
    start_year = str(pd.to_datetime(start_date).year)
    end_year = str(pd.to_datetime(end_date).year)
    filename = f'NYC_release_components_combined_{model}_{node}_' + \
                f'{start_year}_{end_year}' + \
                f'{"logscale" if use_log else ""}'
    if save_svg:
        plt.savefig(f'{fig_dir}/{filename}.svg',
                    bbox_inches='tight', dpi=fig_dpi)
    
        plt.savefig(f'{fig_dir}/{filename}.png',
                    bbox_inches='tight', dpi=fig_dpi)
    
    return





#########################################################################