import sys
import numpy as np
import pandas as pd

import scipy.stats as scs
from scipy.stats import genextreme, pearson3
from scipy.optimize import curve_fit
from scipy.stats import genextreme as gev
from scipy.stats import pearson3
from sklearn.preprocessing import StandardScaler
import spei as si

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.colors import Normalize

from methods.utils.directories import fig_dir, FIG_DIR
from methods.utils.lists import drbc_droughts
from methods.plotting.styles import model_colors, model_labels



axis_label_fsize = 12
legend_fsize = 6


def subplot_drbc_drought_events(ax, start_date, end_date, 
                                event_colors = None,
                                drbc_droughts=drbc_droughts,
                                legend = False):
    # Define colors for different event types
    if event_colors is None:
        event_colors = {'Watch': 'gold', 
                        'Warning': 'peru', 
                        'Emergency': 'saddlebrown'}

    ## DRBC-classified drought events
    # Iterate through the events and create colored bars
    for index, row in drbc_droughts.iterrows():
        color = event_colors[row['event_type']]
        start_date = pd.to_datetime(row['start_date'])
        end_date = pd.to_datetime(row['end_date'])
        ax.axvspan(start_date, end_date, facecolor=color, edgecolor='none', alpha=1, lw=0.0)

    # Adding legend for the event types
    if legend:
        handles = [mpatches.Patch(color=color, label=event_type) for event_type, color in event_colors.items()]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1.1), 
                title='Event Type', title_fontsize=10, fontsize=8, alignment='left')

    ax.set_yticks([])
    return ax


def get_drought_days(ssi):
    # Classify droughts
    in_critical_drought = False
    droughts = np.zeros_like(ssi.values)
    drought_days = []
    for ind in range(len(droughts)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
            
        else:
            if in_critical_drought:
                droughts[drought_days] =1
            in_critical_drought = False
            drought_days = [] 
    
    # Handle edge case
    if in_critical_drought:
        droughts[drought_days] =1
    return droughts



def get_ensemble_drought_days(ensemble_ssi, node, start_date, end_date):
    
    realizations = list(ensemble_ssi.keys())
    n_realizations = len(realizations)
    
    drought_days = np.zeros((n_realizations, len(ensemble_ssi[realizations[0]].loc[start_date:end_date, node])))
    for i, realization in enumerate(realizations):
        ssi = ensemble_ssi[realization].loc[start_date:end_date, node]
        drought_days[i, :] = get_drought_days(ssi)
    return drought_days
        

def get_fraction_of_ensemble_in_drought(ensemble_ssi, node, start_date, end_date):
    realizations = list(ensemble_ssi.keys())
    n_realizations = len(realizations)
    
    drought_days = get_ensemble_drought_days(ensemble_ssi, node, start_date, end_date)
    print(drought_days.shape)
    fraction_in_drought = np.sum(drought_days, axis=0)/n_realizations
    return fraction_in_drought

def modify_cmap_zero_to_white(original_cmap_name):
    # Get the original colormap
    original_cmap = plt.cm.get_cmap(original_cmap_name)

    # Copy the colormap's colors and modify the first color (0 value) to white
    modified_colors = original_cmap(np.arange(original_cmap.N))
    modified_colors[0, :] = [1, 1, 1, 1]  # RGBA for white

    # Create a new colormap from the modified colors
    modified_cmap = mcolors.ListedColormap(modified_colors)

    return modified_cmap

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

def plot_ensemble_percentile_cmap(ensemble_df, model, ax, q_upper_bound, q_lower_bound, alpha=1, zorder=2):
    mirror_cmap = create_mirrored_cmap('Oranges') if 'nhm' in model else create_mirrored_cmap('Blues')
    norm = Normalize(vmin=-0.1, vmax=1.1)
    # norm = Normalize(vmin=0, vmax=1)
    percentiles = np.linspace(q_lower_bound, 0.5, 50) #[::-1]
    delta_percentile = percentiles[1] - percentiles[0]
    overlap = delta_percentile / 3
    for i in range(len(percentiles)-1):
        lower = ensemble_df.quantile(percentiles[i], axis=1, interpolation='linear')
        upper = ensemble_df.quantile(1-percentiles[i], axis=1, interpolation='linear')

        lower = lower.astype(float)
        upper = upper.astype(float)
        
        if lower.isna().sum() > 0:
            print("WARNING: NaN values in lower during plot_ensemble_percentile_cmap.\nlower=")
            print(lower)
            lower = lower.dropna()
        if upper.isna().sum() > 0:
            print("WARNING: NaN values in upper during plot_ensemble_percentile_cmap.\nupper=")
            print(upper)
            upper = upper.dropna()
        
        if not np.isclose(lower.values, upper.values).all():            
            ax.fill_between(ensemble_df.index,
                            lower,
                            upper,
                            color= mirror_cmap(norm(percentiles[i])),
                            interpolate=False, 
                            edgecolor = 'none',
                            alpha=alpha, zorder=zorder, lw=0.0)
        else:
            print(f'Skipping {percentiles[i]}')
    return ax

def add_cbar_to_ax(cbar_ax, 
                   model, 
                   ticks, ticklabels,
                   mirrored_cmap = True,
                   cmap=None, 
                   fontsize=10,
                   orientation='vertical',
                   fname="ssi_droughts.png"):
        """
        Uses the 
        """
        norm = Normalize(vmin=-0.15, vmax=1.15)
        
        if mirrored_cmap:
            ensemble_cmap = create_mirrored_cmap('Oranges') if 'nhm' in model else create_mirrored_cmap('Blues')
        else:
            ensemble_cmap = cmap if cmap is not None else plt.cm.get_cmap('Oranges')
        ensemble_cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=ensemble_cmap, norm=norm),
                            cax=cbar_ax, 
                            orientation=orientation)
        ensemble_cbar.set_label('Ensemble Percentile', labelpad=5)
        # Put label on top of cbar
        ensemble_cbar.set_ticks(ticks)  # Set ticks if you want specific ones
        ensemble_cbar.set_ticklabels(ticklabels)  
        ensemble_cbar.ax.tick_params(labelsize=fontsize)
        ensemble_cbar.ax.xaxis.set_ticks_position('bottom')
        ensemble_cbar.ax.xaxis.set_label_position('top')
        ensemble_cbar.ax.yaxis.set_label_position('left')
        ensemble_cbar.set_label('')

        # Add a label to the top of the colorbar
        ensemble_cbar.ax.text(2.5, 0.5, f'{model_labels[model]} Distribution', 
                              transform=ensemble_cbar.ax.transAxes, 
                              ha='left', va='center', fontsize=fontsize)
        ensemble_cbar.ax.xaxis.label.set_size(fontsize)
        ensemble_cbar.ax.set_ylim([0, 1.0])
        ensemble_cbar.ax.xaxis.set_visible(False)
        for s in ['top', 'bottom', 'left', 'right']:
            cbar_ax.spines[s].set_visible(False)
        ensemble_cbar.outline.set_visible(False)
        return cbar_ax
    
def drought_metric_scatter_plot(drought_metrics):
    fig, ax = plt.subplots(figsize = (7,6))
    p = ax.scatter(drought_metrics['severity'], -drought_metrics['magnitude'],
            c= drought_metrics['duration'], cmap = 'viridis_r', s=100)

    plt.colorbar(p).set_label(label = 'Drought Duration (days)',size=15)
    plt.xlabel(r'Severity ($Minimum SSI_{6}$)', fontsize = 15)
    plt.ylabel(r'Magnitude (Acc. Deficit)', fontsize = 15)
    plt.title(f'Historic Droughts', fontsize = 16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    return



def plot_ssi(ssi, bound = 3.0, figsize= (8, 4), ax = None, gradient = False):
    """Plot the standardized index values as a time series. 
    Modified from https://github.com/martinvonk/SPEI/blob/main/src/spei/plot.py

    Parameters
    ----------
    si : pandas.Series
        Series of the standardized index
    bound : int, optional
        Maximum and minimum ylim of plot
    figsize : tuple, optional
        Figure size, by default (8, 4)
    ax : matplotlib.Axes, optional
        Axes handle, by default None which create a new axes

    Returns
    -------
    matplotlib.Axes
        Axes handle
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(ssi, color="k", lw = 1, label="SSI")
    ax.axhline(0, linestyle="--", color="k")

    nmin = -bound
    nmax = bound
    # Classify droughts
    in_drought = False
    in_critical_drought = False

    droughts = np.zeros_like(ssi.values)
    drought_days = []
    drought_counter = 0
    for ind in range(len(droughts)):
        if ssi.values[ind] < 0:
            in_drought = True
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
                droughts[drought_days] = 1
        else:
            in_drought = False
            in_critical_drought = False
            drought_days = [] 
        
    if gradient:
        droughts = ssi.to_numpy(dtype=float, copy=True)
        droughts[droughts > 0] = 0
        nodroughts = ssi.to_numpy(dtype=float, copy=True)
        nodroughts[nodroughts < 0] = 0
        x, y = np.meshgrid(ssi.index, np.linspace(nmin, nmax, 100))
        ax.contourf(
            x, y, y, cmap=plt.get_cmap("seismic_r"), levels=np.linspace(nmin, nmax, 100)
        )
        ax.fill_between(x=ssi.index, y1=droughts, y2=nmin, color="w")
        ax.fill_between(x=ssi.index, y1=nodroughts, y2=nmax, color="w")
    else:
        ax.fill_between(x=ssi.index, y1=nmax, y2=nmin, where=(droughts>0), color='red', alpha=0.5, interpolate=False, label = 'Drought Period')
    ax.set_ylim(nmin, nmax)
    return ax


def plot_ssi_band(ssi, ymax, ymin, ax = None):
    """Plot the standardized index values as a time series. 
    Modified from https://github.com/martinvonk/SPEI/blob/main/src/spei/plot.py

    Parameters
    ----------
    si : pandas.Series
        Series of the standardized index
    bound : int, optional
        Maximum and minimum ylim of plot
    figsize : tuple, optional
        Figure size, by default (8, 4)
    ax : matplotlib.Axes, optional
        Axes handle, by default None which create a new axes

    Returns
    -------
    matplotlib.Axes
        Axes handle
    """

    # Classify droughts
    in_critical_drought = False
    droughts = np.zeros_like(ssi.values)
    drought_days = []
    for ind in range(len(droughts)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
            
        else:
            if in_critical_drought:
                droughts[drought_days] =1
            in_critical_drought = False
            drought_days = [] 
    
    # Account for edge case where drought ends at end of time series
    if in_critical_drought:
        droughts[drought_days] =1    
    
    ax.fill_between(x=ssi.index, y1=ymax, y2=ymin, 
                    where=(droughts>0), color='red', alpha=0.5, 
                    interpolate=False)
    return ax




def plot_drbc_droughts_and_ssi(ssi, models, node, 
                               start_date, end_date,
                               percentiles_cmap = True, 
                               q_lower_bound = 0.05, 
                               q_upper_bound = 0.95,
                               plot_observed=False,
                               fname='ssi_droughts.png',
                               ssi_window=12):
    
    ### SSI-based droughts
    for m in models:
        assert(m in ssi.keys()), f"Model {m} not in ssi results dict"
    # Define colors for different event types
    event_colors = {'Watch': 'lightgrey', 
                    'Warning': 'grey', 
                    'Emergency': 'black'}

    subplot_labels = ['a)','b)','c)']
    
    
    xs = ssi['obs'].loc[start_date:end_date, node].index
    
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 4), dpi=400,
                                sharex=True, 
                                gridspec_kw={'height_ratios': [1, 1, 5]})

    ### SUBPLOT 1:  DRBC drought events
    subplot_drbc_drought_events(ax=ax1, 
                                start_date=start_date, end_date=end_date,
                                event_colors=event_colors)

    ### SUBPLOT 2: heatmap of fraction of ensemble in drought for every day
    ensemble_models = [m for m in models if 'ensemble' in m]
    grid = np.zeros((len(ensemble_models), len(xs)))
    
    for i, model in enumerate(ensemble_models):
        grid[i, :] = get_fraction_of_ensemble_in_drought(ssi[model], 
                                                         node, 
                                                         start_date, end_date)

    # Find ax2 location lower bound
    ax2_lower_bound = ax2.get_position().bounds[1]
    ax2_height = ax2.get_position().bounds[3]
    
    # Make new ax off to right side to place colorbar in
    cbar_ax = fig.add_axes([0.95, ax2_lower_bound + (ax2_height)/4, 
                            0.15, ax2_height/2])
    use_cmap = modify_cmap_zero_to_white('Reds')
    
    
    ax2b = ax2.twiny()
    ax2b.xaxis.set_ticks_position('bottom')

    clabel = '% of Ensemble in $SSI_{12}$ Drought' if ssi_window == 12 else '% of Ensemble in Drought'

    sns.heatmap(grid*100, ax=ax2b, cmap=use_cmap, 
                vmin=0, vmax=100, 
                cbar=True, cbar_ax=cbar_ax, 
                cbar_kws={'label': clabel,
                          'orientation': 'horizontal'})
    ax2b.set_xlim(0, len(grid[0, :]))
    cbar = ax2b.collections[0].colorbar
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    cbar.ax.tick_params(labelsize=legend_fsize)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.label.set_size(legend_fsize)
    cbar.outline.set_visible(True)
    

    ax2b.set_xticks([])
    ax2b.set_xticklabels([])
    ax2b.set_yticks([])
    ax2b.set_yticklabels([])
    ax2b.patch.set_visible(False)
            
    # Add box around subplot
    for s in ['top', 'bottom', 'left', 'right']:
        ax2b.spines[s].set_visible(True)    
        cbar.ax.spines[s].set_visible(True)
    
    ## Subplot 2: SSI values
    
    for m in models:
        if 'ensemble' in m:                 
            realizations = list(ssi[m].keys())
            for i, real_i in enumerate(realizations):
                ssi_instance = ssi[m][real_i].loc[start_date:end_date, node]
                
                if i == 0:
                    ensemble_data = pd.DataFrame(ssi_instance, columns=[i], 
                                                index=ssi_instance.index)
                else:
                    ensemble_data[i] = ssi_instance
                
                    
            # Full period
            if percentiles_cmap:
                plot_ensemble_percentile_cmap(ensemble_data, m,
                                            ax3, q_lower_bound, q_upper_bound, 
                                            alpha=1, zorder=2)
            else:        
                ax3.fill_between(x=ensemble_data.index,
                        y1=ensemble_data.quantile(q_lower_bound, axis=1),
                        y2=ensemble_data.quantile(q_upper_bound, axis=1),
                        color=model_colors[m], 
                        zorder=3,
                        alpha=0.8, 
                        interpolate=False, 
                        label = model_labels[m])

        else:
            ssi_instance = ssi[m].loc[start_date:end_date, node]
            ax3.plot(ssi_instance,
                    c=model_colors[m], 
                    label = model_labels[m],
                    lw=1, zorder = 6)

    if plot_observed:
        
        model_ls = '--'
        ssi_instance = ssi['obs'].loc[start_date:end_date, node]
        ax3.plot(ssi_instance,
                c=model_colors['obs'], 
                label = f'{model_labels["obs"]}',
                ls=model_ls,
                lw=1, zorder = 10)

    ax1.set_xlim(xs[0], xs[-1])
    ax3.set_xlim(xs[0], xs[-1])
    ax1.set_xticklabels([])
    # ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax3.set_ylim([-3.5,3.5])
    
    ax3.grid(axis='y', zorder=0, color='k', alpha=0.3, ls=':')
    ax3.xaxis.set_minor_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_locator(mdates.YearLocator(10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Subplot labels
    subplot_label_fsize = 10
    ax1.annotate(subplot_labels[0], xy=(0.005, 0.975), 
                 xycoords='axes fraction', 
                 ha='left', va='top', weight='bold',
                 fontsize=subplot_label_fsize)
    ax2b.annotate(subplot_labels[1], xy=(0.005, 0.975), 
                 xycoords='axes fraction', 
                 ha='left', va='top', weight='bold',
                 fontsize=subplot_label_fsize)
    ax3.annotate(subplot_labels[2], xy=(0.005, 0.975), 
                 xycoords='axes fraction', 
                 ha='left', va='top', weight='bold',
                 fontsize=subplot_label_fsize)
    
    ylabel = '$SSI_{12}$' if ssi_window == 12 else 'SSI'
    ax3.set_ylabel(ylabel, fontsize=axis_label_fsize)
    ax3.set_xlabel('Date', fontsize=axis_label_fsize)
    
    # Make a cbar for the ensemble percentiles
    if percentiles_cmap:
        cbar_ax = fig.add_axes([0.95, ax3.get_position().bounds[1] + (ax3.get_position().bounds[3])/3, 
                                0.03, 0.1])
        cbar_ax = add_cbar_to_ax(cbar_ax,
                          models[0], 
                          ticks=[0, 0.5, 1], 
                          ticklabels=['5%', '50%', '95%'],
                          mirrored_cmap = True,
                          fontsize=legend_fsize,
                          orientation='vertical')
    
    ax3.legend(loc='upper left', bbox_to_anchor=(1.06, 0.75), fontsize=legend_fsize, frameon=False)
    
    # Make a legend for ax1 (DRBC drought events)
    handles = [mpatches.Patch(color=color, label=event_type) for event_type, color in event_colors.items()]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1.45), 
            ncols= 1, labelspacing=0.3, frameon = False,
               title='DRBC Drought Event Type', title_fontsize=legend_fsize, fontsize=legend_fsize, alignment='left')

    fig.align_ylabels()
    
    if plot_observed:
        fname = fname.replace('.png', '_withObs.png')
        
    plt.savefig(fname, dpi = 200, bbox_inches='tight')
    
    fname = fname.replace('png', 'svg')
    plt.savefig(fname, dpi = 200, bbox_inches='tight')

    plt.show()
    return 