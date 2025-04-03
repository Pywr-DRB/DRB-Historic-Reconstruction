import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pywrdrb_node_data import obs_pub_site_matches
from config import FIG_DIR, FDC_QUANTILES

def plot_bias_heatmap(bias, 
                      quantiles, 
                      fig_dir='.', 
                      fig_name='bias_heatmap.png',
                      title_text="NHMv1.0 Streamflow Bias\nAcross Diagnostic Streamflow Sites and FDC Quantiles"):
    """
    Plots a heatmap where column widths correspond to the width of the associated quantile.
    
    Parameters:
    bias (pd.DataFrame): 2D array-like structure where rows correspond to sites and 
                         columns correspond to bias values at different quantiles.
    quantiles (list or np.array): List of quantiles used to define column widths.
    fig_dir (str): Directory to save the figure.
    fig_name (str): Name of the saved figure file.
    """

    # Compute column boundaries from quantiles
    col_edges = np.array([0] + list(quantiles))  # Edges for pcolormesh
    row_edges = np.arange(bias.shape[0] + 1)  # Uniform row spacing

    # Create the figure
    fig, ax = plt.subplots(figsize=(7, 12))

    # Use pcolormesh to plot the heatmap with non-uniform column widths
    cmap = sns.color_palette("coolwarm_r", as_cmap=True)
    mesh = ax.pcolormesh(col_edges, row_edges, bias, cmap=cmap, 
                          vmin=-100, vmax=100)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.5, label='Percentage Bias (%)')
    cbar.ax.tick_params(labelsize=12)

    # Set x-axis labels and ticks at column centers
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2
    ax.set_xticks(col_centers)
    ax.set_xticklabels([f"{q:.3f}" for q in quantiles], 
                       fontsize=8, rotation=90)

    # Set y-axis labels
    ax.set_yticks(np.arange(len(bias.index)) + 0.5)
    ax.set_yticklabels(bias.index, rotation=0, fontsize=8)

    # Labels and title
    ax.set_xlabel('FDC Quantiles', fontsize=14)
    ax.set_ylabel('USGS Gage ID', fontsize=14)
    ax.set_title(title_text, fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{fig_name}", dpi=300)
    plt.show()





def plot_mean_predicted_bias(bias_pred,
                             quantiles=FDC_QUANTILES,
                             fig_dir=FIG_DIR,
                             pywrdrb=True):

    if pywrdrb:
        sites = list(bias_pred.keys())
        bias_means = {}
        for site in sites:
            if obs_pub_site_matches[site] is not None:
                continue
            bias_means[site] = bias_pred[site].mean(axis=0)
        bias_means = pd.DataFrame(bias_means, index=quantiles).transpose()

    
    plt.figure(figsize=(12, 17))
    ax = sns.heatmap(
        bias_means,
        cmap="coolwarm_r",
        center=0,
        cbar_kws={'label': 'Percentage Bias (%)', 'shrink': 0.5},  # Adjust shrink if needed
        vmax=200,
        vmin=-200)
    
    # Set label font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Percentage Bias (%)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Set other label font sizes
    plt.xlabel('FDC Quantiles', fontsize=14)
    plt.ylabel('Pywr-DRB Node Name', fontsize=14)
    plt.title('NHMv1.0 Streamflow Bias\nAcross PUB-Reconstructed Pywr-DRB Node Sites and FDC Quantiles', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}nhmv10_percentage_bias_pywrdrb_pub_sites.png", dpi=300)
    plt.show()
    return