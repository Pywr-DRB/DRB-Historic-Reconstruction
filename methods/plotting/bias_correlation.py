import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_bias_correlation_across_quantiles(bias, x_quantiles, y_quantiles,
                                           fig_dir=None, fname=None):
    
    n_rows = len(y_quantiles)
    n_cols = len(x_quantiles)
    fig, ax = plt.subplots(n_rows, n_cols, 
                           figsize=(1.5*n_cols, 1.5*n_rows),
                           dpi = 300)
    
    for j, x_quantile in enumerate(x_quantiles):
        for i, y_quantile in enumerate(y_quantiles):
            ax[i, j].scatter(bias[x_quantile], bias[y_quantile], alpha=0.5)
            
            # turn off tick mark labels
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            
            if i == n_rows-1:
                ax[i, j].set_xlabel(f"{x_quantile}")
            if j == 0:
                ax[i, j].set_ylabel(f"{y_quantile}")
                
            # set limits based on inner 90% value to improve visual
            max_val = max(bias[x_quantile].quantile(0.95), bias[y_quantile].quantile(0.95))
            min_val = min(bias[x_quantile].quantile(0.05), bias[y_quantile].quantile(0.05))
            ax[i, j].set_xlim(min_val, max_val)
            ax[i, j].set_ylim(min_val, max_val)
        
    plt.tight_layout()
    if fig_dir is not None and fname is not None:
        plt.savefig(f"{fig_dir}/{fname}", dpi=300)
    plt.show()
    return 
    
