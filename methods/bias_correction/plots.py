import matplotlib.pyplot as plt
import numpy as np

def plot_observed_vs_predicted(obs, pred):    
    
    assert(obs.shape == pred.shape), 'Observed and predicted arrays must have the same shape'
    if type(obs) != np.ndarray:
        obs = np.array(obs).flatten()
    if type(pred) != np.ndarray:
        pred = np.array(pred).flatten()
    
    lim_min = np.min([np.min(obs), np.min(pred)])
    lim_max = np.max([np.max(obs), np.max(pred)])
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(obs, pred, alpha=0.2, color='k')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            color='r', lw=2, ls='--', zorder=0)
    # ax.text(-1.5, 1.5, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}', fontsize=14)
    ax.set_xlabel('Observed', fontsize=14)
    ax.set_ylabel('Predicted', fontsize=14)
    return plt