import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gauge_rsr_curve(rsr, fig_dir):
    plt.plot(np.arange(len(rsr)), rsr.sort_values())
    plt.hlines(0.2, 0.0, len(rsr), color='k', ls='--', label='RSR=0.2')
    plt.xlabel("Gauge Count", fontsize=14)
    plt.ylabel("Reservoir Storage Ratio (RSR)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}usgs_gauge_reservoir_storage_ratios.png", dpi=300)
    plt.show()
    return