"""
This script makes the final figures!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from methods.generator.inflow_scaling_regression import plot_inflow_scaling_regression


# Inflow scaling regression scatter plots
for model in ['nhmv10', 'nwmv21']:
    plot_inflow_scaling_regression(model, roll_window=3)