"""
Trevor Amestoy

Contains functions for various visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_contributing_gages(IDW_model):
    """
    Parameters:
    ----------
    IDW_model : object
        The QPPQ_Generator object.
    """
    contributing_gages = IDW_model.xTr[IDW_model.KNN_indices, 0:2]
    PUB_gage = IDW_model.xPr[:, 0:2]

    plt.scatter(contributing_gages[:,0], contributing_gages[:,1], label = 'Gauged', color = 'blue')
    plt.scatter(PUB_gage[:,0], PUB_gage[:,1], label = 'Ungauged', color = 'red')
    plt.legend()
    plt.show()
    return

################################################################################
