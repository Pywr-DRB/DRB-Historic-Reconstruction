"""
Contains functions used to prepare data for leave-one-out experiment 
of the historic reconstruction method. 

Trevor Amestoy
"""

import numpy as np
import pandas as pd

def find_matching_sites(usgs_site_ids, modeled_site_ids, 
                        second_modeled_site_ids=None):
    """Finds matches between modeled and USGS sites.
    
    Args:
        usgs_site_ids (list): List of USGS site ids.
        modeled_site_ids (list): List of modeled site ids.
        second_modeled_site_ids (list, optional): List of modeled site ids. Defaults to None.
        
    Returns:
        list: List of modeled site ids that match the USGS site ids.
    """
    
    matching_sites = []
    for usgs_site_id in usgs_site_ids:
        if usgs_site_id in modeled_site_ids:
            matching_sites.append(usgs_site_id)
        elif second_modeled_site_ids is not None:
            if usgs_site_id in second_modeled_site_ids:
                matching_sites.append(usgs_site_id)
    return matching_sites


