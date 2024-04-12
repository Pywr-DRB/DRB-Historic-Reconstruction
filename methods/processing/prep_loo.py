"""
Contains functions used to prepare data for leave-one-out experiment 
of the historic reconstruction method. 

Trevor Amestoy
"""

import numpy as np
import pandas as pd
import pynhd as nhd
import os
import json

from methods.utils.directories import DATA_DIR
from methods.utils.lists import known_managed_sites

def find_matching_sites(usgs_site_ids, 
                        modeled_site_ids, 
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


def get_leave_one_out_sites(Q, 
                            usgs_site_ids,
                            modeled_site_ids,
                            second_modeled_site_ids=None):
    """Returns the leave-one-out sites.
    """
    loo_sites = []
    for site in usgs_site_ids:
        if site in modeled_site_ids:
            if second_modeled_site_ids is not None:
                if site in second_modeled_site_ids:
                    
                    # check that it's not in the Q dataframe
                    if site in Q.columns:
                        obs_datetime = Q.loc[:, site].dropna().index
                        model_datetime = pd.date_range(start='1983-10-01', 
                                                       end='2020-12-01', 
                                                       freq='D')
                        
                        overlap = model_datetime.intersection(obs_datetime)
                        
                        MIN_YEARS_OVERLAP = 10
                        if len(overlap) > (365*MIN_YEARS_OVERLAP):
                            loo_sites.append(site)
            else:
                loo_sites.append(site)
    
    # check that it's not in known managed sites
    for site in loo_sites:
        if site in known_managed_sites:
            loo_sites.remove(site)
    return loo_sites




### THIS FUNCTION IS OUTDATED: See methods.spatial.upstream for a new version

# def get_upstream_gauges(stations,
#                         gauge_meta,
#                         simplify=True,
#                         filename = None):
#     """Returns a dict of {'station_id': [upstream station_ids]}.

#     Args:
#         stations (list): A list of stations to get upstream stations for.
#         gauge_meta (pd.DataFrame): A dataframe of gauge metadata with site numbers as index and a 'comid' column.
#         simplify (bool, optional): If True, only return the largest subcatchments (i.e., only independent catchments).
#     """
#     if filename is None:
#         filename = 'upstream_gauges' if simplify else 'upstream_gauges_all'
#     filepath = f'{DATA_DIR}/USGS/{filename}'
#     filepath = f'{filepath}_simplified.json' if simplify else f'{filepath}.json'
    
#     # Check if upstream_gauges.json exists
#     if os.path.exists(filepath):
#         rebuild = False
#         print('Loading upstream gauges from file...')
#         with open(filepath, 'r') as f:
#             catchment_subcatchments = json.load(f)
        
#         # Check if all sites are in the file
#         for site in stations:
#             if site not in catchment_subcatchments.keys():
#                 print(f'Gauge {site} not found in file. Rebuilding...')
#                 rebuild = True
#                 break                    
            
#         if not rebuild:
#             return catchment_subcatchments

      
#     print('Searching for upstream gauges for each site...')
#     ## use pynhd to identify upstream gauges for each model site
#     catchment_subcatchments = {}
#     nldi = nhd.NLDI()

#     for i, site_id_no in enumerate(stations):
#         print(f'Getting upbasin data for site {i} of {len(stations)}...')

#         comid = str(gauge_meta.loc[site_id_no, 'comid'])
        
#         # Get upstream catchments
#         try:
#             upstream_data = nldi.navigate_byid(fsource = 'comid', fid=comid,
#                                                 navigation='upstreamTributaries', 
#                                                 source='nwissite', distance=1000)
            
#             # Store and clean gauge IDs (just numbers as strings)
#             upstream_gauges = list(upstream_data['identifier'].values)
#             upstream_gauges = [c.split('-')[1] for c in upstream_gauges]
            
#             # Store if the gauge is in the NHM-Gauge matches
#             matching_upstream_gauges=[]
#             for g in upstream_gauges:
#                 if (g in stations) and (g != site_id_no):
#                     matching_upstream_gauges.append(g)
#             matching_upstream_gauges = list(set(matching_upstream_gauges))
            
#             catchment_subcatchments[site_id_no] = matching_upstream_gauges
#         except:
#             pass
    
#     if simplify:
        
#         # Remove gauges that are upstream of other gauges
#         for check_gauge in catchment_subcatchments.keys():
            
#             all_upstream_gauges = catchment_subcatchments[check_gauge]
#             major_catchments = []
        
#             for upstream_gauge in all_upstream_gauges:
#                 subtract_catchment = False
#                 for second_upstream_gauge in all_upstream_gauges:
#                     if second_upstream_gauge not in catchment_subcatchments.keys():
#                         continue
#                     elif upstream_gauge not in catchment_subcatchments[second_upstream_gauge]:
#                         subtract_catchment = True
#                     else:
#                         subtract_catchment = False
#                         break
#                 # Decide to keep or remove the catchment
#                 if subtract_catchment:
#                     major_catchments.append(upstream_gauge)
#             # Remove the catchments
#             catchment_subcatchments[check_gauge] = major_catchments
            
#     # Check that gauges are not contained in their own lists
#     for check_gauge in catchment_subcatchments.keys():
#         if check_gauge in catchment_subcatchments[check_gauge]:
#             catchment_subcatchments[check_gauge].remove(check_gauge)
    
#     # Exort
#     with open(filepath, 'w') as f:
#         json.dump(catchment_subcatchments, f)
        
#     return catchment_subcatchments



    