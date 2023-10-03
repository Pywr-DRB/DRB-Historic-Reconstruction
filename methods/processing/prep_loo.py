"""
Contains functions used to prepare data for leave-one-out experiment 
of the historic reconstruction method. 

Trevor Amestoy
"""

import numpy as np
import pandas as pd
import pynhd as nhd

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



def get_upstream_gauges(stations,
                        gauge_meta,
                        simplify=True):
    """Returns a dict of {'station_id': [upstream station_ids]}.

    Args:
        stations (list): A list of stations to get upstream stations for.
        gauge_meta (pd.DataFrame): A dataframe of gauge metadata with site numbers as index and a 'comid' column.
        simplify (bool, optional): If True, only return the largest subcatchments (i.e., only independent catchments).
    """
    ## use pynhd to identify upstream gauges for each model site
    catchment_subcatchments = {}
    nldi = nhd.NLDI()

    for i, site_id_no in enumerate(stations):
        # print(f'Getting upbasin data for site {i}')

        comid = str(gauge_meta.loc[site_id_no, 'comid'])
        
        # Get upstream catchments
        upstream_data = nldi.navigate_byid(fsource = 'comid', fid=comid,
                                            navigation='upstreamTributaries', 
                                            source='nwissite', distance=1000)
        
        # Store and clean gauge IDs (just numbers as strings)
        upstream_gauges = list(upstream_data['identifier'].values)
        upstream_gauges = [c.split('-')[1] for c in upstream_gauges]
        
        # Store if the gauge is in the NHM-Gauge matches
        matching_upstream_gauges=[]
        for g in upstream_gauges:
            if (g in stations) and (g != site_id_no):
                matching_upstream_gauges.append(g)
        matching_upstream_gauges = list(set(matching_upstream_gauges))
        
        catchment_subcatchments[site_id_no] = matching_upstream_gauges
    
    if simplify:
        
        # Remove gauges that are upstream of other gauges
        for check_gauge in catchment_subcatchments.keys():
            
            all_upstream_gauges = catchment_subcatchments[check_gauge]
            major_catchments = []
        
            for upstream_gauge in all_upstream_gauges:
                subtract_catchment = False
                for second_upstream_gauge in all_upstream_gauges:
                    if upstream_gauge not in catchment_subcatchments[second_upstream_gauge]:
                        subtract_catchment = True
                    else:
                        subtract_catchment = False
                        break
                # Decide to keep or remove the catchment
                if subtract_catchment:
                    major_catchments.append(upstream_gauge)
            # Remove the catchments
            catchment_subcatchments[check_gauge] = major_catchments
                            
    return catchment_subcatchments


def get_basin_catchment_area(feature_id, feature_source='comid'):
    """Returns the catchment area of a comid.
    
    Args:
        feature_id (str): A comid or USGS number as a string.
        feature_source (str): 'comid' or 'usgs'. Defaults to 'comid'.
    Returns:
        float: The catchment area of the basin
    """
    cartesian_crs = 3857
    nldi = nhd.NLDI()
    basin_data = nldi.get_basins(fsource=feature_source, feature_ids=feature_id)
    area = basin_data.to_crs(cartesian_crs).geometry.area.values/10**6
    return area
    