"""
Used for identifying upstream gauges for relative to 
USGS gauge locations or NHD comid numbers.

The module contains the following functions:
    * ensure_file_exists - Ensure that the file exists. If it does not, create it with initial JSON data.
    * update_upstream_gauge_file - Update the upstream gauges file without overwriting it.
    * check_station_in_upstream_gauges - Check if the fid is in the file.
    * get_upstream_gauges_from_file - Get the upstream gauges of a station.
    * get_upstream_gauges_single_site - Get the upstream gauges of a station.
    * get_upstream_gauges_for_id_list - Get the upstream gauges for a list of comid sites.
    * get_immediate_upstream_sites - Get the immediate upstream sites for a given site.
"""


import os
import json
import pynhd as nhd

from methods.utils.directories import DATA_DIR
from methods.utils.constants import GEO_CRS

comid_upstream_gauge_file = f'{DATA_DIR}/comid_upstream_gauges.json'
station_upstream_gauge_file = f'{DATA_DIR}/station_upstream_gauges.json'

####################################################################################################

def ensure_file_exists(file):
    """Ensure that the file exists. If it does not, create it with initial JSON data.

    Args:
    file (str): The file to check.
    """
    try:
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                json.dump({}, f)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    return True

def update_upstream_gauge_file(all_upstream_gauges,
                                file=comid_upstream_gauge_file,
                                overwrite=False):
    """Update the upstream gauges file without overwriting it.
    
    Args:
    all_upstream_gauges (dict): The upstream gauges for each comid.
    """
    ensure_file_exists(file)
    
    # Load
    with open(file, 'r') as f:
        archieved_upstream_gauges = json.load(f)
        
    # Modify 
    for id in all_upstream_gauges.keys():
        if overwrite:
            archieved_upstream_gauges[id] = all_upstream_gauges[id]
        elif not overwrite and id not in archieved_upstream_gauges.keys():
            archieved_upstream_gauges[id] = all_upstream_gauges[id]
    # Save  
    with open(file, 'w') as f:  
        json.dump(archieved_upstream_gauges, f)
    return      


def check_station_in_upstream_gauges(fid,
                                    file=comid_upstream_gauge_file):
    """Check if the fid is in the file.
    The <filename> is checked to see if comid is in the file. If it is, True is returned. If it is not, False is returned.
    
    Args:
    fid (str): The fid for the site
    
    Returns:
    bool: True if comid is in the file, False otherwise.
    """
    ensure_file_exists(file)
    
    # Check if the file exists
    with open(file, 'r') as f:
        upstream_gauges = json.load(f)
        if fid in upstream_gauges.keys():
            return True
        else:
            return False        
        

def get_upstream_gauges_from_file(fid,
                                  fsource='comid',
                                  file=comid_upstream_gauge_file):
    """Get the upstream gauges of a station.
    The <filename> is checked to see if comid is in the file. If it is, the
    upstream gauges are returned. If it is not, None is returned.
    
    Args:
    fid (str): The id for the site
    fsource (str): The source of the id (comid or nwissite)
    
    Returns:
    list: The list of upstream gauges.
    """
    # Check if station is in the file
    if not check_station_in_upstream_gauges(fid, file):
        return None
    else:
        with open(file, 'r') as f:
            upstream_gauges = json.load(f)
        return upstream_gauges[fid]
        

def get_upstream_gauges_single_site(fid, 
                        file=comid_upstream_gauge_file,
                        fsource='comid',
                        overwrite=False):
    """Get the upstream gauges of a station.
    The <filename> is checked to see if comid is in the file. If it is, the
    upstream gauges are returned. If it is not, the upstream gauges are found using pynhd. 

    Args:
    comid (str): The comid for the site

    Returns:
    list: The list of upstream gauges.
    """
    # Check if already in file
    upstream_gauges = get_upstream_gauges_from_file(fid,
                                                    fsource=fsource, 
                                                    file=file)
    
    upstream_gauges = None if overwrite else upstream_gauges
    if upstream_gauges is not None:
        return upstream_gauges
    
    # Else find the upstream gauges using pynhd
    nldi = nhd.NLDI()
    try:
        if fsource == 'nwissite' and 'USGS-' not in fid:
            usgs_fid = f'USGS-{fid}'
        else:
            usgs_fid = fid
        upstream_data = nldi.navigate_byid(fid = usgs_fid,
                                           fsource=fsource,
                                           source='nwissite',
                                           navigation='upstreamTributaries',
                                           distance=1000)
    
        # Store and clean gauge IDs (just numbers as strings)
        if fsource == 'comid':
            upstream_gauges = list(set(list(upstream_data['comid'].values)))
        elif fsource == 'nwissite':
            upstream_gauges = list(upstream_data['identifier'].values)
            upstream_gauges = [str(c.split('-')[1]) for c in upstream_gauges]
            upstream_gauges = list(set(upstream_gauges))
        
        # Remove the current gauge from the list
        upstream_gauges = [str(c) for c in upstream_gauges if c != fid]
        
    except:
        print(f'No upstream gauges found for {fsource}: {fid}')
        return None
    
    # update file    
    update_upstream_gauge_file({fid: upstream_gauges}, 
                                file=file,
                                overwrite=overwrite)
    return upstream_gauges

def get_upstream_gauges_for_id_list(fids,
                                    fsource='comid',
                                    file=comid_upstream_gauge_file,
                                    overwrite=False,
                                    restrict_to_list=False):
    """Get the upstream gauges for a list of comid sites.
    
    Args:
    comids (list): The list of comids for the sites
    
    Returns:
    dict: The upstream gauges for each comid.
    """
    N_SITES = len(fids)
    all_upstream_gauges = {}
    for i, id in enumerate(fids):
        if i % 50 == 0:
            print(f'Getting upstream gauges for {fsource} {i} of {N_SITES}')
        upstream_gauges = get_upstream_gauges_single_site(id, 
                                                          fsource=fsource, 
                                                          file=file,
                                                          overwrite=overwrite)
        if upstream_gauges is not None:
            upstream_gauges_from_list = [c for c in upstream_gauges if c in fids] if restrict_to_list else upstream_gauges
            all_upstream_gauges[id] = upstream_gauges_from_list
    return all_upstream_gauges

                        
def get_immediate_upstream_sites(upstream_gauges_dict, site_number):
    # Check if the site_number is in the dictionary
    if site_number not in upstream_gauges_dict.keys():
        raise ValueError(f"Site number {site_number} not found in the dictionary.")

    # Get all upstream sites for the given site_number
    all_upstream_sites = upstream_gauges_dict[site_number]

    # Identify immediate upstream sites
    immediate_sites = []
    for upstream_site in all_upstream_sites:
        # A site is immediate upstream if it does not appear as an upstream site
        # in the list of any other immediate upstream site
        if not any(upstream_site in upstream_gauges_dict.get(other_site, []) for other_site in all_upstream_sites if other_site != upstream_site):
            immediate_sites.append(upstream_site)
    immediate_sites= list(set(immediate_sites))
    return immediate_sites