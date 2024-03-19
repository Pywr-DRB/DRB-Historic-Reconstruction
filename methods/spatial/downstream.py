"""
Used for identifying downstream gauges relative to 
USGS gauge locations or NHD comid numbers.

This is the downstream version of upstream.py.

The module contains the following functions:
    * ensure_file_exists - Ensure that the file exists. If it does not, create it with initial JSON data.
    * update_downstream_gauge_file - Update the downstream gauges file without overwriting it.
    * check_station_in_downstream_gauges - Check if the fid is in the file.
    * get_downstream_gauges_from_file - Get the downstream gauges of a station.
    * get_downstream_gauges_single_site - Get the downstream gauges of a station.
    * get_downstream_gauges_for_id_list - Get the downstream gauges for a list of comid sites.
    * get_immediate_downstream_sites - Get the immediate downstream sites for a given site.

"""
import os
import json
import pynhd as nhd

from methods.utils.directories import DATA_DIR
from methods.utils.contants import GEO_CRS

comid_downstream_gauge_file = f'{DATA_DIR}/comid_downstream_gauges.json'
station_downstream_gauge_file = f'{DATA_DIR}/station_downstream_gauges.json'

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

def update_downstream_gauge_file(all_downstream_gauges,
                                file=comid_downstream_gauge_file,
                                overwrite=False):
    """Update the downstream gauges file without overwriting it.
    
    Args:
    all_downstream_gauges (dict): The downstream gauges for each comid.
    """
    ensure_file_exists(file)
    
    # Load
    with open(file, 'r') as f:
        archieved_downstream_gauges = json.load(f)
        
    # Modify 
    for id in all_downstream_gauges.keys():
        if overwrite:
            archieved_downstream_gauges[id] = all_downstream_gauges[id]
        elif not overwrite and id not in archieved_downstream_gauges.keys():
            archieved_downstream_gauges[id] = all_downstream_gauges[id]
    # Save  
    with open(file, 'w') as f:  
        json.dump(archieved_downstream_gauges, f)
    return      


def check_station_in_downstream_gauges(fid,
                                    file=comid_downstream_gauge_file):
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
        downstream_gauges = json.load(f)
        if fid in downstream_gauges.keys():
            return True
        else:
            return False        
        

def get_downstream_gauges_from_file(fid,
                                  fsource='comid',
                                  file=comid_downstream_gauge_file):
    """Get the downstream gauges of a station.
    The <filename> is checked to see if comid is in the file. If it is, the
    downstream gauges are returned. If it is not, None is returned.
    
    Args:
    fid (str): The id for the site
    fsource (str): The source of the id (comid or nwissite)
    
    Returns:
    list: The list of downstream gauges.
    """
    # Check if station is in the file
    if not check_station_in_downstream_gauges(fid, file):
        return None
    else:
        with open(file, 'r') as f:
            downstream_gauges = json.load(f)
        return downstream_gauges[fid]
        

def get_downstream_gauges_single_site(fid, 
                        file=comid_downstream_gauge_file,
                        fsource='comid',
                        overwrite=False):
    """Get the downstream gauges of a station.
    The <filename> is checked to see if comid is in the file. If it is, the
    downstream gauges are returned. If it is not, the downstream gauges are found using pynhd. 

    Args:
    comid (str): The comid for the site

    Returns:
    list: The list of downstream gauges.
    """
    # Check if already in file
    downstream_gauges = get_downstream_gauges_from_file(fid,
                                                    fsource=fsource, 
                                                    file=file)
    
    downstream_gauges = None if overwrite else downstream_gauges
    if downstream_gauges is not None:
        return downstream_gauges
    
    # Else find the downstream gauges using pynhd
    nldi = nhd.NLDI()
    try:
        if fsource == 'nwissite' and 'USGS-' not in fid:
            usgs_fid = f'USGS-{fid}'
        else:
            usgs_fid = fid
        downstream_data = nldi.navigate_byid(fid = usgs_fid,
                                           fsource=fsource,
                                           source='nwissite',
                                           navigation='downstreamTributaries',
                                           distance=1000)
    
        # Store and clean gauge IDs (just numbers as strings)
        if fsource == 'comid':
            downstream_gauges = list(set(list(downstream_data['comid'].values)))
        elif fsource == 'nwissite':
            downstream_gauges = list(downstream_data['identifier'].values)
            downstream_gauges = [str(c.split('-')[1]) for c in downstream_gauges]
            downstream_gauges = list(set(downstream_gauges))
        
        # Remove the current gauge from the list
        downstream_gauges = [str(c) for c in downstream_gauges if c != fid]
        
    except:
        print(f'No downstream gauges found for {fsource}: {fid}')
        return None
    
    # update file    
    update_downstream_gauge_file({fid: downstream_gauges}, 
                                file=file,
                                overwrite=overwrite)
    return downstream_gauges

def get_downstream_gauges_for_id_list(fids,
                                    fsource='comid',
                                    file=comid_downstream_gauge_file,
                                    overwrite=False,
                                    restrict_to_list=False):
    """Get the downstream gauges for a list of comid sites.
    
    Args:
    comids (list): The list of comids for the sites
    
    Returns:
    dict: The downstream gauges for each comid.
    """
    N_SITES = len(fids)
    all_downstream_gauges = {}
    for i, id in enumerate(fids):
        if i % 50 == 0:
            print(f'Getting downstream gauges for {fsource} {i} of {N_SITES}')
        downstream_gauges = get_downstream_gauges_single_site(id, 
                                                          fsource=fsource, 
                                                          file=file,
                                                          overwrite=overwrite)
        if downstream_gauges is not None:
            downstream_gauges_from_list = [c for c in downstream_gauges if c in fids] if restrict_to_list else downstream_gauges
            all_downstream_gauges[id] = downstream_gauges_from_list
    return all_downstream_gauges

                        
def get_immediate_downstream_sites(downstream_gauges_dict, site_number):
    # Check if the site_number is in the dictionary
    if site_number not in downstream_gauges_dict:
        raise ValueError(f"Site number {site_number} not found in the dictionary.")

    # Get all downstream sites for the given site_number
    all_downstream_sites = downstream_gauges_dict[site_number]

    # Identify immediate downstream sites
    immediate_sites = []
    for downstream_site in all_downstream_sites:
        # A site is immediate downstream if it does not appear as an downstream site
        # in the list of any other immediate downstream site
        if not any(downstream_site in downstream_gauges_dict.get(other_site, []) for other_site in all_downstream_sites if other_site != downstream_site):
            immediate_sites.append(downstream_site)
    immediate_sites= list(set(immediate_sites))
    return immediate_sites