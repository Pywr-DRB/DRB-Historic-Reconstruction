import numpy as np
import pandas as pd

from pygeohydro import NWIS
import pynhd as pynhd
from methods.utils.contants import GEO_CRS

def get_comid_metadata_from_coords(coords):
    """Get the comid from coordinates.  
    
    Args:
        coords (tuple; long, lat): The coordinates of the site.
    """    
    nldi = pynhd.NLDI()
    try:
        comid = nldi.comid_byloc(coords,
                                 loc_crs=GEO_CRS)
        return comid
    except:
        print('No comid found for coords: ', coords)
        return None

def add_comid_metadata_to_gage_data(gage_data_df):
    gage_data = gage_data_df.copy()
    
    N_SITES = len(gage_data)
    gage_comid = pd.DataFrame(index = gage_data.index, 
                              columns=['comid', 'reachcode', 
                                       'comid-long', 'comid-lat'])
    
    for i, st in enumerate(gage_data.index):
        
        if i % 50 == 0:
            print(f'Getting comid for site {i} of {N_SITES}')
            
        coords = (gage_data.loc[st, ['long']].values[0], 
                  gage_data.loc[st, ['lat']].values[0])
        comid_metadata = get_comid_metadata_from_coords(coords)
        if comid_metadata is not None:
            gage_comid.loc[st, ['comid']] = comid_metadata.comid.values[0]
            gage_comid.loc[st, ['reachcode']] = comid_metadata.reachcode.values[0]
            gage_comid.loc[st, ['comid-long']] = comid_metadata.geometry.x[0]
            gage_comid.loc[st, ['comid-lat']] = comid_metadata.geometry.y[0]
        else:
            gage_comid.loc[st, ['comid']] = np.nan
    gage_data = pd.concat([gage_data, gage_comid], axis=1)
    gage_data = gage_data.dropna(axis=0)
    return gage_data