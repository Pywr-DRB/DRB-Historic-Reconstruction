import pandas as pd
from config import ALL_USGS_DAILY_FLOW_FILE, OBS_DIAGNOSTIC_STREAMFLOW_FILE
from methods.utils.constants import cms_to_mgd


def load_usgs_streamflow(catchment_type='usgs', units = 'MGD'):
    if catchment_type == 'usgs':
        df = pd.read_csv(ALL_USGS_DAILY_FLOW_FILE, 
                        index_col=0, parse_dates=True) 
        if units == 'CMS':
            pass    
        elif units == 'MGD':
            df *= cms_to_mgd
            
    elif catchment_type == 'diagnostic':
        df = pd.read_csv(OBS_DIAGNOSTIC_STREAMFLOW_FILE, 
                        index_col=0, parse_dates=True)
        if units == 'CMS':
            df /= cms_to_mgd
        elif units == 'MGD':
            pass

    return df
