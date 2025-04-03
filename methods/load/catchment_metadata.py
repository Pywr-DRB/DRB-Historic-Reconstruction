import pandas as pd
from config import ALL_SITE_METADATA_FILE
from config import UNMANAGED_GAGE_METADATA_FILE
from config import DIAGNOSTIC_SITE_METADATA_FILE
from config import PYWRDRB_NODE_METADATA_FILE


def load_catchment_metadata(site_type):
    if site_type in ['all', 'usgs']:
        df =  pd.read_csv(ALL_SITE_METADATA_FILE, 
                           index_col=0, dtype={'site_no':str,
                                               'comid':int})
        return df
        
    elif site_type == 'unmanaged':
        return pd.read_csv(UNMANAGED_GAGE_METADATA_FILE, 
                           index_col=0, dtype={'site_no':str,
                                               'comid':int})
    elif site_type == 'diagnostic':
        return pd.read_csv(DIAGNOSTIC_SITE_METADATA_FILE, 
                           index_col=0, dtype={'site_no':str,
                                                  'comid':int})
        
    elif site_type == 'pywrdrb':
        df = pd.read_csv(PYWRDRB_NODE_METADATA_FILE, 
                           index_col=0, dtype={'comid':str})
        df['name'] = df.index.values
        return df
