import pandas as pd
import geopandas as gpd
from methods.load.catchment_metadata import load_catchment_metadata
from config import USGS_NLDI_CHARACTERISTICS_FILE, PYWRDRB_NLDI_CHARACTERISTICS_FILE


def load_nldi_characteristics(catchment_type):
    
    if catchment_type == 'gauge':
        df = pd.read_csv(USGS_NLDI_CHARACTERISTICS_FILE, index_col=0)
        
        metadata = load_catchment_metadata('all')
        
        df['site_no'] = df.index.map(metadata.set_index('comid')['site_no'])
        
        return df
    
    elif catchment_type == 'pywrdrb':
        return pd.read_csv(PYWRDRB_NLDI_CHARACTERISTICS_FILE, index_col=0)
