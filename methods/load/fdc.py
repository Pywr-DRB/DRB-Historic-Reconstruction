import pandas as pd
from config import OUTPUT_DIR

def load_fdc(catchment_type = 'gauge', 
                              flowtype='obs',
                              timescale='daily'):
    if catchment_type in ['gauge', 'usgs']:
        df = pd.read_csv(f'{OUTPUT_DIR}/{flowtype}_diagnostic_gauge_{timescale}_fdc.csv',
                        index_col=0, dtype={'site_no':str})    
    
    elif catchment_type == 'pywrdrb' and flowtype == 'nhm':
        df = pd.read_csv(f'{OUTPUT_DIR}/{flowtype}_pywrdrb_node_{timescale}_fdc.csv',
                        index_col=0, dtype={'comid':int})
    return df