import pandas as pd
from config import DATA_DIR
from config import NHM_DIAGNOSTIC_STREAMFLOW_FILE
from methods.retrieval.NHM import load_nhm_streamflow_from_hdf5
from methods.load.catchment_metadata import load_catchment_metadata

def load_nhm_streamflow(catchment_type):
    if catchment_type == 'usgs':
        metadata = load_catchment_metadata(catchment_type)

        # drop metadata rows with nan or inf nhm_id
        metadata = metadata[metadata['nhm_id'].notna()]
        
        # set nhm_id to int
        metadata['nhm_id'] = metadata['nhm_id'].astype(int)
        metadata['nhm_id'] = metadata['nhm_id'].dropna().astype(int)
        
        Q_nhm = load_nhm_streamflow_from_hdf5(id_subset = None, 
                                      site_metadata=metadata,
                                      column_labels='site_no')
        
    elif catchment_type == 'diagnostic':
        Q_nhm = pd.read_csv(NHM_DIAGNOSTIC_STREAMFLOW_FILE, 
                            index_col=0, 
                            parse_dates=True)

    elif catchment_type == 'pywrdrb':
        
        df =  pd.read_csv(f'{DATA_DIR}/NHMv10/gage_flow_nhmv10.csv', 
                                 index_col=0, parse_dates=True)
        Q_nhm = df.copy()
        
    Q_nhm = Q_nhm.loc['1983-10-01':, :]
    return Q_nhm