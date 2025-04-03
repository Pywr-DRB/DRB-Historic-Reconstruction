
import pandas as pd
from config import DATA_DIR

def load_nhm_streamflow_from_hdf5(id_subset=None,
                                  site_metadata = None,
                                  column_labels='nhm_id',
                                  data_dir = DATA_DIR):
    
    segment_flows = pd.read_hdf(f'{data_dir}/NHMv10/hdf/drb_seg_outflow_mgd.hdf5', 
                                    key = 'df')
    segment_flows.index = pd.to_datetime(segment_flows.index)
    
    df = segment_flows
    
    # keep only 10-01-1983 onwards
    df = df.loc['1983-10-01':, :]
    
    if id_subset is not None:
        df = segment_flows.loc[:, id_subset]
    
    # Relabel columns
    if column_labels == 'site_no':
        df.columns = df.columns.astype(int)
        
        # now relabel using site_no
        matching_ids = site_metadata['nhm_id'].loc[site_metadata['nhm_id'].isin(df.columns)]
        
        new_cols = site_metadata.set_index('nhm_id').loc[matching_ids, 'site_no']
        new_cols.dropna(inplace=True)
        
        df = df.loc[:, new_cols.index]
        df.columns = new_cols

    
        
    elif column_labels == 'nhm_id':
        pass
    elif column_labels == 'comid':
        df.columns = df.columns.map(site_metadata.set_index('nhm_id')['comid'])
        
    return df