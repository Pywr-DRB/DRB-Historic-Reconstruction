import os
import pandas as pd


from methods.retrieval.NHM import load_nhm_streamflow_from_hdf5
from methods.spatial.catchments import CatchmentManager

from methods.load.drb_boundary import load_drb_boundary
from methods.load.catchment_metadata import load_catchment_metadata
from methods.load.nldi_characteristics import load_nldi_characteristics
from methods.load.usgs_streamflow import load_usgs_streamflow
from methods.load.nhm_streamflow import load_nhm_streamflow
from methods.load.daymet import load_daymet_prcp
from methods.load.fdc import load_fdc

from config import DATA_DIR
from config import GEO_CRS

from config import (
    UNMANAGED_GAGE_METADATA_FILE,
    DIAGNOSTIC_SITE_METADATA_FILE,
    PREDICTION_LOCATIONS_FILE,
    ALL_USGS_DAILY_FLOW_FILE,
    PYWRDRB_NODE_CATCHMENT_FILE,
    USGS_GAGE_CATCHMENT_FILE,
)

data_types = [
    'metadata',
    'streamflow', 
    'nldi', 
    'catchments',
    'drb_boundary',
    'daymet'
]


sitetype_options = {
    'metadata' : ['all', 'unmanaged', 'diagnostic', 'pywrdrb'],
    'streamflow' : ['usgs', 'pywrdrb'],
    'catchments' : ['pywrdrb', 'usgs'],
    'daymet' : ['pywrdrb', 'usgs'],
    'nldi' : ['pywrdrb', 'usgs'],
    'drb_boundary' : None,
    'prediction_locations' : None,
    'fdc' : ['pywrdrb', 'gauge']
}

flowtype_options = {
    'streamflow' : ['usgs', 'nhm'],
    'fdc' : ['obs', 'nhm']
}



class Data():
    
    def __init__(self, **kwargs):
        pass
    
    def _verify_file_exists(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f'{file} not found')
    
    def convert_index_labels(self, df, metadata_df,
                             target_id, source_id):
        
        new_df = pd.DataFrame(index=metadata_df[target_id], 
                              columns = df.columns)
            
        for i in new_df.index:
            old_idx = metadata_df.set_index(target_id).loc[i, source_id]
            new_df.loc[i, :] = df.loc[old_idx, :]
        return new_df

    def convert_column_labels(self, df, metadata_df,
                                target_id, source_id):
        
        new_df = pd.DataFrame(index=df.index, 
                              columns = metadata_df[target_id])
        for i in new_df.columns:
            old_idx = metadata_df.set_index(target_id).loc[i, source_id]
            new_df.loc[:, i] = df.loc[:, old_idx]
        return new_df


    def load(self, 
             datatype=None, 
             flowtype=None,
             sitetype=None,
             timescale=None,
             **kwargs):
        
        if datatype is None:
            raise ValueError('datatype must be specified')
        if datatype in ['streamflow', 'metadata', 'catchments']:
            if sitetype is None:
                raise ValueError(f'sitetype must be specified for datatype={datatype}.')
            
        
        if datatype == 'streamflow':
            if flowtype == 'obs':
                data = load_usgs_streamflow(catchment_type=sitetype)
                
            elif flowtype == 'nhm':
                data = load_nhm_streamflow(catchment_type=sitetype)
                
                
        elif datatype == 'metadata':
            data = load_catchment_metadata(sitetype)

        elif datatype == 'catchments':
            manager = CatchmentManager()
            if sitetype == 'pywrdrb':
                data = manager.load_catchments(PYWRDRB_NODE_CATCHMENT_FILE)
                data.index = data.index.values.astype(int)
                data = data.loc[~data.index.duplicated(keep='first')]
                
                metadata = load_catchment_metadata('pywrdrb')
                metadata['comid'] = metadata['comid'].values.astype(int)
                data = self.convert_index_labels(data, metadata,
                                                 source_id='comid', target_id='name')
                
            elif sitetype == 'usgs':
                data = manager.load_catchments(USGS_GAGE_CATCHMENT_FILE)
        
        elif datatype == 'daymet':
            catchment_type = 'gauge' if sitetype == 'usgs' else 'pywrdrb'
            
            data = load_daymet_prcp(catchment_type=catchment_type, 
                                    timescale='monthly')
            
            if sitetype == 'pywrdrb':
                # Convert comid to pywrdrb name
                pywrdrb_metadata = load_catchment_metadata('pywrdrb')
                data = self.convert_column_labels(data, pywrdrb_metadata,
                                                 target_id='name', source_id='comid')


        elif datatype == 'nldi':
            catchment_type = 'pywrdrb' if sitetype == 'pywrdrb' else 'gauge'
            data = load_nldi_characteristics(catchment_type)
            
            if sitetype == 'pywrdrb':
                # Convert comid to pywrdrb name
                pywrdrb_metadata = load_catchment_metadata('pywrdrb')
                pywrdrb_metadata['comid'] = pywrdrb_metadata['comid'].values.astype(int)
                data = self.convert_index_labels(data, pywrdrb_metadata,
                                                 target_id='name', source_id='comid')

            elif sitetype == 'usgs':
                data.set_index('site_no', inplace=True, drop=True)

        
        elif datatype == 'drb_boundary':
            data = load_drb_boundary()
            
        elif datatype == 'prediction_locations':
            data = pd.read_csv(PREDICTION_LOCATIONS_FILE, index_col=0)
        
        elif datatype == 'fdc':
            data = load_fdc(catchment_type=sitetype,
                            flowtype=flowtype,
                            timescale=timescale)
        
        return data