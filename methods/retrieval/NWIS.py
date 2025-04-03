import numpy as np
import pandas as pd
import geopandas as gpd
from pygeohydro import NWIS
import pynhd as pynhd

from methods.utils.filter import filter_usgs_nwis_query_by_type, filter_drb_sites
from methods.retrieval.NLDI import add_comid_metadata_to_gage_data
from config import FILTER_DRB, MIN_YEARS

class RegionalGageDataRetriever:
    def __init__(self,
                 filter_drb = FILTER_DRB,
                 min_years_of_data = MIN_YEARS):
        
        self.gage_data = pd.DataFrame()
        self.stations = []
        self.filter_drb = filter_drb
        self.min_years_of_data = min_years_of_data
    
    def filter_usgs_nwis_query_by_type(self, query_result):
        return filter_usgs_nwis_query_by_type(query_result)
    
    
    def filter_drb_sites(self, gage_data):
        return filter_drb_sites(gage_data)
    
    
    def filter_gage_data_by_record_length(self, 
                                          gage_data):
    
        # Check that begin_date and end_date columns exist
        if 'begin_date' not in gage_data.columns or 'end_date' not in gage_data.columns:
            raise ValueError("gage_data must have columns 'begin_date' and 'end_date'")
        
        ### Filter data 
        ## Drop sites with less than 10 years of data
        gage_data['years_of_data'] = (pd.to_datetime(gage_data['end_date']) - pd.to_datetime(gage_data['begin_date'])).dt.days / 365.25
        gage_data = gage_data[gage_data['years_of_data'] > self.min_years_of_data]
        return gage_data
    
    
    def get_stations_in_bbox(self, 
                             bbox, 
                             return_gage_data = False):
        nwis = NWIS()
        query_request = {"bBox": ",".join(f"{b:.06f}" for b in bbox),
                            "hasDataTypeCd": "dv",
                            "outputDataTypeCd": "dv"}
        query_result = nwis.get_info(query_request, expanded= True)
        query_result = self.filter_usgs_nwis_query_by_type(query_result)
        
        ### Location data (long,lat)
        gage_data = query_result[['site_no', 'dec_long_va', 'dec_lat_va', 'begin_date', 'end_date']]
        gage_data.columns = ['site_no', 'long', 'lat', 'begin_date', 'end_date']        
        gage_data.index = gage_data['site_no']
        gage_data= gage_data.drop('site_no', axis=1)
        
        if self.filter_drb:
            gage_data = self.filter_drb_sites(gage_data)
        
        gage_data = self.filter_gage_data_by_record_length(gage_data)
        
        # drop duplicates, keep row with largest years_of_data
        sorted_gage_data = gage_data.sort_values('years_of_data', ascending=False)
        gage_data = sorted_gage_data[~sorted_gage_data.index.duplicated(keep='first')]
        
        stations = list(set(gage_data.index.values))
        
        self.stations = stations
        self.gage_data = gage_data
        
        if return_gage_data:
            return gage_data

    
    def add_comid_metadata_to_gage_data(self, 
                                        return_gage_data = False,
                                        drop_duplicates = True,
                                        drop_na = True):
        
        gage_data_with_comid = add_comid_metadata_to_gage_data(self.gage_data)
        
        if drop_duplicates:
            gage_data_with_comid = gage_data_with_comid.drop_duplicates(subset=['comid'])
        if drop_na:
            gage_data_with_comid = gage_data_with_comid.dropna(how='any', axis=0)
        gage_data_with_comid['comid'] = gage_data_with_comid['comid'].astype(str)
        
        self.gage_data_with_comid = gage_data_with_comid
        if return_gage_data:
            return self.gage_data_with_comid

    def make_station_to_comid_dicts(self):
        # Create a dictionary directly mapping station IDs to COMIDs using zip
        self.station_to_comid = dict(zip(self.gage_data_with_comid.index, self.gage_data_with_comid['comid']))
        self.comid_to_station = dict(zip(self.gage_data_with_comid['comid'], self.gage_data_with_comid.index))
        
    
    def station_to_comid(self, station):
        return self.station_to_comid[station]
    
    
    def comid_to_station(self, comid):
        return self.comid_to_station[comid]
    
            
    def get_station_list(self):
        return self.stations
    
    def get_streamflow_data(self, 
                            station_list,
                            dates):
        ## Use NWIS to get timeseries data
        nwis = NWIS()
        Q = nwis.get_streamflow(station_list, dates)
        Q.index = pd.to_datetime(Q.index.date)
        Q.columns = [usgs_id.split('-')[1] for usgs_id in Q.columns]
        return Q
