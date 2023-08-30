"""
Queries streamflow data from the USGS NWIS, identifies data within the DRB and removes streamflows which are labeled as being 
downstream of reservoirs.

Data is retrieved from 1900 onward to the present. 
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd

from pygeohydro import NWIS
import pynhd as pynhd

## Import latest pywrdrb node data specs
from methods.generator.pywr_drb_node_data import obs_site_matches, obs_pub_site_matches
from methods.processing.filter import filter_drb_sites

### 0: Specifications ###
filter_drb = True
bbox = (-77.8, 37.5, -74.0, 44.0)
dates = ('1900-01-01', '2022-12-31')
pywrdrb_dir = '../Pywr-DRB/'


### 0.1: Request and save specific pywrdrb gauge flows
## Get historic observations that exist (including management)
pywrdrb_stations = []
for node, sites in obs_site_matches.items():
    if sites:
        for s in sites:
            pywrdrb_stations.append(s)

nwis = NWIS()
Q_pywrdrb = nwis.get_streamflow(pywrdrb_stations, dates)
Q_pywrdrb.index = pd.to_datetime(Q_pywrdrb.index.date)

for s in pywrdrb_stations:
    assert(f'USGS-{s}' in Q_pywrdrb.columns),'PywrDRB gauge {s} is missing from the data.'

# Export
Q_pywrdrb.to_csv('./outputs/streamflow_daily_usgs_1950_2022_cms.csv', sep=',')
Q_pywrdrb.to_csv(f'{pywrdrb_dir}/input_data/usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', sep=',')


### Unmanaged flows: For the prediction at ungauged or managed locations
### we want only unmanaged flow data.  The following retrieves, filters, and exports unmanaged flows across the basin. 
### 1: Query USGS data ###
# Use the national water info system (NWIS)
nwis = NWIS()
print("Initialized")

# Send a query_request for all gage info in the bbox
query_request = {"bBox": ",".join(f"{b:.06f}" for b in bbox),
                    "hasDataTypeCd": "dv",
                    "outputDataTypeCd": "dv"}

query_result = nwis.get_info(query_request, expanded= False, nhd_info= False)

# Filter non-streamflow stations
query_result = query_result.query("site_tp_cd in ('ST','ST-TS')")
query_result = query_result[query_result.parm_cd == '00060']  # https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
query_result = query_result.reset_index(drop = True)

stations = list(set(query_result.site_no.tolist()))
print(f"Gage data gathered, {len(stations)} USGS streamflow gauges found in date range.")


### Location data (long,lat)
gage_data = query_result[['site_no', 'dec_long_va', 'dec_lat_va', 'begin_date', 'end_date']]
gage_data.columns = ['site_no', 'long', 'lat', 'begin_date', 'end_date']
gage_data.index = gage_data['site_no']
gage_data= gage_data.drop('site_no', axis=1)


### 2: Filter data ###
## Remove sites outside the DRB boundary
if filter_drb:
    gage_data = filter_drb_sites(gage_data)
gage_data = gage_data[~gage_data.index.duplicated(keep = 'first')]
stations = gage_data.index.to_list()
print(f'{len(stations)} streamflow gauges after filtering.')

## Remove managed sites
# To do this, wee will use NLDI attributes to find managed sites
# Initialize the NLDI database
nldi = pynhd.NLDI()

# Get COMID for each gauge
gage_comid = pd.DataFrame(index = gage_data.index, columns=['comid', 'reachcode', 'comid-long', 'comid-lat'])
for st in gage_data.index:
    coords = (gage_data.loc[st, ['long']].values[0], gage_data.loc[st, ['lat']].values[0])
    try:
        found = nldi.comid_byloc(coords)
        gage_comid.loc[st, ['comid']] = found.comid.values[0]
        gage_comid.loc[st, ['reachcode']] = found.reachcode.values[0]
        gage_comid.loc[st, ['comid-long']] = found.geometry.x[0]
        gage_comid.loc[st, ['comid-lat']] = found.geometry.y[0]
    except:
        print(f'Error getting COMID for site {st}')
        
gage_data = pd.concat([gage_data, gage_comid], axis=1)
gage_data = gage_data.dropna(axis=0)
gage_data["comid"] = gage_data["comid"].astype('int')

# Specific characteristics of interest, for now we only want reservoir information
all_characteristics = nldi.valid_characteristics
reservoir_characteristics = ['CAT_NID_STORAGE2013', 'CAT_NDAMS2013', 'CAT_MAJOR2013', 'CAT_NORM_STORAGE2013']
TOT_reservoir_characteristics = ['TOT_NID_STORAGE2013', 'TOT_NDAMS2013', 'TOT_MAJOR2013', 'TOT_NORM_STORAGE2013']

## Use the station IDs to retrieve basin information
cat_chars = nldi.getcharacteristic_byid(gage_data.comid, fsource = 'comid', 
                                        char_type= "tot", char_ids= TOT_reservoir_characteristics)

cat = cat_chars.reset_index()
cat.columns = ['comid', 'TOT_MAJOR2013', 'TOT_NDAMS2013',	'TOT_NID_STORAGE2013',	'TOT_NORM_STORAGE2013']
print(f'Found characteristics for {cat_chars.shape} of {gage_data.shape} basins.')

## Make a list of known inflow gauges we want to use
obs_pub_stations = []
for node, sites in obs_pub_site_matches.items():
    if sites:
        if len(sites) > 0:
            for s in sites:
                obs_pub_stations.append(s)

## Remove sites that have reservoirs upstream
gage_with_cat_chars = pd.merge(gage_data, cat, on = "comid")
gage_with_cat_chars.index = gage_data.index
managed_stations = []
for i, st in enumerate(gage_data.index):
    if gage_with_cat_chars.loc[st, TOT_reservoir_characteristics].sum() > 0:
        if st not in obs_pub_stations:
            managed_stations.append(st)

# Take data from just unmanaged
unmanaged_gauge_data = gage_data.drop(managed_stations)
print(f'{len(managed_stations)} of the {gage_data.shape[0]} gauge stations are managed and being removed.')

# Export gage_data
gage_data.to_csv('./data/drb_all_usgs_metadata.csv', sep=',')
unmanaged_gauge_data.to_csv('./data/drb_unmanaged_usgs_metadata.csv', sep=',')


### 3. Retrieve unmanaged flow data

# Retrieve data using NWIS
stations = unmanaged_gauge_data.index
nwis = NWIS()
Q = nwis.get_streamflow(stations, dates)

# Export all data to CSV
Q.to_csv(f'./data/drb_historic_unmanaged_streamflow_cms.csv', sep=',')