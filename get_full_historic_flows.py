"""
Trevor Amestoy
Cornell University

This script prepares data for a multi-output neural net.

This script handles each gauge individually and gets the maximum flow data record legnth without missing data.
"""
#%%

import numpy as np
import pandas as pd
import geopandas as gpd

# From the PyNHD library, import data acuistion tools
from pygeohydro import NWIS, plot

#%% 
drb_gauges = pd.read_csv('data/drb_gauges.csv', sep = ',')



#%%
def filter_drb_sites(x, site_ids = None):
    sdir = '../DRB_water_management/DRB_spatial/DRB_shapefiles'
    crs = 4386
    if type(x) == np.ndarray:
        x = pd.DataFrame(x)

    if site_ids is not None:
        x.index = site_ids

    drb_boarder = gpd.read_file(f'{sdir}/drb_bnd_polygon.shp')
    drb_boarder = drb_boarder.to_crs(crs)
    x_all = gpd.GeoDataFrame(x, geometry = gpd.points_from_xy(x.long, x.lat, crs = crs))
    x_filtered = gpd.clip(x_all, drb_boarder)
    return x_filtered

#%%#############################################################################
### Step 1) Data and specifications
################################################################################
filter_drb = True
bbox = (-77.8, 37.5, -74.0, 44.0)
dates = ("1980-01-01", "2022-12-31")

#%%#############################################################################
### Step 2) Query and filter data
################################################################################

# Use the national water info system (NWIS)
nwis = NWIS()
print("Initialized")

# Send a query_request for all gage info in the bbox
query_request = {"bBox": ",".join(f"{b:.06f}" for b in bbox),
        "hasDataTypeCd": "dv",
        "outputDataTypeCd": "dv"}

query_result = nwis.get_info(query_request)

# Filter non-streamflow stations
query_result = query_result.query("site_tp_cd in ('ST','ST-TS')")
query_result = query_result[query_result.parm_cd == '00060']  # https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
query_result = query_result.reset_index(drop = True)
stations = list(set(query_result.site_no.tolist()))
print(f"Gage data gathered, {len(stations)} USGS streamflow gauges found in date range.")

### Location data (long,lat)
# Initialize storage
loc_data = np.zeros((len(stations), 2))
loc_data_names = np.array(['site_no', 'long', 'lat'])

# Loop through gage info queried earlier
for i,st in enumerate(stations):
    # Pull the lat-long
    long = query_result.set_index('site_no').loc[st.split('-')[1]]['dec_long_va']
    lat = query_result.set_index('site_no').loc[st.split('-')[1]]['dec_lat_va']
    
    # Store
    loc_data[i,0] = st
    loc_data[i,1] = long if len(long.shape) == 0 else long[0]
    loc_data[i,2] = lat if len(lat.shape) == 0 else lat[0]

if filter_drb:
    x = filter_drb_sites(loc_data[:, 1:], sites = loc_data[:,0])
    stations = x.index.to_list()
print(f'{len(stations)} streamflow gauges after filtering.')

#%%#############################################################################
### Step 3) Retrieve data
################################################################################

print("Requesting data...")

qobs = nwis.get_streamflow(stations, dates, mmd=False)
stations = qobs.columns.to_numpy()

print(f'All gage data sourced. You have {qobs.shape[1]} gages after cleaning.')


#%%#############################################################################
### Step 4) Export data
################################################################################

qobs.to_csv(f'./data/full_historic_observed_flow.csv', sep = ',')
geo_data = pd.DataFrame(loc_data, index = qobs.columns, columns = loc_data_names).to_csv(f'./data/all_gauge_locations.csv', sep = ',')

print('DONE! See the data folder.')
