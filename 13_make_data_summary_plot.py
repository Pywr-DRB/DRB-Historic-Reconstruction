import numpy as np
import pandas as pd
import geopandas as gpd

from config import DATA_DIR
from config import GEO_CRS
from config import UNMANAGED_GAGE_METADATA_FILE
from config import DIAGNOSTIC_SITE_METADATA_FILE
from config import PREDICTION_LOCATIONS_FILE
from config import ALL_USGS_DAILY_FLOW_FILE


from methods.plotting.data_plots import plot_data_summary



### Load data

## Flow
Q_obs = pd.read_csv(ALL_USGS_DAILY_FLOW_FILE, 
                    index_col=0, parse_dates=True)

## Spatial
drb_boundary = gpd.read_file(f'{DATA_DIR}DRB_spatial/DRB_shapefiles/drb_bnd_polygon.shp').to_crs(GEO_CRS)


## Metadata
prediction_locations = pd.read_csv(PREDICTION_LOCATIONS_FILE, sep = ',', index_col=0)
prediction_locations = gpd.GeoDataFrame(prediction_locations, 
                                        geometry=gpd.points_from_xy(prediction_locations.long, prediction_locations.lat))


unmanaged_gauge_meta = pd.read_csv(UNMANAGED_GAGE_METADATA_FILE, 
                           index_col=0, dtype={'site_no':str,
                                               'comid':int})
unmanaged_gauge_meta.set_index('site_no', drop=False, inplace=True)


diagnostic_gauge_meta = pd.read_csv(DIAGNOSTIC_SITE_METADATA_FILE, 
                           index_col=0, dtype={'site_no':str,
                                                  'comid':int})
diagnostic_site_list = diagnostic_gauge_meta['site_no'].values


# Filter Q_obs to only include unmanaged gauges
Q_obs = Q_obs.loc[:, unmanaged_gauge_meta.index]



### Diagnostic data summary

print('Making data summary plot...')
plot_data_summary(Q_obs, 
                    unmanaged_gauge_meta, 
                    diagnostic_site_list, 
                    prediction_locations, 
                    drb_boundary, 
                    plot_mainstem=True, 
                    plot_tributaries =True,
                    highlight_diagnostic_sites=False,
                    sort_by='lat')

