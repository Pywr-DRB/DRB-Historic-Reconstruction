import pandas as pd

from methods.utils.directories import DATA_DIR
from methods.utils.constants import GEO_CRS
from methods.processing.catchments import load_station_catchments
from methods.processing.load import load_model_segment_flows, load_gauge_matches
from methods.utils.constants import cms_to_mgd, GEO_CRS
from methods.processing.prep_loo import get_leave_one_out_sites
import concurrent.futures

from pydaymet import pydaymet

### SPECIFICATIONS
MAX_ALLOWABLE_STORAGE = 2000 # acre-feet of total storage in catchment
boundary = 'drb'
N_CORE = 7
timescale = 'daily'
dt_timescale = 'MS' if timescale == 'monthly' else 'D'

start_date = '1980-01-01'
end_date = '2020-12-31'

vars = ['prcp']  #, 'tmax', 'tmin' , 'swe']


def process_catchment(catchment, index, start_date, end_date, timescale, vars, daymet_dates=None):
    try:
        print(f'Processing {index}')
        catchment_geom = catchment['geometry']

        data = pydaymet.get_bygeom(catchment_geom, 
                                   (start_date, end_date), 
                                   time_scale=timescale,
                                   variables=vars)
        if data['prcp'].shape[1] < 20 or data['prcp'].shape[2] < 20:
            print(f'Grid size: {data["prcp"].shape}')
            
        prcp_sum = data['prcp'].sum(axis=1).sum(axis=1).values
        prcp_avg = data['prcp'].mean(axis=1).mean(axis=1).values

        if daymet_dates is None:
            daymet_dates = pd.to_datetime(data['prcp'].time.values)

        return index, prcp_sum, prcp_avg, daymet_dates

    except Exception as e:
        print(f'Error processing {index}: {e}')
        return index, None, None, None


if __name__ == '__main__':
    
    ######### Load
    # usgs gauge data
    Q = pd.read_csv(f'{DATA_DIR}/USGS/drb_streamflow_daily_usgs_cms.csv',
                    index_col=0, parse_dates=True)*cms_to_mgd

    # usgs gauge metadata
    gauge_meta = pd.read_csv(f'{DATA_DIR}/USGS/drb_usgs_metadata.csv', 
                                index_col=0, dtype={'site_no': str, 'comid': str})
    ## Get unmanaged catchments
    unmanaged_gauge_meta = gauge_meta[gauge_meta['total_catchment_storage'] < MAX_ALLOWABLE_STORAGE]
    unmanaged_marginal_gauge_meta = gauge_meta[gauge_meta['marginal_catchment_storage'] < MAX_ALLOWABLE_STORAGE]


    # Gauge matches have columns for both station number and comid or feature id
    gauge_matches = load_gauge_matches()

    # catchment geometries
    station_catchments = load_station_catchments(boundary=boundary,
                                                    crs=GEO_CRS,
                                                    marginal=False)


    ## Get loo sites
    loo_sites = get_leave_one_out_sites(Q, unmanaged_gauge_meta.index.values,
                                            gauge_matches['nwmv21']['site_no'].values,
                                            gauge_matches['nhmv10']['site_no'].values)
    # Do this only for loo sites
    loo_station_catchments = station_catchments[station_catchments.index.isin(loo_sites)]


    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORE) as executor:
        # Prepare a list of futures
        futures = [executor.submit(process_catchment, catchment, index, start_date, end_date, timescale, vars)
                for index, catchment in loo_station_catchments.iterrows()]

        # Initialize data frames outside of loop
        catchment_sum_prcp = pd.DataFrame(columns=station_catchments.index)
        catchment_avg_prcp = pd.DataFrame(columns=station_catchments.index)

        for future in concurrent.futures.as_completed(futures):
            index, prcp_sum, prcp_avg, daymet_dates = future.result()
            if prcp_sum is not None and prcp_avg is not None:
                catchment_sum_prcp[index] = prcp_sum
                catchment_avg_prcp[index] = prcp_avg

            # Initialize the date index only once
            if 'daymet_dates' not in locals() and daymet_dates is not None:
                catchment_sum_prcp.index = daymet_dates
                catchment_avg_prcp.index = daymet_dates


    # Export
    catchment_sum_prcp.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_sum_prcp_{timescale}.csv')
    catchment_avg_prcp.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_avg_prcp_{timescale}.csv')


    # catchment_avg_tmax.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_avg_tmax_{timescale}.csv')
    # catchment_avg_tmin.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_avg_tmin_{timescale}.csv')
    # catchment_sum_swe.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_sum_swe_{timescale}.csv')
    # catchment_avg_swe.to_csv(f'{DATA_DIR}/Daymet/{boundary}_catchment_avg_swe_{timescale}.csv')
