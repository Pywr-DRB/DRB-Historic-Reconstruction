from pywrdrb.plotting.styles import (
    model_label_dict,
    model_colors_historic_reconstruction,
)
for m in list(model_colors_historic_reconstruction.keys()):
    model_colors_historic_reconstruction[
        f"pywr_{m}"
    ] = model_colors_historic_reconstruction[m]



import pandas as pd
import numpy as np

import pywrdrb
from pywrdrb import Data
from pywrdrb.utils.reservoir_data import get_reservoir_capacity
from pywrdrb.utils.timeseries import subset_timeseries

from methods.plotting.flow_contribution import plot_NYC_release_components_combined
from config import DATA_DIR

# Set up directories
output_dir = './pywrdrb_outputs/'
input_dir = './pywrdrb_inputs/'
fig_dir = './figures/nyc_storage_and_flow_contributions/'
pywrdrb.set_directory(input_dir = input_dir)


export_data = False
data_export_fname = f"{output_dir}/pywrdrb_results_export_obs_pub_nhmv10_BC_ObsScaled_ensemble.hdf5"
reload_exported_data = True

### Alternative windows of focus
start_date = pd.to_datetime('1945-01-01')
end_date = pd.to_datetime('2022-12-31')

start_1960s_drought = pd.to_datetime('1963-01-01')
end_1960s_drought = pd.to_datetime('1967-12-31')

start_1980s_drought = pd.to_datetime('1980-06-01')
end_1980s_drought = pd.to_datetime('1982-10-31')

start_ffmp = pd.to_datetime('2017-10-01')
end_ffmp = pd.to_datetime('2022-12-31')


date_ranges = {'1960s_drought' : (start_1960s_drought, end_1960s_drought),
               'post_ffmp' : (start_ffmp, end_ffmp),
               }


########################################
### Load data
########################################

datatypes = ['obs', 'output']

results_sets = ['major_flow',
               'res_storage',
               'reservoir_downstream_gage',
               "lower_basin_mrf_contributions",
               'ibt_diversions',
               'catchment_consumption',
               'ffmp_level_boundaries',
               'inflow', 
               'nyc_release_components',
               'res_release',
               "all_mrf"]

model_label = 'drb_output_obs_pub_nhmv10_BC_ObsScaled_ensemble'
model = model_label

output_files = [
    rf"{output_dir}{model_label}.hdf5",
]


data = Data(print_status=True, 
            input_dir=input_dir,)


if reload_exported_data:
    data.load_from_export(data_export_fname)
else:
    data.load(datatypes=datatypes, 
          output_filenames=output_files, 
          results_sets=results_sets,
          input_dir=input_dir,
          )

    print("Calculating NYCAgg storage for reservoir_downstream_gage")
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    obs_nycAgg = data.reservoir_downstream_gage['obs'][0].loc[:, nyc_reservoirs].sum(axis=1)
    data.reservoir_downstream_gage['obs'][0]['NYCAgg'] = obs_nycAgg

    realizations = list(data.major_flow[model].keys())
    for r in realizations:
        r_nycAgg = data.reservoir_downstream_gage[model][r].loc[:, nyc_reservoirs].sum(axis=1)
        data.reservoir_downstream_gage[model][r]['NYCAgg'] = r_nycAgg

    ### Manually add obs storage data for more recent period
    reservoir_list_nyc = ['cannonsville', 'pepacton', 'neversink']
    
    # get reservoir storage capacities
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    from pywrdrb.utils.directories import input_dir
    input_dir = input_dir.split("site-packages")[0]
    historic_nyc_storage_file = f"{input_dir}/site-packages/pywrdrb/input_data/historic_NYC/NYC_storage_daily_2000-2021.csv"
    historic_storage_part1 = pd.read_csv(historic_nyc_storage_file, 
                                   index_col=0, parse_dates=True)
    historic_storage_part1 = subset_timeseries(historic_storage_part1['Total'], start_date, end_date)
    historic_storage_part1 *= 100/capacities['combined']
    
    historic_storage_part2 = data.res_storage["obs"][0]
    historic_storage_part2["Total"] = historic_storage_part2.loc[:, reservoir_list_nyc].sum(axis=1)
    historic_storage_part2["Total"] *= 100/capacities['combined']
    
    scale_factor = historic_storage_part2.loc['2021-04-16', "Total"] / historic_storage_part1.loc['2021-04-16']
    historic_storage_part2["Total"] /= scale_factor
    
    historic_storage = pd.DataFrame(index=pd.date_range(start_date, "2024-12-31", freq='D'), 
                                    columns=["Total"])
    match_date = pd.to_datetime("2021-11-29")
    use_odrm_data_idx = historic_storage_part1.loc[:match_date].index
    use_usgs_data_idx = historic_storage_part2.loc[match_date:].index
    
    # drop duplicate index
    historic_storage_part1 = historic_storage_part1.loc[~historic_storage_part1.index.duplicated(keep='first')]
    historic_storage_part2 = historic_storage_part2.loc[~historic_storage_part2.index.duplicated(keep='first')]
    
    historic_storage.loc[use_odrm_data_idx, "Total"] = historic_storage_part1.loc[use_odrm_data_idx].values
    historic_storage.loc[use_usgs_data_idx, "Total"] = historic_storage_part2.loc[use_usgs_data_idx, "Total"].values

    # Add to data object
    data.res_storage['obs'][0]["Total"] = historic_storage["Total"]
    

print("Successfully loaded data")

# Export data to speedup future loads
if export_data and not reload_exported_data:
    data.export(data_export_fname)
    print(f"Exported data to: {data_export_fname}")


units = 'MCM'
from methods.utils.constants import cm_to_mg, mg_to_mcm
# convert all data from MG to MCM
if units == 'MCM':
    data_mc = data
    
    # loop through all results sets
    all_attrs = data_mc.__dict__.keys()
    for attr in all_attrs:
        if isinstance(getattr(data_mc, attr), dict):

            # loop through all datasets
            for key in getattr(data_mc, attr).keys():
                
                # if attr name is "ffmp_level_boundaries" then skip
                if attr == "ffmp_level_boundaries":
                    continue
                if isinstance(getattr(data_mc, attr)[key], pd.DataFrame):
                    getattr(data_mc, attr)[key] = getattr(data_mc, attr)[key] * mg_to_mcm
                elif isinstance(getattr(data_mc, attr)[key], dict):
                    realizations = list(getattr(data_mc, attr)[key].keys())
                    for r in realizations:
                        getattr(data_mc, attr)[key][r] = getattr(data_mc, attr)[key][r] * mg_to_mcm
    data = data_mc
    
    


########################################
### Plot
########################################

print("Plotting")

model_colors_historic_reconstruction[model] = model_colors_historic_reconstruction['obs_pub_nhmv10_BC_ObsScaled_ensemble']
model_colors_historic_reconstruction[f"{model}_median"] = model_colors_historic_reconstruction['obs_pub_nhmv10_BC_ObsScaled']
model_label_dict[model] = model_label_dict['obs_pub_nhmv10_BC_ObsScaled_ensemble']
model_label_dict[f"{model}_median"] = model_label_dict['obs_pub_nhmv10_BC_ObsScaled'] + " Median"


for datelabel , date_range in date_ranges.items():

    for node in ['delTrenton', 'delMontague']:
        plot_obs = True if datelabel in ["1980s_drought", "post_ffmp", "full"] else False
        
        print(f"Making flow contribution plot for {datelabel} at {node}")
        
        plot_NYC_release_components_combined(
            storages = data.res_storage,
            ffmp_level_boundaries = data.ffmp_level_boundaries[model][0],
            model = model,
            node = node,
            nyc_release_components = data.nyc_release_components,
            lower_basin_mrf_contributions = data.lower_basin_mrf_contributions,
            reservoir_releases = data.res_release,
            reservoir_downstream_gages = data.reservoir_downstream_gage,
            major_flows = data.major_flow,
            inflows = data.inflow,
            diversions = data.ibt_diversions,
            consumptions = data.catchment_consumption,
            all_mrf=data.all_mrf,
            colordict=model_colors_historic_reconstruction,
            start_date=date_range[0],
            end_date=date_range[1],
            use_log=False,
            plot_observed=plot_obs,
            plot_flow_target=True,
            fill_ffmp_levels=True,
            percentile_cmap=True,
            ensemble_fill_alpha=0.8,
            smoothing_window=7,
            q_lower_bound=0.05,
            q_upper_bound=0.95,
            fig_dir=fig_dir,
            fig_dpi=200,
            save_svg=True,
            units='MCM/d',
        )