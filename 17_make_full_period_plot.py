import sys
sys.path.append("./Pywr-DRB/src/")

import matplotlib.pyplot as plt
import pywrdrb
from pywrdrb import Data
import pandas as pd
import numpy as np

output_dir = './pywrdrb_outputs/'
input_dir = './pywrdrb_inputs/'
pywrdrb.set_directory(input_dir = input_dir)



datatypes = ['output']

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


reload_exported_data = True
data_export_fname = f"{output_dir}/pywrdrb_results_for_figures.hdf5"

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


nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
lower_basin_reservoirs = ['blueMarsh', 'beltzvilleCombined', 'neversink']
start = '1945-10-01'
end = '2022-12-31'
dt = pd.date_range(start, end, freq='D')

realizations = list(data.res_storage[model_label].keys())

nyc_data_arr = np.zeros((len(realizations), len(dt)))
lb_data_arr = np.zeros((len(realizations), len(dt)))

for sid in realizations:
    nyc_data_arr[sid, :] = data.res_storage[model_label][sid].loc[start:end, 
                                                nyc_reservoirs].sum(axis=1)

    lb_data_arr[sid, :] = data.res_storage[model_label][sid].loc[start:end, 
                                                lower_basin_reservoirs].sum(axis=1)
    
    
### NYC STORAGE
fig, ax = plt.subplots(figsize=(10,5))
ax.fill_between(dt, nyc_data_arr.min(axis=0), nyc_data_arr.max(axis=0), alpha=0.5,
                color='grey', label='Bias-Corrected Reconstruction Ensemble Range')
ax.plot(dt, nyc_data_arr.mean(axis=0), lw=0.5, 
        color='black', label='Bias-Corrected Reconstruction Ensemble Mean')
ax.set_ylabel('Total NYC Storage')
ax.set_xlabel('Date')
ax.set_xlim([dt[0], dt[-1]])
ax.set_ylim([0.0, nyc_data_arr.max()])
plt.tight_layout()
plt.legend()
plt.savefig('nyc_storage_full_simulation.png', dpi=300)
plt.show()


### LB STORAGE
fig, ax = plt.subplots(figsize=(10,5))
ax.fill_between(dt, lb_data_arr.min(axis=0), lb_data_arr.max(axis=0), alpha=0.5,
                color='grey', label='Bias-Corrected Reconstruction Ensemble Range')
ax.plot(dt, lb_data_arr.mean(axis=0), lw=0.5,
        color='black', label='Bias-Corrected Reconstruction Ensemble Mean')
ax.set_ylabel('Combined Blue Marsh, Beltzville, &\nNockamixon Storage')
ax.set_xlabel('Date')
ax.set_xlim([dt[0], dt[-1]])
ax.set_ylim([0.0, lb_data_arr.max()])
plt.tight_layout()
plt.legend()
plt.savefig('lb_storage_full_simulation.png', dpi=300)
plt.show()