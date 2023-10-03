import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from methods.utils import data_dir, output_dir, pywrdrb_dir, ROOT_DIR, fig_dir
from methods.utils.constants import cms_to_mgd

path_to_nhm_data = f'{ROOT_DIR}/NHM-Data-Retrieval/outputs/hdf/'

# Dict of different gauge/HRU IDs for different datasets
scaling_site_matches = {'cannonsville':{'nhmv10_gauges': ['1556', '1559'],
                                'nhmv10_hru': ['1562'],
                                'nwmv21_gauges': ['01423000', '0142400103'],
                                'nwmv21_hru': ['2613174'],
                                'obs_gauges': ['01423000', '0142400103']},
                        'neversink': {'nhmv10_gauges': ['1645'],
                                      'nhmv10_hru': ['1638'],
                                      'nwmv21_gauges': [],
                                      'nwmv21_hru': [],
                                      'obs_gauges': ['01435000']},
                        'pepacton': {'nhmv10_gauges': ['1440', '1441', '1443', '1437'],
                                        'nhmv10_hru': ['1449'],
                                        'nwmv21_gauges': ['01415000', '01414500', '01414000', '01413500'],
                                        'nwmv21_hru': ['1748473'],
                                        'obs_gauges': ['01415000', '01414500', '01414000', '01413500']},
                        'fewalter': {'nhmv10_gauges': ['1684', '1691'],
                                        'nhmv10_hru': ['1684', '1691', '1694'],
                                        'nwmv21_gauges': ['01447720', '01447500'],
                                        'nwmv21_hru': ['4185065'],
                                        'obs_gauges': ['01447720', '01447500']},
                        'beltzvilleCombined': {'nhmv10_gauges': ['1703'],
                                        'nhmv10_hru': ['1710'],
                                        'nwmv21_gauges': ['01449360'],
                                        'nwmv21_hru': ['4186689'],
                                        'obs_gauges': ['01449360']}}

# List of all reservoirs able to be scaled
nhmv10_scaled_reservoirs = list(scaling_site_matches.keys())
nwmv21_scaled_reservoirs = ['cannonsville', 'pepacton', 'fewalter', 'beltzvilleCombined']

# Quarters to perform regression over
quarters = ('DJF','MAM','JJA','SON')





# Function for compiling flow data for regression
def prep_inflow_scaling_data():

    # Load observed, NHM, and NWM flow
    ## USGS
    obs_flows = pd.read_csv(f'{data_dir}historic_unmanaged_streamflow_1900_2022_cms.csv', 
                            index_col=0, parse_dates=True)*cms_to_mgd
    usgs_gauge_ids = [c.split('-')[1] for c in obs_flows.columns]
    obs_flows.columns = usgs_gauge_ids
    obs_flows.index = pd.to_datetime(obs_flows.index.date)

    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv('./data/drb_unmanaged_usgs_metadata.csv', sep = ',', 
                                    dtype = {'site_no':str})
    unmanaged_gauge_meta.set_index('site_no', inplace=True)


    ## NHM
    # Streamflow
    nhmv10_flows = pd.read_csv(f'{pywrdrb_dir}/input_data/modeled_gages/streamflow_daily_nhmv10_mgd.csv', 
                            index_col=0, parse_dates=True)
    nhmv10_flows = nhmv10_flows.loc['1983-10-01':, :]


    ## NWMv2.1
    # modeled gauge flows
    nwm_gauge_flows = pd.read_csv(f'{data_dir}/NWMv21/nwmv21_unmanaged_gauge_streamflow_daily_mgd.csv', 
                                        sep = ',', index_col=0, parse_dates=True)
    nwm_gauge_flows= nwm_gauge_flows.loc['1983-10-01':, :]

    # Metadata
    nwm_gauge_meta = pd.read_csv(f'{data_dir}/NWMv21/nwmv21_unmanaged_gauge_metadata.csv', 
                                        sep = ',', 
                                        dtype={'site_no':str, 'comid':str})
    # Replace nwm reachcodes with gauge ids
    for reachcode in nwm_gauge_flows.columns:
        if reachcode in nwm_gauge_meta['comid'].values:
            site_no = nwm_gauge_meta.loc[nwm_gauge_meta['comid'] == reachcode, 'site_no'].values[0]
            nwm_gauge_flows.rename(columns={reachcode:site_no}, 
                                   inplace=True)

    # modeled lake inflows and segment flows
    nwm_lake_inflows = pd.read_csv(f'{pywrdrb_dir}/input_data/modeled_gages/streamflow_daily_nwmv21_mgd.csv', 
                                        index_col=0, parse_dates=True)
    nwm_lake_inflows = nwm_lake_inflows.loc['1983-10-01':, :]
    
    # Combine NWM data
    nwmv21_flows = pd.concat([nwm_gauge_flows, nwm_lake_inflows], axis=1)


    # Compile data for each reservoir in df
    data = pd.DataFrame()
    for node, flowtype_ids in scaling_site_matches.items():
        for flowtype, ids in flowtype_ids.items():
            if 'nhm' in flowtype:            
                data[f'{node}_{flowtype}'] = nhmv10_flows[ids].sum(axis=1)
            elif 'obs' in flowtype:
                data[f'{node}_{flowtype}'] = obs_flows[ids].sum(axis=1)
            elif 'nwm' in flowtype:
                data[f'{node}_{flowtype}'] = nwmv21_flows[ids].sum(axis=1)
                
    return data



def get_quarter(m):
    """Return a string indicator for the quarter of the month.

    Args:
        m (int): The numerical month.

    Returns:
        str: Either 'DJF', 'MAM', 'JJA', or 'SOR'.
    """
    if m in (12,1,2):
        return 'DJF'
    elif m in (3,4,5):
        return 'MAM'
    elif m in (6,7,8):
        return 'JJA'
    elif m in (9,10,11):
        return 'SON'

def train_inflow_scale_regression_models(reservoir, inflows, 
                                         dataset='nhmv10',
                                         rolling = True, 
                                         window =3):
    """_summary_

    Args:
        reservoir (_type_): _description_
        inflows (_type_): _description_

    Returns:
        (dict, dict): Tuple with OLS model, and fit model
    """
    dataset_opts = ['nhmv10', 'nwmv21']
    assert(dataset in dataset_opts), f'Specified dataset invalid. Options: {dataset_opts}'
    
    # Rolling mean flows
    if rolling:
        inflows = inflows.rolling(f'{window}D').mean()
        inflows = inflows[window:-window]
        inflows = inflows.dropna()
    
    inflows.loc[:,['month']] = inflows.index.month.values
    inflows.loc[:, ['quarter']] = [get_quarter(m) for m in inflows['month']]
    inflows.loc[:, [f'{reservoir}_{dataset}_scaling']] = inflows[f'{reservoir}_{dataset}_hru'] / inflows[f'{reservoir}_{dataset}_gauges']
    
    lrms = {q: sm.OLS(inflows[f'{reservoir}_{dataset}_scaling'].loc[inflows['quarter'] == q].values.flatten(),
                    sm.add_constant(np.log(inflows[f'{reservoir}_{dataset}_gauges'].loc[inflows['quarter'] == q].values.flatten()))) for q in quarters}

    lrrs = {q: lrms[q].fit() for q in quarters}
    return lrms, lrrs


def predict_inflow_scaling(lrr, log_flow):

    X = sm.add_constant(log_flow)
    scaling = lrr.predict(X)
        
    scaling[scaling<1] = 1
    scaling = pd.DataFrame(scaling, index = log_flow.index, 
                           columns =['scale'])
    return scaling



def plot_inflow_scaling_regression(donor_model = 'nhmv10', roll_window = 3):
    # Prepare the inflow scaling data
    inflow_data = prep_inflow_scaling_data()
    scatter_colors= {'DJF':'cornflowerblue', 'MAM':'darkgreen', 'JJA':'maroon', 'SON':'gold'}
    scaled_reservoirs = nhmv10_scaled_reservoirs if donor_model == 'nhmv10' else nwmv21_scaled_reservoirs
    n_rows = len(scaled_reservoirs)
    n_cols = len(quarters)
    
    # Initialize the plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    
    # Loop through each reservoir and train regression models
    for i, reservoir in enumerate(scaled_reservoirs):
        lrms, lrrs = train_inflow_scale_regression_models(reservoir, inflow_data,
                                                          dataset=donor_model,
                                                          window=roll_window)
        
        # Plotting for each quarter
        for j, quarter in enumerate(quarters):
            ax = axes[i, j]
            lrr = lrrs[quarter]
            
            # Extracting log_flow and scaling_coeff from the OLS result object
            log_flow = lrr.model.exog[:, 1]  # Exclude constant term
            scaling_coeff = lrr.model.endog
            
            ax.scatter(log_flow, scaling_coeff, c = scatter_colors[quarter], 
                       alpha = 0.2)
            ax.plot(log_flow, lrr.predict(), c='k', 
                    linestyle='--', lw=2, zorder=4)
            
            # Annotate R-squared value
            ax.annotate(f"R^2 = {lrr.rsquared:.2f}\np-val = {lrr.pvalues[1]:.4f}", 
                        xy=(0.1, 0.75), xycoords='axes fraction', fontsize=12)
        
            # Setting labels and title for the subplot
            if i == 0:
                ax.set_title(f"{quarter}")
            if j == 0:
                ax.set_ylabel(f"{reservoir}")
            if i == n_rows - 1:
                ax.set_xlabel("Log Flow")
        i += 1
    
    plt.suptitle(f'Inflow Scaling Regressions (Log Flow vs. Scaling Coefficient)\nUsing model {donor_model.upper()}', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}inflow_scaling_regression_{donor_model}_rolling_{roll_window}.png', dpi=300)
    plt.show()
    return
