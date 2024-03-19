
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from methods.utils import FIG_DIR
from methods.utils.constants import cms_to_mgd

# Dict of different gauge/HRU IDs for different datasets
scaling_site_matches = {'cannonsville':{'nhmv10_gauges': ['1556'],   # '1559' matches '0142400103' but is not used (see lower comment)
                                'nhmv10_hru': ['1562'],
                                'nwmv21_gauges': ['01423000'],  # '0142400103' should be incuded but does not begin till 1996
                                'nwmv21_hru': ['2613174'],
                                'obs_gauges': ['01423000']},     # '0142400103' should be incuded but does not begin till 1996
                        'neversink': {'nhmv10_gauges': ['1645'],
                                      'nhmv10_hru': ['1638'],
                                      'nwmv21_gauges': ['01435000'],
                                      'nwmv21_hru': ['4146742'],
                                      'obs_gauges': ['01435000']},
                        'pepacton': {'nhmv10_gauges': ['1440', '1441', '1437'],  # '1443' matches '01414000' but is not used (see lower comment)
                                        'nhmv10_hru': ['1449'],
                                        'nwmv21_gauges': ['01415000', '01414500', '01413500'], # '01414000' should be incuded but does not begin till 1996
                                        'nwmv21_hru': ['1748473'],
                                        'obs_gauges': ['01415000', '01414500', '01413500']},  # '01414000' should be incuded but does not begin till 1996
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
nwmv21_scaled_reservoirs = list(scaling_site_matches.keys())


# Quarters to perform regression over
quarters = ('DJF','MAM','JJA','SON')


# Function for compiling flow data for regression
def prep_inflow_scaling_data():
    """
    Prepares the data for the inflow scaling regression:
    - Loads observed, NHM, and NWM inflows for different reservoirs
    - Aggregates inflows upstream of each reservoir
    - Combines inflows for each reservoir and all datasets into a single dataframe

    Returns:
        pd.DataFrame: Dataframe with inflows for each reservoir and dataset.
    """

    # Load observed, NHM, and NWM flow
    ## USGS
    obs_flows = pd.read_csv(f'{OUTPUT_DIR}/USGS/streamflow_daily_usgs_1950_2022_cms.csv', 
                            index_col=0, parse_dates=True)*cms_to_mgd
    usgs_gauge_ids = [c.split('-')[1] for c in obs_flows.columns]
    obs_flows.columns = usgs_gauge_ids
    obs_flows.index = pd.to_datetime(obs_flows.index.date)

    # Metadata: USGS site number, longitude, latitude, comid, etc.
    unmanaged_gauge_meta = pd.read_csv(f'{OUTPUT_DIR}/USGS/drb_unmanaged_usgs_metadata.csv', sep = ',', 
                                    dtype = {'site_no':str})
    unmanaged_gauge_meta.set_index('site_no', inplace=True)

    ## NHM
    # Streamflow
    nhmv10_flows = pd.read_csv(f'{PYWDRB_DIR}/input_data/modeled_gages/streamflow_daily_nhmv10_mgd.csv', 
                            index_col=0, parse_dates=True)
    nhmv10_flows = nhmv10_flows.loc['1983-10-01':, :]


    ## NWMv2.1
    # modeled gauge flows
    nwm_gauge_flows = pd.read_csv(f'{OUTPUT_DIR}/NWMv21/nwmv21_unmanaged_gauge_streamflow_daily_mgd.csv', 
                                        sep = ',', index_col=0, parse_dates=True)
    nwm_gauge_flows= nwm_gauge_flows.loc['1983-10-01':, :]


    # Metadata
    nwm_gauge_meta = pd.read_csv(f'{OUTPUT_DIR}/NWMv21/nwmv21_unmanaged_gauge_metadata.csv', 
                                        sep = ',', 
                                        dtype={'site_no':str, 'comid':str})
    # Replace nwm reachcodes with gauge ids
    for reachcode in nwm_gauge_flows.columns:
        if reachcode in nwm_gauge_meta['comid'].values:
            site_no = nwm_gauge_meta.loc[nwm_gauge_meta['comid'] == reachcode, 'site_no'].values[0]
            nwm_gauge_flows.rename(columns={reachcode:site_no}, 
                                   inplace=True)

    # modeled lake inflows and segment flows
    nwm_lake_inflows = pd.read_csv(f'{PYWDRB_DIR}/input_data/modeled_gages/streamflow_daily_nwmv21_mgd.csv', 
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
    """
    Trains multiple linear regression models used to predict inflow scaling coefficients.
    A unique model for each season for the specified reservoir.

    Args:
        reservoir (str): Name of the reservoir.
        inflows (pd.DataFrame): pd.DataFrame with inflows for the reservoir and dataset.

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
    """
    Predicts the inflow scaling coefficient using a specific regression model.
    
    Args:
        lrr (statsmodels.regression.linear_model.RegressionResultsWrapper): Regression model.
        log_flow (pd.Series): Log of the inflow.
    """

    X = sm.add_constant(log_flow)
    scaling = lrr.predict(X)
        
    scaling[scaling<1] = 1
    scaling = pd.DataFrame(scaling, index = log_flow.index, 
                           columns =['scale'])
    return scaling



def generate_scaled_inflows(start_date, end_date, 
                            scaling_rolling_window=3, 
                            donor_model='nhmv10', 
                            export=True):
    """
    Goes through the process of generating scaled inflows for all reservoirs using a specific
    dataset (donor_model) to estimate the scaling relationship.
    
    Args:
        start_date (str): Start date of the prediction period.
        end_date (str): End date of the prediction period.
        scaling_rolling_window (int): Number of days to use for rolling mean inflow.
        donor_model (str): Dataset to use for estimating the scaling relationship.
        export (bool): Whether to export the scaled inflows to a csv file.
    Returns:
        pd.DataFrame: Scaled inflows for all reservoirs.
    """

    # Load historic USGS obs
    Q_obs = pd.read_csv(f'{OUTPUT_DIR}USGS/streamflow_daily_usgs_1950_2022_cms.csv',
                                    index_col=0, parse_dates=True)*cms_to_mgd
    if '-' in Q_obs.columns[0]:
        usgs_gauge_ids = [c.split('-')[1] for c in Q_obs.columns]
        Q_obs.columns = usgs_gauge_ids
    Q_obs.index = pd.to_datetime(Q_obs.index.date)
    Q_obs = Q_obs.loc[start_date:end_date, :]
    Q_obs_scaled = Q_obs.copy()
    
    # Train models
    scaled_reservoirs = nhmv10_scaled_reservoirs if donor_model == 'nhmv10' else nwmv21_scaled_reservoirs
    linear_models = {}
    linear_results = {}
    for reservoir in scaled_reservoirs:
        scaling_training_flows = prep_inflow_scaling_data()
        linear_models[reservoir], linear_results[reservoir] = train_inflow_scale_regression_models(reservoir,
                                                                                                scaling_training_flows,
                                                                                                dataset=donor_model,
                                                                                                rolling=True,
                                                                                                window=scaling_rolling_window)
            
    for reservoir in scaled_reservoirs:
        inflow_gauges = scaling_site_matches[reservoir][f'obs_gauges']
        unscaled_inflows = Q_obs.loc[:, inflow_gauges].sum(axis=1)
        
        # Use linear regression to find inflow scaling coefficient
        # Different models are used for each quarter; done by month batches
        for m in range(1,13):
            quarter = get_quarter(m)
            rolling_unscaled_inflows = unscaled_inflows.rolling(window=scaling_rolling_window,
                                                                min_periods=1).mean()
            rolling_unscaled_month_inflows = rolling_unscaled_inflows.loc[unscaled_inflows.index.month == m]
            rolling_unscaled_month_log_inflows = np.log(rolling_unscaled_month_inflows.astype('float64'))
            
            month_scaling_coefs = predict_inflow_scaling(linear_results[reservoir][quarter], 
                                                        log_flow= rolling_unscaled_month_log_inflows)
                            
            ## Apply scaling across gauges for full month batch 
            # Match column names to map to df
            for site in inflow_gauges:
                month_scaling_coefs[site] = month_scaling_coefs.loc[:,'scale']
                # Multiply
                Q_obs_scaled.loc[Q_obs.index.month==m, site] = Q_obs.loc[Q_obs.index.month==m, site] * month_scaling_coefs[site]
    
        Q_obs_scaled.loc[:, reservoir] = Q_obs_scaled.loc[:, inflow_gauges].sum(axis=1)

    Q_obs_scaled = Q_obs_scaled.loc[start_date:end_date, scaled_reservoirs]    
    # Export
    if export:
        Q_obs_scaled.to_csv(f'{OUTPUT_DIR}/Hybrid/scaled_inflows_{donor_model}.csv', sep=',')
        Q_obs_scaled.to_csv(f'{PYWDRB_DIR}/input_data/scaled_inflows/scaled_inflows_{donor_model}.csv', sep=',')
        return Q_obs_scaled
    else:
        return Q_obs_scaled 



def plot_inflow_scaling_regression(donor_model = 'nhmv10', 
                                   roll_window = 3):
    """
    Creates a plot with all inflow scaling regressions for a specific dataset.
    
    Args:
        donor_model (str): Dataset to use for estimating the scaling relationship.
        roll_window (int): Number of days to use for rolling mean inflow.
    Returns:
        None
    """
    
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
                       alpha = 0.1, s=10)
            ax.plot(log_flow, lrr.predict(), c='k', 
                    linestyle='-', lw=1, zorder=4)
            
            # Annotate R-squared value
            # Get p value of regression model
            ax.annotate(f"R^2 = {lrr.rsquared:.2f}\np-val = {lrr.pvalues[1]:.4f}", 
                        xy=(0.1, 0.7), xycoords='axes fraction', fontsize=12)
        
            # Annotate regression equation
            ax.annotate(f"y = {lrr.params[1]:.2f}x + {lrr.params[0]:.2f}", 
                        xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12)
        
            # Setting labels and title for the subplot
            if i == 0:
                ax.set_title(f"{quarter}")
            if j == 0:
                ax.set_ylabel(f"{reservoir.capitalize()}\nScaling Coefficient")
            if i == n_rows - 1:
                ax.set_xlabel("Log Flow (MGD)")
        i += 1
    
    plt.suptitle((f'Inflow Scaling Coefficient Regressions\n'+
                  f'{donor_model.upper()} used to estimate scaling coefficient dependent on {roll_window} day rolling mean log-flow'),
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}inflow_scaling_regression_{donor_model}_rolling{roll_window}.png', dpi=300)
    plt.close()
    return

if __name__ == '__main__':

    for rolling_mean_window in [1, 3, 5, 7]:    
        export_scaled_inflows = True if rolling_mean_window == 3 else False
        generate_scaled_inflows(start_date='1983-10-01', end_date='2020-12-31', 
                                scaling_rolling_window=rolling_mean_window, 
                                donor_model='nhmv10',
                                export=export_scaled_inflows)
        plot_inflow_scaling_regression(donor_model = 'nhmv10', roll_window = rolling_mean_window)


        generate_scaled_inflows(start_date='1983-10-01', end_date='2020-12-31', 
                                scaling_rolling_window=rolling_mean_window, 
                                donor_model='nwmv21',
                                export=export_scaled_inflows)
        plot_inflow_scaling_regression(donor_model = 'nwmv21', roll_window = rolling_mean_window)