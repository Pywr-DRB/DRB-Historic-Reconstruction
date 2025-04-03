from methods.processing.transform import transform_results_dict_flow
from methods.processing.load import load_historic_datasets
from methods.diagnostics.drought import calculate_ssi_values
from methods.plotting.drought_plots import plot_drbc_droughts_and_ssi

from config import DATES, FIG_DIR
start_date = DATES[0]
end_date = DATES[1]

if __name__ == '__main__':
    pub_model = 'obs_pub_nhmv10_BC_ObsScaled_ensemble'
    model_names = [
        'obs', 
        pub_model
        ]

    Q = load_historic_datasets(models=model_names,
                            flowtype='gage_flow',
                            start_date='1945-01-01',
                            end_date='2022-12-31')

    realization_numbers = list(Q[pub_model].keys())
    compare_ssi_windows = [12]

    print('Calculating SSI values')
    Q_monthly_rolling = {}
    ssi_values = {}
    
    # copy delDRCanal to delTrenton (they are coincident in the model)
    for realization in realization_numbers:
        Q[pub_model][realization].loc[:, 'delTrenton'] = Q[pub_model][realization].loc[:, 'delDRCanal']
    
    Q_monthly = transform_results_dict_flow(Q.copy(), 
                                            transform = 'aggregation',  
                                            window=1,
                                            aggregation_type = 'sum', 
                                            aggregation_length='MS')

    
    for ssi_window in compare_ssi_windows:
        
        ssi_values[ssi_window] = calculate_ssi_values(Q_monthly, 
                                                      window=ssi_window, 
                                                      nodes=['delTrenton'])

    print('Making SSI drought event plots')
    for ssi_window in compare_ssi_windows:
        ssi = ssi_values[ssi_window]
        for plot_obs in [True, False]:
            
            
            fname = f"{FIG_DIR}/droughts/ssi_{ssi_window}_delTrenton.png"
            plot_drbc_droughts_and_ssi(ssi, 
                                    ['obs_pub_nhmv10_BC_ObsScaled_ensemble'], 
                                    'delTrenton', 
                                    '1946-01-01', '2021-12-31',
                                    plot_observed=plot_obs,
                                    percentiles_cmap=True,
                                    fname=fname)