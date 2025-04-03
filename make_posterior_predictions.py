import numpy as np
import pandas as pd
import sys
import logging

from methods.spatial.upstream import UpstreamGaugeManager
from methods.utils.directories import OUTPUT_DIR

from methods.bias_correction.prep import load_bias_correction_inputs, load_bias_correction_training_outputs
from methods.bias_correction.utils import export_posterior_samples_to_hdf5
from methods.bias_correction.models import BARTLinearInterpolatedBias


from config import BART_PREDICTION_QUANTILES, BART_REGRESSION_WEIGHTS
from config import BART_PARAMS, BART_USE_X_SCALED
from config import MCMC_N_TUNE, MCMC_N_SAMPLE, MCMC_TARGET_ACCEPT, MCMC_N_CHAINS
from config import BART_N_FEATURES
from config import SEED, FIG_DIR
from config import FEATURE_MUTUAL_INFORMATION_FILE
from config import STATION_UPSTREAM_GAGE_FILE


def get_site_posterior_predictions(X, Y, site, 
                                BART_PREDICTION_QUANTILES, 
                                BART_PARAMS, 
                                MCMC_N_TUNE, 
                                MCMC_N_SAMPLE,
                                station_upstream_gauges, 
                                y_scaled_outputs=False,
                                y_scaler=None,
                                plot_convergence=False,
                                bayesian_weights=False,
                                weights = BART_REGRESSION_WEIGHTS,
                                fig_dir=FIG_DIR,):
    
    if not bayesian_weights:
        assert weights is not None, 'Weights must be provided if not using Bayesian weights'
        
    
    # Drop any upstream gauges
    site_upstream_gauges = station_upstream_gauges.get(site, [])
    x_train = X.drop(X.index.intersection(site_upstream_gauges), axis=0)
    y_train = Y.drop(Y.index.intersection(site_upstream_gauges), axis=0)
    x_train = x_train.drop(site)
    y_train = y_train.drop(site)

    # Keep just the single site
    x_test = X.loc[[site], :]
    y_test = Y.loc[[site], :]
    x_test_indices = x_test.index

    # stack a copy of x_test and y_test on themselves 
    x_test = pd.concat([x_test, x_test], axis=0)
    y_test = pd.concat([y_test, y_test], axis=0)

    bart_model = BARTLinearInterpolatedBias(x_train, y_train, 
                                            bayesian_linear_regression=True, 
                                            predict_quantiles=BART_PREDICTION_QUANTILES, 
                                            seed=SEED,
                                            bayesian_weights=bayesian_weights, 
                                            weights=weights,
                                            test_size=0.0)

    # Manually set train & test data
    bart_model.x_train = x_train.values.astype(float)
    bart_model.x_test = x_test.values.astype(float)
    bart_model.y_train = y_train.values.astype(float)
    bart_model.y_test = y_test.values.astype(float)
    bart_model.test_indices = x_test_indices
    bart_model.train_indices = x_train.index
    bart_model._data_is_split = True

    bart_model.mcmc(n_tune=MCMC_N_TUNE, n_sample=MCMC_N_SAMPLE, params=BART_PARAMS)
    
    if plot_convergence:
        fig_fname= f'{fig_dir}/bias_correction/convergence/convergence_{site}.png'
        bart_model.plot(type='convergence', 
                        savefig=True,
                        fname=fig_fname)
    
    bart_model._in_test_mode = True
    
    y_pred_scaled = bart_model.predict(chain_mean=True)
    
    
    if y_scaled_outputs:
        y_pred = np.zeros_like(y_pred_scaled)
        n_samples = y_pred_scaled.shape[0]
        for i in range(n_samples):
            y_pred[i, :, :] = y_scaler.inverse_transform(y_pred_scaled[i, :, :])
    else:
        y_pred = y_pred_scaled
        
    return y_pred, x_test_indices





if __name__ == '__main__':

    np.random.seed(SEED)

    # Argument parsing
    site_file = sys.argv[1]
    rank = int(sys.argv[2])

    # Read the site indices from the file
    with open(site_file, 'r') as f:
        test_site_numbers = f.read().splitlines()
        
    test_site_numbers = list(test_site_numbers)
    print(f'RANK {rank}: Got test sites: {test_site_numbers}')
    try:
        len(test_site_numbers)
    except:
        print('No len() for test_site_numbers', test_site_numbers)


    # Load and preprocess training inputs
    x_train_full = load_bias_correction_inputs(scaled=BART_USE_X_SCALED, pywrdrb=False)
    y = load_bias_correction_training_outputs(percent_bias=True)
    y = y.loc[x_train_full.index, :]
    assert(x_train_full.index.equals(y.index)), 'Index mismatch between inputs and outputs'
    training_sites = x_train_full.index.values
    training_sites = list(training_sites)

    # Load selected features
    selected_features = pd.read_csv(FEATURE_MUTUAL_INFORMATION_FILE, index_col=0)
    selected_features = selected_features.iloc[:BART_N_FEATURES, :].index.values
    selected_features = list(selected_features)

    # Load station metadata and upstream gauges
    import json
    with open(STATION_UPSTREAM_GAGE_FILE, 'r') as f:
        station_upstream_gauges = json.load(f)


    rank_posterior_samples = np.zeros((MCMC_N_SAMPLE*BART_PARAMS['n_cores'], 
                                    len(test_site_numbers), 
                                    y.shape[1]))

    X = x_train_full.loc[:, selected_features]
    Y = y

    for si, site in enumerate(test_site_numbers):
        print(f'RANK {rank}: Site {si+1}/{len(test_site_numbers)}: {site}')
        y_pred, test_indices = get_site_posterior_predictions(X, Y, 
                                                              site=site, 
                                                            BART_PREDICTION_QUANTILES=BART_PREDICTION_QUANTILES,
                                                            BART_PARAMS=BART_PARAMS,
                                                            MCMC_N_TUNE=MCMC_N_TUNE,
                                                            MCMC_N_SAMPLE=MCMC_N_SAMPLE,
                                                            station_upstream_gauges=station_upstream_gauges,
                                                            y_scaled_outputs=False,
                                                            y_scaler=None,
                                                            bayesian_weights=False,
                                                            weights=BART_REGRESSION_WEIGHTS)
        rank_posterior_samples[:, si, :] = y_pred[:,0,:]


    # Save results in HDF5
    output_fname = f'{OUTPUT_DIR}LOO/loo_bias_correction/nhmv10_bias_posterior_samples_rank{rank}.hdf5'
    export_posterior_samples_to_hdf5(rank_posterior_samples, test_site_numbers, output_fname)

    print(f'RANK {rank}: Saved posterior samples to {output_fname}')
