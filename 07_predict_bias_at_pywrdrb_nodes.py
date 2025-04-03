import numpy as np
import pandas as pd
import sys

from config import BART_PREDICTION_QUANTILES, BART_REGRESSION_WEIGHTS
from config import BART_PARAMS, BART_USE_X_SCALED
from config import MCMC_N_TUNE, MCMC_N_SAMPLE
from config import BART_N_FEATURES
from config import SEED, FIG_DIR
from config import FEATURE_MUTUAL_INFORMATION_FILE

from methods.bias_correction.prep import load_bias_correction_inputs, load_bias_correction_training_outputs
from methods.bias_correction.utils import export_posterior_samples_to_hdf5
from methods.bias_correction.models import BARTLinearInterpolatedBias
from methods.utils.directories import OUTPUT_DIR




def predict_nhm_bias_at_pywrdrb_nodes(X_train, 
                                      Y_train,
                                      X_test, 
                                BART_PREDICTION_QUANTILES, 
                                BART_PARAMS, 
                                MCMC_N_TUNE, 
                                MCMC_N_SAMPLE,
                                y_scaled_outputs=False,
                                y_scaler=None,
                                plot_convergence=True,
                                bayesian_weights=False,
                                weights = BART_REGRESSION_WEIGHTS,
                                fig_dir=FIG_DIR,):
    
    if not bayesian_weights:
        assert weights is not None, 'Weights must be provided if not using Bayesian weights'
        
    

    bart_model = BARTLinearInterpolatedBias(X_train, Y_train, 
                                            bayesian_linear_regression=True, 
                                            predict_quantiles=BART_PREDICTION_QUANTILES, 
                                            seed=SEED,
                                            bayesian_weights=bayesian_weights, 
                                            weights=weights,
                                            test_size=0.0)

    # Manually set train & test data
    bart_model.x_train = X_train.values.astype(float)
    bart_model.x_test = X_test.values.astype(float)
    bart_model.y_train = Y_train.values.astype(float)
    bart_model.y_test = None
    bart_model.test_indices = None
    bart_model.train_indices = X_train.index
    bart_model._data_is_split = True

    bart_model.mcmc(n_tune=MCMC_N_TUNE, 
                    n_sample=MCMC_N_SAMPLE, 
                    params=BART_PARAMS)
    
    if plot_convergence:
        fig_fname= f'{fig_dir}/bias_correction/convergence/pywrdrb_convergence.png'
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
        
    return y_pred






if __name__ == '__main__':

    np.random.seed(SEED)

    ### Load
    # Training data (diagnostic usgs sites)
    x_train = load_bias_correction_inputs(scaled=BART_USE_X_SCALED, 
                                               pywrdrb=False)
    
    # y_train is NHM FDC percent bias at diagnostic sites    
    y_train = load_bias_correction_training_outputs(percent_bias=True)
    y_train = y_train.loc[x_train.index, :]
    assert(x_train.index.equals(y_train.index)), 'Index mismatch between inputs and outputs'


    # Test data (pywrdrb nodes)
    x_test = load_bias_correction_inputs(scaled=BART_USE_X_SCALED,
                                                  pywrdrb=True)
    x_test_indices = x_test.index.values
    
    # Load selected features
    selected_features = pd.read_csv(FEATURE_MUTUAL_INFORMATION_FILE, index_col=0)
    selected_features = selected_features.iloc[:BART_N_FEATURES, :].index.values
    selected_features = list(selected_features)


    x_train = x_train.loc[:, selected_features]
    x_test = x_test.loc[:, selected_features]

    bias_pred = predict_nhm_bias_at_pywrdrb_nodes(X_train=x_train, 
                                                  Y_train=y_train,
                                                  X_test=x_test,
                                                    BART_PREDICTION_QUANTILES=BART_PREDICTION_QUANTILES,
                                                    BART_PARAMS=BART_PARAMS,
                                                    MCMC_N_TUNE=MCMC_N_TUNE,
                                                    MCMC_N_SAMPLE=MCMC_N_SAMPLE,
                                                    y_scaled_outputs=False,
                                                    y_scaler=None,
                                                    bayesian_weights=False,
                                                    weights=BART_REGRESSION_WEIGHTS,
                                                    plot_convergence=True)

    # Save results in HDF5
    output_fname = f'{OUTPUT_DIR}nhmv10_pywrdrb_bias_posterior_samples.hdf5'
    export_posterior_samples_to_hdf5(data = bias_pred,
                                     site_numbers=x_test_indices,
                                     filename=output_fname)

    print(f'Saved posterior samples of bias prediction at pywrdrb nodes to {output_fname}')