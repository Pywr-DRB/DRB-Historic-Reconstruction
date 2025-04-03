import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from config import SEED 
from config import BART_REGRESSION_WEIGHTS, BART_PREDICTION_QUANTILES

from methods.bias_correction.plots import plot_observed_vs_predicted

import pymc as pm
import pymc_bart as pmb
import arviz as az
import pytensor as pt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from methods.utils.directories import FIG_DIR

import logging
logger = logging.getLogger("pymc")
logger.propagate = False

os.environ["PYTENSOR_FLAGS"] = "gcc__cxxflags='-I/opt/ohpc/pub/compiler/gcc/9.3.0/include'"


def generate_posterior_samples_for_test(model, trace, 
                                        x_dict,
                                        num_samples=1000):
    
    assert(type(x_dict)==dict),'x must be a dictionary with model data param_name:values.'

    # Update the model with test data
    with model:
        pm.set_data(x_dict)

        # Generate posterior predictive samples for the test data
        ppc_test = pm.sample_posterior_predictive(trace, predictions=True)

    # Return the posterior predictive samples for the test data
    return ppc_test


default_bart_params = {'m': 20,
                        'target_accept': 0.95,
                        'n_chains': 4,
                        'n_cores': 1,
                        'response': 'linear',
                        'beta':2.0,
                        'alpha':0.95,
                        'separate_trees': True,
                        'sigma_beta': 0.2,
                        'bayesian_weights' : False}


class BARTLinearInterpolatedBias():
    def __init__(self,
                 X, Y,
                 test_size=0.2,
                 predict_quantiles = BART_PREDICTION_QUANTILES,
                 bayesian_linear_regression=True,
                 seed=SEED,
                 DEBUGGING=True,
                 bayesian_weights=False,
                 weights=BART_REGRESSION_WEIGHTS) -> None:
        
        # make sure predict quantiles are in y columns
        assert(all([q in Y.columns for q in predict_quantiles])), 'predict_quantiles must be in Y columns.'
        
        self.X_cols = X.columns.tolist()
        self.Y_cols = Y.columns.tolist()
        self.X = X
        self.Y = Y
        
        self.predict_quantiles = predict_quantiles
        self.predict_column_indices = [self.Y_cols.index(q) for q in predict_quantiles]
        self.bayesian_linear_regression = bayesian_linear_regression
        self.bayesian_weights = bayesian_weights
        self.test_size = test_size
        self.seed = seed
        
        # Constants
        self.L = len(self.predict_quantiles)
        self.N, self.M = X.shape
        self.n_quantiles = len(self.Y_cols)

        self._data_is_split = False
        self._posterior_available = False
        self.DEBUGGING = DEBUGGING
        self._in_test_mode = False
        
        assert(not self.bayesian_weights), "Bayesian weights are no longer used."

        assert weights is not None, "weights must be provided if bayesian_weights=False"
        assert isinstance(weights, np.ndarray), "weights must be a numpy array"
            
        assert weights.shape == (self.L, self.n_quantiles), f"weights must have shape {(self.n_quantiles, self.L)}, got {weights.shape}"
        self.weights = weights
        return
    
    def split_test_train(self):
        if self.test_size==0.0:
            self.x_train = self.X.values.astype(float)
            self.y_train = self.Y.values.astype(float)
            self._data_is_split = True
            self.x_test = None
            self.y_test = None
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, 
                                                                test_size=self.test_size)
            self.x_train = x_train.values.astype(float)
            self.x_test = x_test.values.astype(float)
            self.y_train = y_train.values.astype(float)
            self.y_test = y_test.values.astype(float)
            self._data_is_split = True
            self.test_indices = x_test.index
            self.train_indices = x_train.index    
        return
    
    def get_normal_priors(self, param_name='alpha'):
        # for the function f(x_i)= alpha + beta*x_j
        # where x_j are BART predicted values and x_i are values are other quantiles
        # local regression around the BART predicted values
        
        
        assert(param_name != "weight"), "Do not use bayesian weights in this model."

        params_with_priors = ['alpha', 'beta']
        assert(param_name in params_with_priors), f'{param_name} must be in {params_with_priors}'
        
        mu_priors = np.zeros((self.n_quantiles, self.L))
        sigma_priors = np.ones((self.n_quantiles, self.L))
        
        for i in range(self.n_quantiles):
            qi=i
            for j, qj in enumerate(self.predict_column_indices):
                # if at BART predicted quantile
                # expect it to be equal to the BART prediction
                if qi==qj:
                    if param_name == 'alpha':                    
                        mu_priors[i, j] = 0.0 
                    elif param_name == 'beta':
                        mu_priors[i,j] = 1.0
                    sigma_priors[i, j] = 0.001


                else:
                    
                    # use only lower 90% of data for fitting
                    # avoid overfitting to extreme high bias
                    x_fit = self.y_train[:, qj]
                    y_fit = self.y_train[:, qi]
                    x_fit = x_fit[y_fit < np.quantile(y_fit, 0.9)]
                    y_fit = y_fit[y_fit < np.quantile(y_fit, 0.9)]
                    
                    # fit OLS
                    bi, ai = np.polyfit(x_fit, y_fit, 1)
                    rsqr = r2_score(y_fit, bi*x_fit + ai)
                    
                    # assign mu priors
                    if param_name=='beta':
                        if rsqr < 0.1:
                            mu_priors[i,j] = 0.0
                        else:
                            mu_priors[i,j] = bi
                            
                        sigma_priors[i, j] = max(1 - rsqr, 0.01)
                        
                        # limit beta mu to [0, 2]
                        mu_priors[i,j] = np.clip(mu_priors[i,j], 0.01, 2.0)
                        sigma_priors[i, j] = np.clip(sigma_priors[i, j], 0.01, 0.9)

                    elif param_name=='alpha':
                        mu_priors[i,j] = ai
                        sigma_priors[i, j] = np.std(y_fit)
                        
                        # limit alpha to [-100, 100]
                        mu_priors[i,j] = np.clip(mu_priors[i,j], -100.0, 100.0)
                        sigma_priors[i, j] = np.clip(sigma_priors[i, j], 0.01, 100.0)        

        return mu_priors, sigma_priors

    def mcmc(self, n_tune=1000, 
             n_sample=1000,
             params=default_bart_params,
             progressbar=False):
        
        self.params = params
        for k, v in default_bart_params.items():
            if k not in self.params:
                self.params[k] = v
        
        # split test/train
        if not self._data_is_split:
            self.split_test_train()
            
        # get priors
        self.beta_mu_priors, self.beta_sigma_priors = self.get_normal_priors(param_name='beta')
        self.alpha_mu_priors, self.alpha_sigma_priors = self.get_normal_priors(param_name='alpha')
        
        self.n_predictions = self.x_train.shape[0]
        
        X = self.x_train
        
        # limit prediction to range inner 99% range of data 
        y_min_bound = -95.0
        y_max_bound = 510.0 
        
        # clip the training data range
        self.y_train = np.clip(self.y_train, y_min_bound+0.5, y_max_bound-1)
        
        # define the model and sample
        with pm.Model() as model:       
            x = pm.Data('x', X)
            n_predictions, m_features = x.shape

            y = self.y_train.copy()

            Z = pmb.BART('Z', 
                         x, y[:, self.predict_column_indices], 
                         m=params['m'],
                         shape=(self.L, n_predictions),
                         separate_trees=params['separate_trees'], 
                         response=params['response'],
                         alpha=params['alpha'],
                         beta=params['beta'])            
            
            if self.bayesian_linear_regression:
                beta = pm.TruncatedNormal('beta', 
                                          mu=self.beta_mu_priors, 
                                          sigma=self.beta_sigma_priors, 
                                          shape=(self.n_quantiles, self.L),
                                          lower=0.0, upper=2.0)
                alpha = pm.TruncatedNormal('alpha',
                                           mu=self.alpha_mu_priors, 
                                           sigma=self.alpha_sigma_priors,
                                           shape=(self.n_quantiles, self.L),
                                           lower=-100.0, upper=100.0)
            else:
                beta = pt.as_symbolic(self.beta_mu_priors, name='beta')
                alpha = pt.as_symbolic(self.alpha_mu_priors, name='alpha')


            W = pt.as_symbolic(self.weights, name="W")

            W_expanded = W[np.newaxis, :, :]
            
            y_mu_unweighted = Z.T[:,:,np.newaxis]* beta.T  + alpha.T
            
            y_mu = pm.math.sum(y_mu_unweighted * W_expanded, axis=1)

            y_sigma = pm.HalfCauchy('y_sigma', beta=0.1, 
                    shape=(self.n_quantiles))
            
            y_hat = pm.TruncatedNormal('y_hat', 
                            mu=y_mu, 
                            sigma=y_sigma, 
                            observed=y,
                            shape=(n_predictions, self.n_quantiles),
                            lower=y_min_bound,
                            upper=y_max_bound)

                            
            # sample
            trace = pm.sample(n_sample, 
                              tune=n_tune, 
                              target_accept=params['target_accept'],
                              chains=params['n_chains'],
                              cores=params['n_cores'],
                              progressbar=progressbar,
                              random_seed=self.seed)
            # store trace
            self.trace = trace
            self.Z = Z
            self.y_hat = y_hat
        
        self.model = model
        return
    
    def sample_posterior_predictive(self):
        with self.model:
            posterior = pm.sample_posterior_predictive(self.trace,
                                                       predictions=False)
        self.posterior = posterior
        self.posterior_values = posterior['posterior_predictive']['y_hat'].values
        self._posterior_available = True
        return posterior
    
    
    def predict(self,
                X=None,
                chain_mean=True):
        
        self._in_test_mode = True
        
        # set test data as x
        with self.model:
            if X is not None:
                if type(X)==pd.DataFrame:
                    X = X.values.astype(float)
                X = np.atleast_2d(X)
                self.n_predictions = X.shape[0]
                X = X.reshape((self.n_predictions, self.M))
                print(f'Using new test data with shape {X.shape}')
                pm.set_data({'x': X})
            elif self.x_test is not None:
                print(f'Using x_test data with shape {self.x_test.shape}')
                self.n_predictions = self.x_test.shape[0]
                pm.set_data({'x': self.x_test})
            else:
                err_msg = 'No test data provided.' 
                err_msg += 'Set X argument with new inputs'
                err_msg += 'Or set test_size>0.0 to split training data during initialization.'
                raise ValueError(err_msg)

            predictions = pm.sample_posterior_predictive(self.trace,
                                                         predictions=True)
        print(f'Raw predictions shape: {predictions["predictions"]["y_hat"].shape}')
        
        if chain_mean:
            y_pred = predictions['predictions']['y_hat'].mean(dim=['chain']).values
        else:
            y_pred = predictions['predictions']['y_hat'].stack(samples=('chain', 'draw')).values
            print(f'Predictions shape after stacking chains: {y_pred.shape}')

            if len(y_pred.shape)==2:
                y_pred = y_pred[:, np.newaxis, :]
                
            y_pred = y_pred.transpose((2, 0, 1))
        self.y_pred = y_pred
        return y_pred
    
    def plot(self,
             type='convergence',
             variable_subset=['Z'],
             savefig=False, fname='.png'):
        
        type_opts = ['convergence', 'trace', 
                     'variable_importance',
                     'pcp', 'obs_pred'] #'pdp'
        
        assert(type in type_opts), f'Invalid type. Choose from {type_opts}'
        
        if type == 'convergence':
            axs = pmb.plot_convergence(self.trace, 'Z')
            
        elif type == 'trace':
            axs = az.plot_trace(self.trace, compact=True, 
                                var_names=variable_subset)

        elif type == 'variable_importance':
            axs = pmb.plot_variable_importance(self.trace, 
                                         self.Z, self.x_train)
            
        elif type == 'pdp':
            pass
        
        elif type == 'pcp':
            axs = az.plot_ppc(self.posterior,
                        kind='cumulative',
                        var_names=['y_hat'])
        elif type == 'obs_pred':
            axs = plot_observed_vs_predicted(self.y_test, self.y_pred)
            
            
        else:
            print(f'Invalid type specified in plot(). Options are {type_opts}')
            return
        
        if savefig:
            plt.tight_layout()
            plt.savefig(fname)
        return axs 

