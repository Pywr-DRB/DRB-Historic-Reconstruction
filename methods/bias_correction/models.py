

import os
os.environ["THEANO_FLAGS"] = "gcc__cxxflags=-C:\mingw-w64\mingw64\bin"

import pymc as pm


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



def bayesian_bias(x, y, 
                  n_tune=1000, 
                  n_sample=1000, 
                  target_accept=0.95,
                  beta_dist='normal',
                  alpha_dist='normal',
                  param_sigma_prior = 0.25,
                  n_cores=6):
    
    
    with pm.Model() as model:

        # Priors for alpha, beta, gamma
        if alpha_dist == 'normal':
            alpha = pm.Normal('alpha', mu=1.0, 
                            sigma=param_sigma_prior, 
                            shape=x.shape[1])
        elif alpha_dist == 'studentt':
            alpha = pm.StudentT('alpha', nu=2, mu=1.0, 
                            sigma=param_sigma_prior, 
                            shape=x.shape[1])
        if beta_dist == 'normal':
            # Prior for the first quantile
            beta = pm.Normal('b_0', mu=1, 
                            sigma=param_sigma_prior, 
                            shape=x.shape[1])
        elif beta_dist == 'studentt':
            beta = pm.StudentT('b_0', nu=2, mu=1, 
                            sigma=param_sigma_prior, 
                            shape=x.shape[1])
        # Model error        
        sigma_eps = pm.HalfNormal('sigma_eps', 
                                  sigma=1.0, shape=x.shape[1])

        x_data = pm.MutableData('x', x)
        
        # Expected value of outcome
        mu_Y = alpha + beta*x_data  # + beta_quantile * x_quantile


        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu_Y, 
                          sigma=sigma_eps, 
                          observed=y,
                          shape=x_data.shape)

        # MCMC sampling
        trace = pm.sample(n_sample, 
                          tune=n_tune, 
                          target_accept=target_accept,
                          cores=n_cores)

    return trace, model




def hierarchical_bayesian_bias(x, y, 
                        n_tune=1000, n_sample=1000, 
                        target_accept=0.95,
                        n_cores=6):
    
    with pm.Model() as model:
        
        # Hyperpriors
        mu_alpha = pm.Normal('mu_alpha', mu=2, sigma=1)
        sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=1)
        
        mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
        sigma_beta = pm.HalfCauchy('sigma_beta', beta=1)
                
        # Priors
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, 
                          shape=x.shape[1])
        beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta,
                        shape=x.shape[1])
                                   
        # Model error
        sigma_eps = pm.HalfCauchy('eps', beta=1)

        # X data
        x_data = pm.MutableData('x', x)
        
        # Expected value of outcome
        mu_Y = alpha + beta*x_data  # + beta_slope*x_mod_fdc_slope_data


        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu_Y, 
                          sigma=sigma_eps, 
                          observed=y,
                          shape=x_data.shape)

        # MCMC sampling
        trace = pm.sample(n_sample, 
                          tune=n_tune, 
                          target_accept=target_accept,
                          cores=n_cores)

    return trace, model




### OLD MODELS
def hierarchical_bayesian_regression(x_prcp, x_area, y_ObsCDF, N, M):
    with pm.Model() as model:
        # Priors for alpha, beta, gamma
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10, shape=(1, M))
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=10, shape=(1, M))
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=(1, M))

        mu_beta = pm.Normal('mu_beta', mu=0, sigma=10, shape=(1, M))
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=10, shape=(1, M))
        beta_prcp = pm.Normal('beta_prcp', mu=mu_beta, sigma=sigma_beta, shape=(1, M))

        mu_gamma = pm.Normal('mu_gamma', mu=0, sigma=10)
        sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=10)
        gamma_area = pm.Normal('gamma_area', mu=mu_gamma, sigma=sigma_gamma)

        # Model error
        sigma_eps = pm.HalfNormal('sigma_eps', sigma=10)

        # Expected value of outcome
        mu_Y = alpha + beta_prcp * x_prcp + gamma_area * x_area.reshape(-1,1)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu_Y, sigma=sigma_eps, observed=y_ObsCDF)

        # MCMC sampling
        trace = pm.sample(1000, tune=1000, target_accept=0.95)

    return trace, model

def bayesian_regression(x_df, y_df, 
                        N, M,
                        n_tune=1000, n_sample=1000, 
                        target_accept=0.95,
                        n_cores=6):
    
    with pm.Model() as model:
    
        x_prcp = x_df['x_prcp'].values
        x_area = x_df['x_area'].values
        x_ModFDC = x_df['x_model'].values
        x_ModFDC_prior = x_df['x_shifted'].values
        quantile = x_df['quantile'].values
        y_ObsCDF = y_df['y_ObsFDC'].values
        
        
        param_sigma = 1.0
        # Priors for alpha, beta, gamma
        alpha = pm.Normal('alpha', mu=0.0, 
                          sigma=param_sigma, shape=1)

        beta_prcp = pm.Normal('beta_prcp', 
                              mu=1.0, 
                              sigma=param_sigma, shape=1)
        
        beta_model_fdc = pm.Normal('beta_model_fdc',
                                    mu=1.0,
                                    sigma=param_sigma, shape=1)
        beta_model_fdc_prior = pm.Normal('beta_model_fdc_prior',
                                            mu=1.0,
                                            sigma=param_sigma, 
                                            shape=1)
        beta_quantile = pm.Normal('beta_quantile',
                                    mu=1.0,
                                    sigma=param_sigma, shape=1)
        gamma_area = pm.Normal('gamma_area', 
                               mu=1.0, 
                               sigma=param_sigma,
                               shape=1)

        # Model error
        sigma_eps = pm.HalfNormal('sigma_eps', 
                                  sigma=1.0)

        x_prcp_data = pm.MutableData('x_prcp', x_prcp)
        x_area_data = pm.MutableData('x_area', x_area)
        x_model_fdc_data = pm.MutableData('x_model_fdc', x_ModFDC)
        x_model_fdc_prior_data = pm.MutableData('x_model_fdc_prior', x_ModFDC_prior)        
        x_quantile = pm.MutableData('x_quantile', quantile)
        
        
        # Expected value of outcome
        mu_Y = alpha + beta_prcp * x_prcp_data + gamma_area * x_area_data + beta_model_fdc * x_model_fdc_data + beta_model_fdc_prior * x_model_fdc_prior_data + beta_quantile * x_quantile

        # Monotonicity constraint
        for i in range(1, N):
            
            pm.Potential(f'monotonicity_{i}', 
                         pm.math.switch(mu_Y[:, i] - mu_Y[:,i - 1] < 0, -99, 0.0))


        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', 
                          mu=mu_Y, 
                          sigma=sigma_eps, 
                          observed=y_ObsCDF,
                          shape=x_prcp_data.shape[0])

        # MCMC sampling
        trace = pm.sample(n_sample, 
                          tune=n_tune, 
                          target_accept=target_accept,
                          cores=n_cores)

    return trace, model
