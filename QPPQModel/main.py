"""
Trevor Amestoy
Cornell University
Fall 2022

The function IDW_generator takes:
- A set of observed flow timeseries
- FDC at a target site
- The locations (lat, long) of the observed sites
- The location (lat, long) of the target site


And performs an inverse-distance weighted prediction of flow timeseries at the
target site based on flow at `K` surrounding sites.

"""
import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Custom functions
from QPPQModel.utils.QPPQ_utils import interpolate_FDC, find_continuous_FDC, find_NEPs
from QPPQModel.utils.QPPQ_utils import l2distance

################################################################################

class StreamflowGenerator():
    def __init__(self, K, observed_flow, observation_locations, **kwargs):
        
        ## Handle args
        assert(type(observed_flow) == pd.DataFrame), 'Observed flows (observed_flow) but be a pd.DataFrame.'
        assert(type(observed_flow.index) == pd.DatetimeIndex), 'Observed flows (observed_flow) but be a pd.DataFrame with datetime as index.'
        assert(sum(observed_flow.columns == observation_locations.index) == len(observed_flow.columns)), 'Dataframes observation_locations and observed_flow must have the same column names.'
        
        # Assign values
        self.Qobs = observed_flow
        self.observation_locations = observation_locations
        self.K = K
        
        ## Handle **kwargs
        # Check if an invalid kwarg was provided
        self.valid_kwargs = ['square_IDW', 'return_all', 'probabalistic_sample',
                             'log_fdc_interpolation', 
                             'fdc_quantiles', 'remove_zero_flow']
        for kwarg in kwargs.keys():
            if kwarg not in self.valid_kwargs:
                raise Exception(f'{kwarg} is not a valid key. Valid keys: {self.valid_kwargs}')

        # Substitute default values
        default_quantiles = np.linspace(0.00001, 0.99999, 200)
        
        self.square_IDW = kwargs.get('square_IDW', True)
        self.log_fdc_interpolation = kwargs.get('log_fdc_interpolation', False)
        self.return_all = kwargs.get('return_all', False)
        self.qs = kwargs.get('fdc_quantiles', default_quantiles)
        self.probabalistic_sample = kwargs.get('probabalistic_sample', False)
        self.remove_zero_flow_sites = kwargs.get('remove_zero_flow', True)
        

    def get_KNN(self):
        """Finds the KNN relative to the prediciton point."""
        distances = np.zeros((self.N_OBSERVATIONS))
        for i, site_id in enumerate(self.observation_locations.index):
            distances[i] = geodesic(self.prediction_location, self.observation_locations.loc[site_id, ['long', 'lat']].values).kilometers

        self.KNN_dists = np.sort(distances, axis = 0)[0:self.K].flatten()
        self.KNN_indices = np.argsort(distances, axis = 0)[0:self.K].flatten()
        self.KNN_site_ids = self.observation_locations.index[self.KNN_indices]

    def streamflow_to_nonexceedance(self, Q):
        # Get historic FDC
        if self.log_fdc_interpolation:
            fdc = np.quantile(np.log(Q), self.qs)
            Q = np.log(Q)
        else:
            fdc = np.quantile(Q, self.qs)
    
        # Translate flow to NEP
        nep = np.interp(Q, fdc, self.qs, right = self.qs[-1] + 0.0000001, left = self.qs[0]-0.000001)
        return nep

    def nonexceedance_to_streamflow(self, nep_timeseries):
        
        # The bound_percentage will determine how much (+/- %) random flow 
        # is sampled when NEP > fdc_quantiles[-1] or NEP < fdc_quantiles[0] 
        bound_percentage = 0.01
        high_flow_bound = np.random.uniform(self.predicted_fdc[-1], 
                                            self.predicted_fdc[-1] + bound_percentage*self.predicted_fdc[-1])
        
        low_flow_bound = np.random.uniform(self.predicted_fdc[0] - bound_percentage*self.predicted_fdc[0], 
                                           self.predicted_fdc[-0])
        Q = np.interp(nep_timeseries, self.qs, self.predicted_fdc, right = high_flow_bound, left = low_flow_bound)
        return Q


    def predict_streamflow(self, *args, **kwargs):
        """
        Run the QPPQ prediction method for a single locations.

        Parameters:
        ----------
        prediction_inputs : ndarray
            The matrix of feature values for the location of interest.
        predicted_fdc : ndarray
            An array of discrete FDC values.

        Returns:
        --------
        predicted_flow : ndarray
            A timeseries of predicted streamflow at the location.
        """

        ### Handle prediction specific inputs
        self.prediction_location, self.predicted_fdc = args

        # Change FDC to log
        if self.log_fdc_interpolation:
            self.predicted_fdc = np.log(self.predicted_fdc)
            
        prediction_kwargs = ['start_date', 'end_date']        
        for k in kwargs.keys():
            if k not in prediction_kwargs:
                if k in self.valid_kwargs:
                    print(f'{k} must be provided upon initialization of the model.')
                else:
                    print(f'{k} is not a valid kwarg for predict_streamflow(). Valid kwargs: {prediction_kwargs}.')

        start_date = kwargs.get('start_date', self.Qobs.index[0])
        end_date = kwargs.get('end_date', self.Qobs.index[-1])
        
        # Remove locations with missing data in the period
        self.Qobs = self.Qobs.loc[start_date:end_date,:]
        self.Qobs = self.Qobs.dropna(axis=1)
        self.observation_locations = self.observation_locations.loc[self.Qobs.columns, :]

        # Store constants
        self.T = self.Qobs.shape[0]  # Time duration
        self.N_OBSERVATIONS = self.Qobs.shape[1]  # Number of observed gages

        ### Find KNN
        self.get_KNN()

        ### Calculate weights
        buffer = 0.05    # Needed to avoid Nans when location is a gauge 
        if self.square_IDW:
            self.wts = 1/(self.KNN_dists + buffer)**2
        else:
            self.wts = 1/(self.KNN_dists + buffer)

        self.norm_wts = self.wts/np.sum(self.wts)

        ### Make predictions
        if self.probabalistic_sample:
            
            # Probabilitically sample one of the K locations
            sample_site_id = np.random.choice(self.KNN_site_ids, p=self.norm_wts)
            
            # Find the NEP timeseries from the sampled site
            self.observed_nep_timeseries = self.streamflow_to_nonexceedance(self.Qobs.loc[:, sample_site_id].values)
            
            # Convert NEP to flow timeseries
            self.predicted_flow = self.nonexceedance_to_streamflow(self.observed_nep_timeseries)
            
        else:
            # Create timeseries of NEPs
            self.observed_nep_timeseries = np.zeros((self.T, self.K))
            
            for i, site_id in enumerate(self.KNN_site_ids):
                self.observed_nep_timeseries[:, i] = self.streamflow_to_nonexceedance(self.Qobs.loc[:, site_id].values)

            # Calculated predicted NEP using IDW weights
            self.weighted_nep_timeseries = np.multiply(self.observed_nep_timeseries, self.norm_wts.flatten())
            self.predicted_nep_timeseries = np.sum(self.weighted_nep_timeseries, axis=1)

            ## Convert from NEP to flow
            self.predicted_flow = self.nonexceedance_to_streamflow(self.predicted_nep_timeseries)        
        
        # Convert log-flow to flow if needed
        if self.log_fdc_interpolation:
            self.predicted_flow = np.exp(self.predicted_flow)
        
        self.predicted_streamflow = pd.DataFrame(self.predicted_flow, index = self.Qobs.index)
        return self.predicted_streamflow
