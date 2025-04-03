import numpy as np
import pandas as pd
from geopy.distance import geodesic

class StreamflowGenerator:
    def __init__(self, 
                 K, 
                 observed_flow, 
                 observation_locations, 
                 **kwargs):
        """
        A class to generate streamflow predictions using variations of the QPPQ method.
        
        Parameters
        ----------
        K : int
            Number of nearest neighbors to use in the prediction.
        observed_flow : pd.DataFrame
            Observed flow timeseries at multiple sites with datetime index and site IDs as columns.
        observation_locations : pd.DataFrame
            Dataframe of observation locations with site IDs as index and columns ['long', 'lat'].
        **kwargs : dict
            Additional configuration options:
                - probabalistic_sample (bool): If True, sample from KNN sites based on distance.
                - log_fdc_interpolation (bool): If True, interpolate the FDC in log-space.
                - fdc_quantiles (array): Quantiles for discretizing the FDC. Defaults to np.linspace(0.00001, 0.99999, 200).
        """
        # Ensure correct types
        assert isinstance(observed_flow, pd.DataFrame), 'observed_flow must be a DataFrame.'
        assert isinstance(observed_flow.index, pd.DatetimeIndex), 'observed_flow must have a DatetimeIndex.'
        assert isinstance(observation_locations, pd.DataFrame), 'observation_locations must be a DataFrame.'
        
        # Clean columns to standardize site IDs
        observed_flow = self.clean_columns(observed_flow)
        
        
        # Assign values
        self.K = K
        self.observed_flow = observed_flow.copy()
        self.observation_locations = observation_locations.copy()
        
        
        # Handle kwargs
        self.log_fdc_interpolation = kwargs.get('log_fdc_interpolation', True)
        self.probabalistic_sample = kwargs.get('probabalistic_sample', False)
        self.qs = kwargs.get('fdc_quantiles', np.linspace(0.00001, 0.99999, 200))
        
        # setup empty storage for observed site FDCs
        self.observed_fdcs = {}
        

    def clean_columns(self, observed_flow):
        """
        Standardizes site identifiers by removing 'USGS-' prefixes if present.
        
        Parameters
        ----------
        observed_flow : pd.DataFrame
            DataFrame containing observed streamflow with site IDs as columns.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned column names.
        """
        observed_flow.columns = [col.replace('USGS-', '') if col.startswith('USGS-') else col 
                                 for col in observed_flow.columns]
        return observed_flow

    def verify_sites_overlap(self, observed_flow, observation_locations):
        """
        Ensures that site IDs in observed_flow match those in observation_locations.
        
        Parameters
        ----------
        observed_flow : pd.DataFrame
            Observed flow DataFrame.
        observation_locations : pd.DataFrame
            Observation location metadata.
        
        Raises
        ------
        AssertionError
            If site IDs do not match between the datasets.
        """
        missing_sites = set(observed_flow.columns) - set(observation_locations.index)
        assert not missing_sites, f"Sites {missing_sites} in observed_flow are missing from observation_locations.index."
        
    
        
    def get_obs_fdc(self, site_id):
        """
        Calculates the flow duration curve (FDC) for a given site using observed streamflow.
        
        Parameters
        ----------
        site_id : str
            Site identifier for which to calculate the FDC.
        
        Returns
        -------
        np.ndarray
            Discrete FDC values.
        """
        if site_id not in self.observed_fdcs.keys():
            Q = self.observed_flow.loc[:, site_id].dropna().values
            Q = Q[Q > 0]
            
            fdc = np.quantile(Q, self.qs)
            
            # store
            self.observed_fdcs[site_id] = fdc
        
        return self.observed_fdcs[site_id]
    
    def get_KNN(self, 
                 prediction_location,
                 donor_locations):
        """
        Finds the K nearest neighbors (KNN) relative to the given prediction location.
        
        Parameters
        ----------
        prediction_location : tuple
            (longitude, latitude) of the prediction site.
        donor_locations : pd.DataFrame
            DataFrame of observation locations with site IDs as index and columns ['long', 'lat'].
        
        Returns
        -------
        None
            Updates self.KNN_dists, self.KNN_indices, and self.KNN_site_ids.
        """
        distances = np.array([geodesic(prediction_location, 
                                       donor_locations.loc[site_id, ['long', 'lat']].values).kilometers
                               for site_id in donor_locations.index])
        
        KNN_indices = np.argpartition(distances, self.K)[:self.K]
        self.KNN_dists = distances[KNN_indices]
        self.KNN_site_ids = donor_locations.index[KNN_indices]
    
    def get_weights_from_KNN(self):
        """
        Calculates weights based on the distances to the K nearest neighbors (KNN).
        
        Returns
        -------
        None
            Updates self.wts and self.norm_wts.
        """
        # buffer so that we don't divide by 0
        buffer = 0.05
        
        # use inverse distance weighting
        wts = 1/(self.KNN_dists + buffer)**2
        norm_wts = wts/np.sum(wts)
        return norm_wts
    
    def streamflow_to_nonexceedance(self, 
                                    Q, 
                                    fdc):
        """
        Converts flow timeseries to non-exceedance probability (NEP) timeseries.
        
        Parameters
        ----------
        Q : array
            Flow values to be transformed.
        fdc : array
            FDC values for the site being transformed.
        
        Returns
        -------
        array
            NEP timeseries.
        """
        
        
        if self.log_fdc_interpolation:
            Q = Q[Q > 0]  # Avoid log(0)
            Q = np.log(Q)
            
            if sum(fdc<=0.0) > 1:
                print('WARNING: 0.0 value in FDC.')
                
            fdc = np.log(fdc)
            
        nep = np.interp(Q, fdc, self.qs, right=self.qs[-1], left=self.qs[0])
        return nep
    
    def nonexceedance_to_streamflow(self, 
                                    nep,
                                    fdc):
        """
        Converts non-exceedance probability (NEP) back to streamflow.
        
        Parameters
        ----------
        nep : array
            NEP values to convert.
        fdc : array
            FDC values for the site being transformed.
            
        Returns
        -------
        array
            Predicted streamflow values.
        """
        nep = np.clip(nep, self.qs[0], self.qs[-1])
        
        fdc = np.log(fdc) if self.log_fdc_interpolation else fdc
                    
        Q = np.interp(nep, self.qs, fdc, right=fdc[-1], left=fdc[0])
        
        Q = np.exp(Q) if self.log_fdc_interpolation else Q
        
        return Q


    def predict_nep(self, 
                    prediction_location,  
                    start_date,
                    end_date,
                    **kwargs):
        """
        Predicts non-exceedance probability (NEP) timeseries at the prediction location.

        Parameters:
        ----------
        prediction_location : tuple
            A tuple of (lat, long) for the location to predict streamflow at.
        start_date : str
            The start date of the prediction period.
        end_date : str
            The end date of the prediction period.

        Returns:
        --------
        predicted_nep : ndarray
            A timeseries of predicted NEP values at the location.
        """

        self.KNN_site_ids = None
        self.KNN_dists = None


        # Subset the observed flow to the prediction period
        donor_flow = self.observed_flow.loc[start_date:end_date, :]
        T = donor_flow.shape[0]  # Time duration
        
        # for each column, if <10 days of nan, fill with median else drop
        use_donor_sites = [c for c in donor_flow.columns if donor_flow[c].isna().sum() < 10]
        donor_flow = donor_flow.loc[:, use_donor_sites]        
        donor_flow = donor_flow.fillna(donor_flow.median())        

        # Subset the observation locations to the donor sites
        matching_sites = self.observation_locations.index.intersection(donor_flow.columns)
        donor_locations = self.observation_locations.loc[matching_sites, :]
        

        ### Find KNN sites and corresponding weights
        self.get_KNN(prediction_location=prediction_location,
                     donor_locations=donor_locations)
        assert self.KNN_site_ids is not None, 'KNN site IDs not found.'

        weights = self.get_weights_from_KNN()  # Calculate weights for each of the KNN sites


        ### Make predictions of NEP timeseries      
        ## Probabilistic sample: Sample from KNN sites based on distance
        if self.probabalistic_sample:
            
            # Probabilitically sample one of the K locations
            sample_site_id = np.random.choice(self.KNN_site_ids, 
                                              p=weights, 
                                              size = self.K, 
                                              replace=True)
            
            
            donor_nep_timeseries = np.zeros((T, self.K))
            
            # Store a copy of each, so that we don't need to re-calculate the same site twice
            sample_site_neps = {}
            for i, site_id in enumerate(sample_site_id):
                
                # If already calculated, use that same timeseries
                if site_id in sample_site_neps.keys():
                    donor_nep_timeseries[:, i] = sample_site_neps[site_id]
                    
                # Else calculate NEPs
                else:
                    sample_site_fdc = self.get_obs_fdc(site_id)
                    
                    sample_site_neps[site_id] = self.streamflow_to_nonexceedance(Q = donor_flow.loc[:, site_id].values,
                                                                                 fdc = sample_site_fdc)
                    
                    donor_nep_timeseries[:, i] = sample_site_neps[site_id]
            
            # Predicted NEP is mean of observed NEP timeseries
            predicted_nep = donor_nep_timeseries.mean(axis=1)            
        
                            
        ## Simple aggregate: Weighted mean of KNN sites
        else:
            # Create timeseries of NEPs
            donor_nep_timeseries = np.zeros((T, self.K))
            
            for i, site_id in enumerate(self.KNN_site_ids):
                sample_site_fdc = self.get_obs_fdc(site_id)
                
                donor_nep_timeseries[:, i] = self.streamflow_to_nonexceedance(Q = donor_flow.loc[:, site_id].values,
                                                                              fdc = sample_site_fdc)

            # Predicted NEP is made using IDW weighting of all sites
            self.weighted_nep_timeseries = np.multiply(donor_nep_timeseries, weights.flatten())
            predicted_nep = np.sum(self.weighted_nep_timeseries, axis=1)

        # Store
        self.predicted_nep = predicted_nep
         
        return predicted_nep
    
    
    def predict_streamflow(self,
                            prediction_location,  
                            predicted_fdc,
                            start_date,
                            end_date,
                            **kwargs):
        """
        Predicts streamflow timeseries at the prediction location.
        
        Parameters:
        ----------
        prediction_location : tuple
            A tuple of (lat, long) for the location to predict streamflow at.
        predicted_fdc : array
            FDC values for the site being predicted.
        start_date : str
            The start date of the prediction period.
        end_date : str
            The end date of the prediction period.
        
        Returns:
        --------
        predicted_flow : ndarray
            A timeseries of predicted streamflow values at the location.
        """

        ### Predict NEP timeseries
        predicted_nep = self.predict_nep(prediction_location=prediction_location,
                                                        start_date=start_date,
                                                        end_date=end_date)
        
        
        ### Convert from predicted NEP to flow timeseries
        predicted_flow = self.nonexceedance_to_streamflow(nep = predicted_nep,
                                                                fdc = predicted_fdc)        
        
        # Store
        self.predicted_flow = predicted_flow

        return predicted_flow