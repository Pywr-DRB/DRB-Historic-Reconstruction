"""Run all variations of historic reconstruction.
"""

from generate_reconstruction import generate_reconstruction


if __name__ == '__main__':

    generate_ensembles= True
    n_ensemble_samples= 30

    # Constants across variations
    start_year= 1945
    end_year= 2022
    remove_mainstem_gauges= True
    K_knn= 5
    
    for fdc_source in ['nhmv10', 'nwmv21']:
        print(f'Generating {fdc_source}-FDC based reconstructions.')
        # Without NYC scaling
        generate_reconstruction(start_year=start_year, end_year=end_year,
                                N_REALIZATIONS=1,
                                donor_fdc= fdc_source, K=K_knn,
                                regression_nhm_inflow_scaling= False,
                                remove_mainstem_gauges=remove_mainstem_gauges)

        # With NYC scaling
        generate_reconstruction(start_year=start_year, end_year=end_year,
                                N_REALIZATIONS= 1,
                                donor_fdc= fdc_source, K=K_knn,
                                regression_nhm_inflow_scaling= True,
                                remove_mainstem_gauges=remove_mainstem_gauges)
            
        # Create an ensemble of reconstructions using QPPQ sampling from KNN gauges
        if generate_ensembles:
                
                print(f'Generating ensemble of size {n_ensemble_samples} with {fdc_source} based FDCs at PUB locations.')                 
                generate_reconstruction(start_year=start_year, end_year=end_year,
                                        N_REALIZATIONS= n_ensemble_samples,
                                        donor_fdc= fdc_source, K=10, 
                                        regression_nhm_inflow_scaling= True,
                                        remove_mainstem_gauges=remove_mainstem_gauges)   

print('Done! Go to reconstruction_diagnostics.ipynb to see the result.')
