import numpy as np
import pandas as pd
import h5py
import glob
import os

from methods.utils.directories import OUTPUT_DIR


def export_posterior_samples_to_hdf5(data, site_numbers, filename):
    """
    Export a 3D numpy array to an HDF5 file with datasets named by site numbers.

    :param data: NumPy array of shape (N, M, 18)
    :param site_numbers: list or array of site numbers corresponding to axis=1 of data
    :param filename: string, name of the HDF5 file to create
    """
    with h5py.File(filename, 'w') as file:
        for i, site_number in enumerate(site_numbers):
            # Ensure each site_number is used as a dataset name and store the corresponding slice of data
            file.create_dataset(site_number, data=data[:, i, :])


def load_posterior_samples_from_hdf5(filename, return_type='array'):
    """
    Load data from an HDF5 file.

    :param filename: string, name of the HDF5 file to read
    :param return_type: string, 'array' to return a NumPy array of shape (N, M, 18),
                        or 'dict' to return a dictionary {site_number: np.array(shape=(N, 18))}
    :return: NumPy array or dictionary of NumPy arrays based on return_type
    """
    with h5py.File(filename, 'r') as file:
        site_numbers = list(file.keys())
        if return_type == 'array':
            # Preallocate array
            data_array = np.empty((len(file[site_numbers[0]]), len(site_numbers), 18))
            for i, site in enumerate(site_numbers):
                data_array[:, i, :] = file[site][:]
            return data_array
        elif return_type == 'dict':
            data_dict = {site: file[site][:] for site in site_numbers}
            return data_dict
        else:
            raise ValueError(f"Invalid return_type '{return_type}'")


def combine_posterior_samples(filenames, output_filename):
    """
    Combine posterior sample files and export a single file with all the data.

    :param filenames: list of strings, names of the HDF5 files to combine
    :param output_filename: string, name of the output HDF5 file to create
    """
    combined_data = None
    combined_site_numbers = []

    for filename in filenames:
        data = load_posterior_samples_from_hdf5(filename, return_type='array')
        site_numbers = list(load_posterior_samples_from_hdf5(filename, return_type='dict').keys())
        # print(f'Site numbers: {site_numbers}')
        if combined_data is None:
            combined_data = data
        else:
            combined_data = np.concatenate((combined_data, data), axis=1)

        for s in site_numbers:
            if s in combined_site_numbers:
                raise ValueError(f"Site number {s} already in combined_site_numbers. From file {filename}")
        combined_site_numbers.extend(site_numbers)
    export_posterior_samples_to_hdf5(combined_data, combined_site_numbers, output_filename)
        

def combine_leave_one_out_bias_samples(nxm):
    
    rank_output_fname = f'{OUTPUT_DIR}LOO/loo_bias_correction/{nxm}_bias_posterior_samples_rank*'
    final_output_fname = f'{OUTPUT_DIR}LOO/loo_bias_correction/{nxm}_bias_posterior_samples.hdf5'
    
    # Get all the files for this model
    filenames = glob.glob(rank_output_fname)
    
    if len(filenames) > 0:

        try:
            combine_posterior_samples(filenames, final_output_fname)

            for f in filenames:
                os.remove(f)
        except Exception as e:
            print(f"Error combining files: {e}")
            
    return

    
def filter_biases(Y_model, 
                  b_model, 
                  N_samples, 
                  return_corrected=True, 
                  max_sample_iter=1000,
                  start_min_variance=True, 
                  start_quantile=None,
                  print_warnings=False):
    """
    Efficiently create a filtered set of biases ensuring positive, 
    monotonically increasing, and non-NaN corrected values.
    """
    filtered_biases = {}
    y_corrected = {}
    all_quantiles = Y_model.columns.to_list()
    
    for site, biases in b_model.items():
        biases_df = pd.DataFrame(biases, columns=all_quantiles)
        Y_site = Y_model.loc[site, :].values
        variances = biases_df.var(axis=0) if start_min_variance else None
        
        if start_min_variance and start_quantile is None:
            start_quantile = variances.idxmin()
        elif start_quantile is None:
            start_quantile = variances.idxmax()
        
        start_idx = all_quantiles.index(start_quantile)
        num_quantiles = len(all_quantiles)

        selected_biases = np.zeros((N_samples, num_quantiles))
        corrected_values = np.zeros((N_samples, num_quantiles))

        # Precompute correction candidates
        bias_corrected = {q: Y_site[i] / ((biases_df.loc[:, q] / 100) + 1) for i, q in enumerate(all_quantiles)}

        # Sample the start quantile
        for i in range(N_samples):
            valid_candidates = biases_df[start_quantile][bias_corrected[start_quantile] > 0]
            if not valid_candidates.empty:
                chosen_bias = np.random.choice(valid_candidates.values)
            else:
                if print_warnings and i%100==0:
                    print(f'No viable bias range for start quantile {start_quantile} in site {site}. Using 0.')
                chosen_bias = 0.0
            
            selected_biases[i, start_idx] = chosen_bias
            corrected_values[i, start_idx] = Y_site[start_idx] / ((chosen_bias / 100) + 1) if chosen_bias else Y_site[start_idx]

        # Process right (increasing quantiles) and left (decreasing quantiles)
        for direction, quant_range in [("right", range(start_idx + 1, num_quantiles)), 
                                       ("left", range(start_idx - 1, -1, -1))]:
            for i in range(N_samples):
                for q_idx in quant_range:

                    prev_idx = q_idx - 1 if direction == "right" else q_idx + 1
                    prev_corrected = corrected_values[i, prev_idx]

                    # Get valid bias candidates
                    if direction == "right":
                        valid_candidates = biases_df.iloc[:, q_idx][bias_corrected[all_quantiles[q_idx]] > prev_corrected]
                    else:
                        valid_candidates = biases_df.iloc[:, q_idx][bias_corrected[all_quantiles[q_idx]] < prev_corrected]
                    
                    if not valid_candidates.empty:
                        selected_bias = np.random.choice(valid_candidates.values)
                    else:
                        if print_warnings and i%100==0:
                            print(f'No viable range of samples for quantile {all_quantiles[q_idx]} in site {site}')
                        selected_bias = selected_biases[i, prev_idx]

                    selected_biases[i, q_idx] = selected_bias
                    corrected_values[i, q_idx] = Y_site[q_idx] / (selected_bias / 100 + 1)

        filtered_biases[site] = selected_biases
        y_corrected[site] = corrected_values

    return (filtered_biases, y_corrected) if return_corrected else filtered_biases