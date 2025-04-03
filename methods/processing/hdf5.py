import h5py
import pandas as pd

from pywrdrb_node_data import obs_site_matches

pywrdrb_all_nodes = list(obs_site_matches.keys())


class HDF5DataManager:
    def __init__(self):
        pass
    
    def export_ensemble_to_hdf5(self,
                                dict,
                                output_filename):
        pass
    
    def extract_realization_from_hdf5(self,
                                    hdf5_file,
                                    realization,
                                    stored_by_node=False):
        pass
    
    def get_realization_numbers(self, 
                                filename):
        pass
    
    def combine_hdf5_files(self,
                           filenames,
                           output_filename):
        pass
    
    def extract_loo_results_from_hdf5(self,
                                      filename):
        pass
                           


def export_ensemble_to_hdf5(dict, 
                            output_file):
    """
    Export a dictionary of ensemble data to an HDF5 file.
    Data is stored in the dictionary as {realization number (int): pd.DataFrame}.
    
    Args:
        dict (dict): A dictionary of ensemble data.
        output_file (str): Full output file path & name to write HDF5.
        
    Returns:
        None    
    """
    
    dict_keys = list(dict.keys())
    N = len(dict)
    T, M = dict[dict_keys[0]].shape
    column_labels = dict[dict_keys[0]].columns.to_list()
    
    with h5py.File(output_file, 'w') as f:
        for key in dict_keys:
            data = dict[key]
            datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
            
            grp = f.create_group(key)
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels

            # Create dataset for dates
            grp.create_dataset('date', data=datetime)
            
            # Create datasets for each array subset from the group
            for j in range(M):
                dataset = grp.create_dataset(column_labels[j], 
                                             data=data[column_labels[j]].to_list())
    return








def extract_loo_results_from_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        
        ensemble_flows = {}
        for site_number in f.keys():
            
            # Bad gauge: see https://waterdata.usgs.gov/nwis/uv?site_no=01422389&legacy=1
            if site_number == '01422389':
                continue
            
            site_ensemble = f[site_number]
        
            # Extract column labels
            column_labels = site_ensemble.attrs['column_labels']
            
            # Extract timeseries data for each location
            data = {}
            for label in column_labels:
                dataset = site_ensemble[label]
                data[label] = dataset[:]
            
            # Get date indices
            dates = site_ensemble['date'][:].tolist()
            # dates = pd.to_datetime([d[1:] for d in dates])
            
            # Combine into dataframe
            df = pd.DataFrame(data, index = dates)
            df.index = pd.to_datetime(df.index.astype(str))
        
            # Store in dictionary
            ensemble_flows[site_number] = df

    return ensemble_flows
    
    


def combine_hdf5_files(files, output_file):
    """
    This function reads multiple hdf5 files and 
    combines all data into a single file with the same structure.

    The function assumes that all files have the same structure.
    """    
    assert(type(files) == list), 'Input must be a list of file paths'
    
    output_file = output_file + '.hdf5' if ('.hdf' not in output_file) else output_file
    
    # Extract all
    results_dict ={}
    for i, f in enumerate(files):
        if '.' not in f:
            f = f + '.hdf5'
        assert(f[-5:] == '.hdf5'), f'Filename {f} must end in .hdf5'
        results_dict[i] = extract_loo_results_from_hdf5(f)

    # Combine all data
    # Each item in the results_dict has site_number as key and dataframe as values.
    # We want to combine all dataframes into a single dataframe with columns corresponding to realization numbers
    combined_results = {}
    combined_column_names = []
    for i in results_dict:
        for site in results_dict[i]:
            if site in combined_results:
                combined_results[site] = pd.concat([combined_results[site], results_dict[i][site]], axis=1)
            else:
                combined_results[site] = results_dict[i][site]

    # Reset the column names so that there are no duplicates
    all_sites = list(combined_results.keys())
    n_realizations = len(combined_results[all_sites[0]].columns)
    combined_column_names = [str(i) for i in range(n_realizations)]
    for site in all_sites:
        assert len(combined_results[site].columns) == n_realizations, f'Number of realizations is not consistent for site {site}'
        combined_results[site].columns = combined_column_names
    
    # Write to file
    export_ensemble_to_hdf5(combined_results, output_file)        
    return




def get_hdf5_realization_numbers(filename):
    """
    Checks the contents of an hdf5 file, and returns a list 
    of the realization ID numbers contained.
    Realizations have key 'realization_i' in the HDF5.

    Args:
        filename (str): The HDF5 file of interest

    Returns:
        list: Containing realizations ID numbers; realizations have key 'realization_i' in the HDF5.
    """
    realization_numbers = []
    with h5py.File(filename, 'r') as file:
        # Get the keys in the HDF5 file
        keys = list(file.keys())

        # Get the df using a specific node key
        node_data = file[keys[0]]
        column_labels = node_data.attrs['column_labels']
        
        # Iterate over the columns and extract the realization numbers
        for col in column_labels:
            
            # handle different types of column labels
            if type(col) == str:
                if col.startswith('realization_'):
                    # Extract the realization number from the key
                    realization_numbers.append(int(col.split('_')[1]))
                else:
                    realization_numbers.append(col)
            elif type(col) == int:
                realization_numbers.append(col)
            else:
                err_msg = f'Unexpected type {type(col)} for column label {col}.'
                err_msg +=  f'in HDF5 file {filename}'
                raise ValueError(err_msg)
    return realization_numbers


def extract_realization_from_hdf5(hdf5_file, 
                                  realization,
                                  stored_by_node=False):
    """
    Pull a single inflow realization from an HDF5 file of inflows. 

    Args:
        hdf5_file (str): The filename for the hdf5 file
        realization (int): Integer realization index
        stored_by_node (bool): Whether the data is stored with node name as key.

    Returns:
        pandas.DataFrame: A DataFrame containing the realization
    """
    
    with h5py.File(hdf5_file, 'r') as f:
        if stored_by_node:
            # Extract timeseries data from realization for each node
            data = {}
                
            for node in pywrdrb_all_nodes:
                node_data = f[node]
                column_labels = node_data.attrs['column_labels']
                
                err_msg = f'The specified realization {realization} is not available in the HDF file.'
                assert(realization in column_labels), err_msg + f' Realizations available: {column_labels}'
                data[node] = node_data[realization][:]
            
            dates = node_data['date'][:].tolist()
            
        else:
            realization_group = f[realization]
            
            # Extract column labels
            column_labels = realization_group.attrs['column_labels']
            # Extract timeseries data for each location
            data = {}
            for label in column_labels:
                dataset = realization_group[label]
                data[label] = dataset[:]
            
            # Get date indices
            dates = realization_group['date'][:].tolist()
        data['datetime'] = dates
        
    # Combine into dataframe
    df = pd.DataFrame(data, index = dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df


