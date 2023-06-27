import h5py
import pandas as pd

def export_dict_ensemble_to_hdf5(dict, output_file):
    N = len(dict)

    T, M = dict[f'realization_0'].shape
    column_labels = dict[f'realization_0'].columns
    
    with h5py.File(output_file, 'w') as f:
        for i in range(N):
            grp = f.create_group(f"realization_{i+1}")
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels
            
            # Create datasets for each location's timeseries
            for j in range(M):
                data = dict[f'realization_{i+1}']
                dataset = grp.create_dataset(column_labels[j], data=data.iloc[:,j])



def extract_realization_from_hdf5(hdf5_file, realization):
    with h5py.File(hdf5_file, 'r') as f:
        realization_group = f[f"realization_{realization}"]
        
        # Extract column labels
        column_labels = realization_group.attrs['column_labels']
        
        # Extract timeseries data for each location
        data = {}
        for label in column_labels:
            dataset = realization_group[label]
            data[label] = dataset[:]
    
    return pd.DataFrame(data)
