import h5py
import pandas as pd


def export_dict_ensemble_to_hdf5(dict, output_file):
    N = len(dict)
    T, M = dict[f'realization_0'].shape
    column_labels = dict[f'realization_0'].columns.to_list()
    
    with h5py.File(output_file, 'w') as f:
        for i in range(N):
            data = dict[f'realization_{i}']
            datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
            
            grp = f.create_group(f"realization_{i+1}")
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels

            # Create dataset for dates
            grp.create_dataset('date', data=datetime)
            
            # Create datasets for each location's timeseries
            for j in range(M):
                dataset = grp.create_dataset(column_labels[j], data=data[column_labels[j]].to_list())



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
        
        # Get date indices
        dates = realization_group['date'][:].tolist()
        # dates = pd.to_datetime([d[1:] for d in dates])
        
    # Combine into dataframe
    df = pd.DataFrame(data, index = dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df
