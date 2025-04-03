import numpy as np
import pandas as pd

def aggregate_ensemble_data(data,
                            results_sets, 
                            dataset_label, 
                            new_label, 
                            how='median'):
    """
    Combine all integer-labeled DataFrames in `d` by taking the mean or median.
    Result is stored in d[how].
    
    Parameters
    ----------
    d : dict
        Dictionary whose integer keys map to DataFrames of identical shape, 
        with the same index and columns.
    how : {'mean', 'median'}, default='median'
        Aggregation function.
    """
    
    d_rs = getattr(data, results_sets)
    d = d_rs[dataset_label]
    
    # get keys
    keys = list(d.keys())
    
    # Pull out the first DataFrame as a reference for index/columns
    ref_df = d[keys[0]]
    idx = ref_df.index
    cols = ref_df.columns
    
    # Stack all DataFrame values into a 3D NumPy array of shape (N, nrows, ncols)
    # Here N = number of integer-labeled DataFrames
    arr_3d = np.stack([d[k].to_numpy() for k in keys], axis=0)
    
    # Compute along the first axis => shape becomes (nrows, ncols)
    if how == 'mean':
        aggregated = np.mean(arr_3d, axis=0)
    elif how == 'median':
        aggregated = np.median(arr_3d, axis=0)
    else:
        raise ValueError(f"Unknown aggregation requested: {how}")
    
    # Create a new DataFrame with the same shape/index/columns
    out_df = pd.DataFrame(aggregated, index=idx, columns=cols)
    
    # Place the result back into the dictionary
    d_rs[new_label] = {}
    d_rs[new_label][0] = out_df
    setattr(data, results_sets, d_rs)    
    return data