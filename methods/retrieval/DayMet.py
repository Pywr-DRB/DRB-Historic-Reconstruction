
import pandas as pd
from pydaymet import pydaymet


def get_catchment_dayment(catchment, index, 
                          start_date, end_date, 
                          timescale, vars, 
                          daymet_dates=None,
                          aggregations=['sum', 'mean']):
    """
    Get the DayMet data (vars) for a single catchment geometry. 
    Aggregates data using `aggregations` (e.g., sum, mean) for each timestep.

    Args:
        catchment (_type_): _description_
        index (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_
        timescale (_type_): _description_
        vars (_type_): _description_
        daymet_dates (_type_, optional): _description_. Defaults to None.
        aggregations (_type_, optional): _description_. Defaults to ['sum', 'mean'].
    Returns:
        _type_: _description_
    """
    try:
        
        catchment_geom = catchment['geometry']
        data = pydaymet.get_bygeom(catchment_geom, 
                                   (start_date, end_date), 
                                   time_scale=timescale,
                                   variables=vars)

        # daymet data resolution square count
        n_cells = data[vars[0]].shape[1] * data[vars[0]].shape[2] 
        
        if n_cells < 20:
            print(f'Warning, small gridsize with n_cells = {n_cells}. DayMet data may be inaccurate.')
        
        # process the data        
        outputs = {}
        for v in vars:
            if v not in data.keys():
                print(f'Variable {v} not found in data')
                return index, None, None, None
            else:
                for agg in aggregations:
                    if agg == 'sum':
                        outputs[f'{v}_{agg}'] = data[v].sum(axis=1).sum(axis=1).values
                    elif agg == 'mean':
                        outputs[f'{v}_{agg}'] = data[v].mean(axis=1).mean(axis=1).values
                    else:
                        print(f'Aggregation {agg} not supported')
                        return index, None, None, None

        if daymet_dates is None:
            daymet_dates = pd.to_datetime(data[vars[0]].time.values)

        outputs['date'] = daymet_dates
        outputs['n_cells'] = n_cells
        outputs['index'] = index
        return outputs

    except Exception as e:
        print(f'Error processing {index}: {e}')
        return None