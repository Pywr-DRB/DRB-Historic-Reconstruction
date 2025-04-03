import pandas as pd
from config import DATA_DIR

def load_daymet_prcp(catchment_type, 
                     timescale='monthly'):
    df = pd.read_csv(f'{DATA_DIR}/Daymet/{catchment_type}_catchment_{timescale}_mean_prcp.csv', 
                                    index_col=0).dropna(axis=1)
    return df