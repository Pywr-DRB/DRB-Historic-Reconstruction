from pygeohydro import NID
import geopandas as gpd
import pandas as pd

from methods.utils.directories import DATA_DIR, PYWRDRB_DIR
from methods.utils.contants import GEO_CRS


def load_drb_boundary(crs = GEO_CRS,
                      shpfile = f'{PYWRDRB_DIR}DRB_spatial/DRB_shapefiles/drb_bnd_polygon.shp'):
    """Loads the DRB boundary shapefile.
    
    Args:
        crs (str, optional): The coordinate reference system. Defaults to GEO_CRS.
    
    Returns:
        gpd.GeoDataFrame: The Delaware River Basin boundary shapefile.
    """
    drb_boarder = gpd.read_file(shpfile)
    drb_boarder = drb_boarder.to_crs(crs)
    return drb_boarder


def load_drb_mainstem(crs = GEO_CRS,
                        shpfile = f'{PYWRDRB_DIR}DRB_spatial/DRB_shapefiles/delawareriver.shp'):
        """Loads the DRB mainstem shapefile.
        
        Args:
            crs (str, optional): The coordinate reference system. Defaults to GEO_CRS.
        
        Returns:
            gpd.GeoDataFrame: The Delaware River Basin mainstem shapefile.
        """
        drb_mainstem = gpd.read_file(shpfile)
        drb_mainstem = drb_mainstem.to_crs(crs)
        return drb_mainstem

def export_nid_metadata():
    """Exports metadata for the NID database.
    """
    nid = NID()
    nid.fields_meta.to_csv(f'{DATA_DIR}/NID/NID_fields_meta.csv')
    return

def get_nid_data(crs=GEO_CRS, 
                     MIN_STORAGE=0,
                     filter_drb=True,
                     filter_bbox=None):    
    
    nid= NID()
    drb_states = ["New Jersey", "New York",  
              "Pennsylvania", "Delaware"]

    for i, state in enumerate(drb_states):
        query_list = [
            {"state": [state]},
            {"nidStorage": [f"[{MIN_STORAGE} 99999999999]"]}
        ]
        results = nid.get_byfilter(query_list)
        dams_in_state = results[0]
        dams_in_sizerange = results[1]
        
        dams_in_state_in_sizerange = dams_in_state[dams_in_state['federalId'].isin(dams_in_sizerange['federalId'])]
        
        if i == 0:
            drb_dams_df = dams_in_state_in_sizerange
        else:
            drb_dams_df = pd.concat([drb_dams_df, dams_in_state_in_sizerange], 
                                    axis=0, ignore_index=True)
    drb_dams_df = drb_dams_df.to_crs(GEO_CRS)

    if filter_drb:
        drb_boarder = load_drb_boundary(crs=crs)
        drb_dams_df = drb_dams_df.clip(drb_boarder.geometry.values[0])

    if filter_bbox is not None:
        drb_dams_df = drb_dams_df.cx[filter_bbox[0]:filter_bbox[2], filter_bbox[1]:filter_bbox[3]]
        
    drb_dams_df.reset_index(drop=True, inplace=True)
    return drb_dams_df


def get_dam_storages(dams_df):
    # Get dam storage 
    nid = NID()
    dam_storage = []
    for i, dam in dams_df.iterrows():
        if i % 50 == 0:
            print(f"Getting storage for dam {i} of {len(dams_df)}.")
        dam_id = dam['federalId']
        dam_storage.append(nid.inventory_byid(dam_id).nidStorage.values[0])

    return dam_storage


# Function for getting dam data within catchments
def get_dam_data_within_catchment(catchment_df, 
                                  dams_df):
    catchment_dams_df = pd.DataFrame(index=catchment_df.index)
    catchment_dams_df['n_dams'] = 0
    catchment_dams_df['total_storage'] = 0
    
    for i, id in enumerate(catchment_dams_df.index):
        if i % 50 == 0:
            print(f'Getting NID data for catchment {i} of {len(catchment_dams_df.index)}.')
        
        # Find all dams within geometry and add data to df
        # for dam_id in dams_df.index:
        #     if dams_df.loc[dam_id, 'geometry'].within(catchment_df.loc[id, 'geometry']):
        #         catchment_dams_df.loc[id, 'n_dams'] += 1
        #         catchment_dams_df.loc[id, 'total_storage'] += dams_df.loc[dam_id, 'storage'] 
        #         if catchment_dams_df.loc[id,:].isna().any():
        #             print(f"Error: NaN in catchment_dams_df for catchment {id}.")   
        
        # Filter dams_df by clipping using catchment geometry
        catchment_geometry = catchment_df.loc[id, 'geometry']
        dams_within_catchment = dams_df[dams_df.geometry.within(catchment_geometry)]
        catchment_dams_df.loc[id, 'n_dams'] = dams_within_catchment.shape[0]
        catchment_dams_df.loc[id, 'total_storage'] = dams_within_catchment['storage'].sum()
    catchment_dams_df.index.name = 'site_no'
    return catchment_dams_df