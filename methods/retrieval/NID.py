
import os

from pygeohydro import NID
import geopandas as gpd
import pandas as pd

from methods.utils.directories import DATA_DIR, PYWRDRB_DIR
from methods.utils.constants import GEO_CRS

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



class NIDManager:
    """
    A class to manage and process data from the National Inventory of Dams (NID).

    Attributes:
    ----------
    data_dir : str
        Base directory for storing data files.
    """

    def __init__(self, data_dir=DATA_DIR):
        """
        Initialize the DamDataManager.

        Args:
        -----
        data_dir : str
            Base directory for data files.
        """
        self.data_dir = data_dir
        self.nid = NID()

    def get_nid_data(self, 
                 crs=GEO_CRS, 
                 min_storage=0, 
                 filter_drb=True, 
                 filter_bbox=None):
        """
        Retrieve dam data with optional filtering by size, boundary, and bounding box.

        Args:
        -----
        crs : str
            Coordinate reference system for the output GeoDataFrame.
        min_storage : float
            Minimum storage capacity for filtering dams.
        filter_drb : bool
            Whether to filter dams to only those within the Delaware River Basin.
        filter_bbox : tuple
            Bounding box (xmin, ymin, xmax, ymax) for filtering dams.

        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame containing the filtered dams.
        """
        drb_states = ["New Jersey", "New York", "Pennsylvania", "Delaware"]
        all_dams = []

        for state in drb_states:
            query_list = [
                {"state": [state]},
                {"nidStorage": [f"[{min_storage} 99999999999]"]}
            ]
            results = self.nid.get_byfilter(query_list)
            dams_in_state = results[0]
            dams_in_sizerange = results[1]
            filtered_dams = dams_in_state[dams_in_state['federalId'].isin(dams_in_sizerange['federalId'])]
            all_dams.append(filtered_dams)

        all_dams_df = pd.concat(all_dams, ignore_index=True)
        all_dams_gdf = gpd.GeoDataFrame(all_dams_df).set_crs(crs)

        if filter_drb:
            drb_boundary = load_drb_boundary(crs=crs)
            all_dams_gdf = all_dams_gdf.clip(drb_boundary.geometry.values[0])

        if filter_bbox is not None:
            all_dams_gdf = all_dams_gdf.cx[filter_bbox[0]:filter_bbox[2], filter_bbox[1]:filter_bbox[3]]

        all_dams_gdf.reset_index(drop=True, inplace=True)
        print(f"colums: {all_dams_gdf.columns}")
        return all_dams_gdf

    def get_dam_storages(self, dams_df):
        """
        Retrieve the storage capacities for a list of dams.

        Args:
        -----
        dams_df : gpd.GeoDataFrame
            GeoDataFrame containing dam information.

        Returns:
        --------
        list
            List of storage capacities for the dams.
        """
        dam_storages = []
        for i, dam in dams_df.iterrows():
            dam_id = [dam['federalId']]
            try:
                storage = self.nid.inventory_byid(dam_id).nidStorage.values[0]
                dam_storages.append(storage)
            except Exception as e:
                print(f"Failed to retrieve storage for dam {dam_id}: {e}")
                dam_storages.append(None)
        return dam_storages
    
    def add_dam_storages_to_gdf(self, dams_df):
        """
        Add storage capacities to a GeoDataFrame of dams.

        Args:
        -----
        dams_df : gpd.GeoDataFrame
            GeoDataFrame containing dam information.

        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame with storage capacities added.
        """
        dam_storages = self.get_dam_storages(dams_df)
        dams_df['storage'] = dam_storages
        return dams_df

    def get_dams_within_catchments(self, catchment_df, dams_df):
        """
        Identify and summarize dams within each catchment.

        Args:
        -----
        catchment_df : gpd.GeoDataFrame
            GeoDataFrame containing catchment boundaries.
        dams_df : gpd.GeoDataFrame
            GeoDataFrame containing dam data.

        Returns:
        --------
        pd.DataFrame
            DataFrame summarizing the number of dams and total storage for each catchment.
        """
        if not isinstance(catchment_df, gpd.GeoDataFrame) or not isinstance(dams_df, gpd.GeoDataFrame):
            raise ValueError("Both inputs must be GeoDataFrames.")

        # Avoid conflict between index name and columns
        if 'site_no' in catchment_df.columns:
            catchment_df = catchment_df.reset_index(drop=True)         
   
        dams_within_catchments = gpd.sjoin(dams_df, catchment_df, how="inner", predicate='within')
        grouped = dams_within_catchments.groupby(dams_within_catchments.index_right).agg(
            n_dams=('id', 'size'),
            total_storage=('storage', 'sum')
        )

        result_df = catchment_df.merge(grouped, left_index=True, right_index=True, how='left')
        result_df['n_dams'].fillna(0, inplace=True)
        result_df['total_storage'].fillna(0, inplace=True)
        
        return result_df.dropna()

    def save(self, 
             data, 
             file,
             **kwargs):
        """
        Save data to a file in the specified format.

        Args:
        -----
        data : pd.DataFrame or gpd.GeoDataFrame
            Data to save.
        file : str
            Path to the output file.
        kwargs : dict
            Additional arguments to pass to the save method.

        Returns:
        --------
        None
        """
        filetype = file.split('.')[-1]
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if filetype == 'shp':
            data.to_file(file, **kwargs)
        elif filetype == 'csv':
            data.to_csv(file, **kwargs)
            
    def load(self, file, **kwargs):
        """
        Load data from a file.

        Args:
        -----
        file : str
            Path to the input file.
        kwargs : dict
            Additional arguments to pass to the load method.

        Returns:
        --------
        pd.DataFrame or gpd.GeoDataFrame
            Loaded data.
        """
        filetype = file.split('.')[-1]
        if filetype == 'shp':
            return gpd.read_file(file, **kwargs)
        elif filetype == 'csv':
            return pd.read_csv(file, **kwargs)
        























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
        dam_id = [dam_id]
        dam_storage.append(nid.inventory_byid(dam_id).nidStorage.values[0])

    return dam_storage


# Function for getting dam data within catchments
# Assuming both catchment_df and dams_df are GeoDataFrames
def get_dam_data_within_catchment(catchment_df, dams_df):
    # Ensure the data frames are GeoDataFrames
    if not isinstance(catchment_df, gpd.GeoDataFrame) or not isinstance(dams_df, gpd.GeoDataFrame):
        raise ValueError("Input dataframes must be GeoDataFrames.")

    # Use spatial join to find dams within each catchment
    # This operation automatically matches the catchment geometry with each dam's geometry
    dams_within_catchments = gpd.sjoin(dams_df, catchment_df, how="inner", op='within')

    # Group by the catchment index and aggregate data
    grouped = dams_within_catchments.groupby('index_right').agg(
        n_dams=('index_right', 'size'), 
        total_storage=('storage', 'sum')
    )

    # Merge results back to catchment_df to retain all catchments
    catchment_dams_df = catchment_df.merge(grouped, left_index=True, right_index=True, how='left')

    # Fill NaN values where no dams are found within a catchment
    catchment_dams_df['n_dams'].fillna(0, inplace=True)
    catchment_dams_df['total_storage'].fillna(0, inplace=True)

    # drop geometry
    if 'geometry' in catchment_dams_df.columns:
        catchment_dams_df.drop('geometry',axis=1, inplace=True)
    catchment_dams_df.dropna(axis=0, inplace=True)

    return catchment_dams_df