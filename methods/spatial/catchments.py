import os
import json
import pandas as pd
import geopandas as gpd
import pynhd as nhd

from methods.utils.directories import DATA_DIR
from methods.utils.constants import GEO_CRS
from config import BOUNDARY
from config import COMID_UPSTREAM_GAGE_FILE, STATION_UPSTREAM_GAGE_FILE

class CatchmentManager:
    """
    A class to manage catchment data for hydrological stations and nodes.

    Attributes:
    ----------
    data_dir : str
        Base directory for storing and loading data files.
    boundary : str
        Identifier for the boundary or region used in file naming.
    """

    def __init__(self, 
                 data_dir=DATA_DIR, 
                 boundary=BOUNDARY):
        """
        Initialize the CatchmentManager.

        Args:
        -----
        data_dir : str
            Base directory for data files.
        boundary : str
            Identifier for the boundary or region.
        """
        self.data_dir = data_dir
        self.boundary = boundary
        self.nldi = nhd.NLDI()
        
        self.upstream_gage_dict = None


    def get_basin_data(self, fid, fsource):
        """Get the basin data from nldi for a given ID.
        
        Args:
            fid (str): The id for the site (comid or site_no)
            fsource (str): The source of the id (comid or nwissite)
            
        Returns:
            geopandas.geodataframe.GeoDataFrame: The basin for the given ID.
        """
        
        try:
            basin_data = self.nldi.get_basins(fsource=fsource, 
                                        feature_ids=fid,
                                        split_catchment=True) 
            if basin_data is None:
                print(f'No catchment found for {fsource}: ', fid)
                return None
            basin_data.set_geometry('geometry', inplace=True, crs=GEO_CRS)            
            return basin_data
        
        except Exception as e:
            print(f"Failed to get catchment geometry for ID {fid}: {e}")
            return None


    def get_catchment_geometry(self, fid, fsource='nwissite'):
        """Get the basin geometry from nldi for a fid.
        
        Args:
            fid (str): The id for the site
            fsource (str): The source of the id (comid or nwissite)
            
        Returns:
            shapely.geometry.polygon.Polygon: The basin of the fid.
        """
        basin_data = self.get_basin_data(fid=fid,
                                    fsource=fsource)
        if basin_data is not None:
            return basin_data.geometry
        else:
            return None

    def get_catchments_from_list(self, fid_list, fsource='nwissite', keep_largest=True):
        """
        Generate catchment geometries for a list of feature IDs.

        Args:
        -----
        fid_list : list
            List of feature IDs for which to generate catchments.
        fsource : str
            Source of the IDs (e.g., 'nwissite' or 'comid').

        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame containing catchment geometries for the input list.
        """
        assert isinstance(fid_list, list), f"Input must be a list; got {type(fid_list)}."
        station_catchments = None

        if fsource == 'nwissite':
            fid_list = [f'USGS-{fid}' for fid in fid_list if 'USGS' not in fid]

        n_sites = len(fid_list)

        for i, fid in enumerate(fid_list):
            if i % 50 == 0:
                print(f"Getting catchment geometry for site {i+1} of {n_sites}...")
            
            catchment = self.get_catchment_geometry(fid=fid, 
                                                    fsource=fsource)
            if catchment is not None:
                if station_catchments is None:
                    station_catchments = catchment
                else:
                    station_catchments = pd.concat([station_catchments, catchment])

        if station_catchments is not None:
            station_catchments = station_catchments.explode(index_parts=False).dropna(axis=0, how='any')
            valid_indices = station_catchments.index.intersection(fid_list)
            station_catchments = station_catchments.loc[valid_indices]
            
            # drop duplicates, keep largest area catchment
            if keep_largest:
                station_catchment_areas = station_catchments.apply(self.calculate_geometry_area)
                station_catchment_areas = station_catchment_areas.sort_values(ascending=False)
                station_catchments = station_catchments[~station_catchment_areas.index.duplicated(keep='first')]

        # convert to GeoDataFrame with geometry column
        station_catchments = gpd.GeoDataFrame(station_catchments, geometry='geometry', crs=GEO_CRS)
        station_catchments.index = station_catchments.index.astype(str)
        
        # remove USGS- prefix from index if nwissite
        if fsource == 'nwissite':
            station_catchments.index = station_catchments.index.str.replace('USGS-', '')

        station_catchments.index.name = fsource
        return station_catchments
    
    def calculate_geometry_area(self, geometry):
        """Returns the catchment area of a geometry.
        
        Args:
            geometry (shapely.geometry.polygon.Polygon): The geometry of the basin.
        """
        area = geometry.area
        return area
    
            
    def save_catchments(self, catchments, file):
        """
        Save catchment geometries to a file.

        Args:
        -----
        catchments : gpd.GeoDataFrame
            GeoDataFrame containing catchment geometries.
        file_name : str
            File name for saving the catchments.
        """
        os.makedirs(os.path.dirname(file), exist_ok=True)
        catchments.to_file(file)

        

    def load_catchments(self, file):
        """
        Load catchment geometries from a file.

        Args:
        -----
        file_name : str
            File name for loading the catchments.

        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame containing loaded catchment geometries.
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        index_name = 'comid' if 'pywrdrb' in file else 'site_no'
        catchments = gpd.read_file(file, dtype={index_name: str})
        catchments.index = catchments[index_name]
        catchments.drop(index_name, axis=1)
        
        # Convert to a projected CRS
        # before calculating area
        # catchments.to_crs(GEO_CRS, inplace=True)
        catchments = catchments.to_crs(epsg=3857)
        catchments['area_m2'] = catchments.area
        catchments['area_km2'] = catchments['area_m2'] / 1e6
        
        # convert back to geo crs
        catchments.to_crs(GEO_CRS, inplace=True)
        return catchments






####################################################################################################
### OLD
####################################################################################################

# def ensure_file_exists(file):
#     """Ensure that the file exists. If it does not, create it with initial JSON data.

#     Args:
#     file (str): The file to check.
#     """
#     try:
#         if not os.path.isfile(file):
#             with open(file, 'w') as f:
#                 json.dump({}, f)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False
#     return True

# ####################################################################################################

# def get_basin_data(fid, fsource):
#     """Get the basin data from nldi for a given ID.
    
#     Args:
#         fid (str): The id for the site (comid or site_no)
#         fsource (str): The source of the id (comid or nwissite)
        
#     Returns:
#         geopandas.geodataframe.GeoDataFrame: The basin for the given ID.
#     """
#     nldi = nhd.NLDI()
#     try:
#         basin_data = nldi.get_basins(fsource=fsource, 
#                                     feature_ids=fid,
#                                     split_catchment=True) # ensures the gauge is at the outlet of basin
#         return basin_data
#     except:
#         print(f'No basin found for {fsource}: ', fid)
#         return None


# ####################################################################################################

# def get_basin_geometry(fid, 
#                        fsource='comid'):
#     """Get the basin geometry from nldi for a fid.
    
#     Args:
#         fid (str): The id for the site
        
#     Returns:
#         shapely.geometry.polygon.Polygon: The basin of the fid.
#     """
#     basin_data = get_basin_data(fid=fid,
#                                 fsource=fsource)
#     if basin_data is not None:
#         return basin_data.geometry
#     else:
#         return None

# ####################################################################################################


# def calculate_geometry_area(geometry):
#     """Returns the catchment area of a geometry.
    
#     Args:
#         geometry (shapely.geometry.polygon.Polygon): The geometry of the basin.
#     """
#     geom = geometry.to_crs(GEO_CRS)
#     area = geom.area.values[0]/10**6
#     return area
        
# ####################################################################################################