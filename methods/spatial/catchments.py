
import os
import json
import pynhd as nhd

from methods.utils.directories import DATA_DIR
from methods.utils.contants import GEO_CRS


####################################################################################################

def ensure_file_exists(file):
    """Ensure that the file exists. If it does not, create it with initial JSON data.

    Args:
    file (str): The file to check.
    """
    try:
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                json.dump({}, f)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    return True

####################################################################################################

def get_basin_data(fid, fsource):
    """Get the basin data from nldi for a given ID.
    
    Args:
        fid (str): The id for the site (comid or site_no)
        fsource (str): The source of the id (comid or nwissite)
        
    Returns:
        geopandas.geodataframe.GeoDataFrame: The basin for the given ID.
    """
    nldi = nhd.NLDI()
    try:
        basin_data = nldi.get_basins(fsource=fsource, 
                                    feature_ids=fid,
                                    split_catchment=True) # ensures the gauge is at the outlet of basin
        return basin_data
    except:
        print(f'No basin found for {fsource}: ', fid)
        return None


####################################################################################################

def get_basin_geometry(fid, 
                       fsource='comid'):
    """Get the basin geometry from nldi for a fid.
    
    Args:
        fid (str): The id for the site
        
    Returns:
        shapely.geometry.polygon.Polygon: The basin of the fid.
    """
    basin_data = get_basin_data(fid=fid,
                                fsource=fsource)
    if basin_data is not None:
        return basin_data.geometry
    else:
        return None

####################################################################################################


def calculate_geometry_area(geometry):
    """Returns the catchment area of a geometry.
    
    Args:
        geometry (shapely.geometry.polygon.Polygon): The geometry of the basin.
    """
    geom = geometry.to_crs(GEO_CRS)
    area = geom.area.values[0]/10**6
    return area
        
####################################################################################################