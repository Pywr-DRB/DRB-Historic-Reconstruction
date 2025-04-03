import geopandas as gpd
from config import DATA_DIR, GEO_CRS

def load_drb_boundary(crs = GEO_CRS,
                      shpfile = f'{DATA_DIR}DRB_spatial/DRB_shapefiles/drb_bnd_polygon.shp'):
    """Loads the DRB boundary shapefile.
    
    Args:
        crs (str, optional): The coordinate reference system. Defaults to GEO_CRS.
    
    Returns:
        gpd.GeoDataFrame: The Delaware River Basin boundary shapefile.
    """
    drb_boarder = gpd.read_file(shpfile)
    drb_boarder = drb_boarder.to_crs(crs)
    return drb_boarder