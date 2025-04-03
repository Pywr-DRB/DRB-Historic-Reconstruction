import planetary_computer
import pystac_client
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np

from config import GEO_CRS

daymet_proj = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"



class DayMetRetriever:
    """Handles retrieval and spatial subsetting of Daymet data."""
    
    def __init__(self, 
                 timescale: str = "annual"):
        
        self.collection_name = f"daymet-{timescale}-na"
        self.daymet_proj = daymet_proj
        self.ds = None
        
        # Configure logging to suppress Azure storage messages
        import logging
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
        
        
    def get_daymet_in_bbox(self, 
                           bbox: List[float]) -> xr.Dataset:
        """Retrieves Daymet data for specified bounding box and time range."""
        if len(bbox) != 4:
            raise ValueError("bbox must contain [west, south, east, north] coordinates")
            
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1")
        collection = catalog.get_collection(self.collection_name)
        
        signed = planetary_computer.sign(collection.assets["zarr-abfs"])
        storage_options = signed.to_dict()['xarray:storage_options']
        
        self.ds = xr.open_zarr(signed.href, 
                               consolidated=True,
                               storage_options=storage_options)
        return self.ds
    

    
class DayMetProcessor:
    """Handles aggregation of Daymet data across dimensions."""

    daymet_proj = daymet_proj
    

    def subset_daymet_using_catchment(self, 
                                      ds: xr.Dataset, 
                                      catchment_geom) -> xr.Dataset:
        """Subsets Daymet data using a catchment geometry."""
            
        # Transform catchment to Daymet projection
        bounds = catchment_geom.bounds
        
        # Subset and clip dataset
        ds_subset = ds.sel(
            x=slice(bounds[0], bounds[2]),
            y=slice(bounds[3], bounds[1])
        )
        
        ds_subset = ds_subset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        ds_subset.rio.write_crs(self.daymet_proj, inplace=True)
        
        return ds_subset.rio.clip([catchment_geom], drop=True)
    
    @staticmethod
    def aggregate_spatial(ds: xr.Dataset, 
                          method: str = 'mean',
                          variable: str = None) -> pd.DataFrame:
        """Aggregates data across spatial dimensions."""
        if method not in ['mean', 'sum']:
            raise ValueError("Method must be either 'mean' or 'sum'")
            
        
        if method == 'mean':
            val = ds[variable].mean(dim=['x', 'y']).to_numpy()
        else:
            val = ds[variable].sum(dim=['x', 'y']).to_numpy()
            
        return val


    def plot_catchment_daymet_data(self, 
                                   ds_clipped: xr.Dataset, 
                                   catchment_geom: gpd.GeoDataFrame,
                                   catchment_id: str,
                                   variable: str = 'prcp', 
                                   time_agg: str = 'mean',
                                   figsize: Tuple[int, int] = (10, 8),
                                   figdir: str = ".") -> None:
        """Plots Daymet data with catchment overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if time_agg == 'mean':
            ds_clipped[variable].mean(dim=['time']).plot(ax=ax)
        elif time_agg == 'sum':
            ds_clipped[variable].sum(dim=['time']).plot(ax=ax)
            
        catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_geom])
        catchment_gdf.boundary.plot(ax=ax, color='k', linewidth=2)
        
        plt.title(f'{time_agg.capitalize()} {variable} with Catchment Boundary')
        plt.tight_layout()
        plt.savefig(Path(figdir) / f'{catchment_id}_{variable}_{time_agg}_catchment.png')
        plt.close()
        return 



class DayMetManager:
    """Handles data storage and retrieval."""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_netcdf(self, 
                    ds: xr.Dataset, 
                    filename: str, 
                    ) -> None:
        """Saves dataset as NetCDF file."""
        filepath = self.base_dir / filename
        # Define valid encoding parameters for netCDF4
        valid_encoding = {
            'dtype', 'zlib', 'complevel', 'chunksizes', 
            'shuffle', '_FillValue', 'contiguous'
        }
        
        encoding = {}
        for var in ds.variables:
            # Extract original encoding
            original_encoding = ds[var].encoding
            
            # Create new encoding with only valid parameters
            encoding[var] = {
                k: v for k, v in original_encoding.items() 
                if k in valid_encoding
            }
            
            if var == 'time':
                encoding[var]['units'] = original_encoding['units']
            
            # Ensure chunking is properly specified if present
            if 'chunks' in original_encoding:
                encoding[var]['chunksizes'] = original_encoding['chunks']
        
        ds.to_netcdf(
            filepath,
            encoding=encoding,
            engine='netcdf4',
            format='NETCDF4'
        )
        
    def load_netcdf(self, filename: str) -> xr.Dataset:
        """Loads dataset from NetCDF file."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")
        return xr.open_dataset(filepath, engine='netcdf4', chunks='auto')
        
    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Saves aggregated data as CSV."""
        filepath = self.base_dir / filename
        df.to_csv(filepath)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Loads aggregated data from CSV."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")
        return pd.read_csv(filepath, index_col=0)
    
    


def process_daymet_data_for_catchments(
    catchment_type, 
    ds: xr.Dataset,
    station_catchments: gpd.GeoDataFrame,
    timescale: str, 
    processor: DayMetProcessor = None,
    manager: DayMetManager = None,
    plot_results: bool = False,
    figdir: str = ".",
    ) -> None:
    
    agg_methods = ['mean', 'sum']
    all_catchment_agg_data = {}
    all_catchment_agg_data['mean'] = {}
    all_catchment_agg_data['sum'] = {}
    
    proj_catchments = station_catchments.to_crs(processor.daymet_proj)
    
    # Process each catchment
    for idx, _ in station_catchments.iterrows():
        
        catchment_geom = proj_catchments.loc[idx, 'geometry']
        
        # catchment_geom may have multiple rows, with matching idx. Use the first
        if isinstance(catchment_geom, gpd.GeoSeries):
            catchment_geom = catchment_geom.iloc[0]
        
        # Subset data for catchment
        ds_clipped = processor.subset_daymet_using_catchment(ds, catchment_geom)
        
        for method in agg_methods:
            # Process and save aggregated data
            agg_prcp = processor.aggregate_spatial(ds_clipped, 
                                                   method=method,
                                                   variable='prcp')
            all_catchment_agg_data[method][idx] = agg_prcp
        
        # Create visualization
        if plot_results:
            processor.plot_catchment_daymet_data(ds_clipped, 
                                                catchment_geom,
                                                catchment_id=idx,
                                                variable='prcp',
                                                time_agg='mean',
                                                figsize=(10, 8),
                                                figdir=figdir)
    
    all_catchment_mean_prcp_df = pd.DataFrame(all_catchment_agg_data['mean'])
    all_catchment_sum_prcp_df = pd.DataFrame(all_catchment_agg_data['sum'])
    
    manager.save_csv(all_catchment_mean_prcp_df, 
                                 f"{catchment_type}_catchment_{timescale}_mean_prcp.csv")
    manager.save_csv(all_catchment_sum_prcp_df,
                                    f"{catchment_type}_catchment_{timescale}_sum_prcp.csv")
        
    print('DONE WITH DAYMET!')
    return