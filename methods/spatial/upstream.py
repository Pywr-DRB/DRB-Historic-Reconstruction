"""
This module provides a class-based implementation for managing upstream gauge data
associated with USGS gauge locations or NHD COMID numbers. It encapsulates file handling,
upstream gauge retrieval, and processing logic into a clean, reusable, and object-oriented workflow.

The `UpstreamGaugeManager` class allows users to:
- Ensure that required data files exist, creating them if necessary.
- Add or update upstream gauge data to JSON files without overwriting existing entries by default.
- Retrieve upstream gauges for single or multiple sites, either from stored data or dynamically using `pynhd`.
- Identify immediate upstream sites from a given set of upstream gauge relationships.

Example usage:
manager = UpstreamGaugeManager(data_dir="/path/to/data")
upstream_gauges = manager.get_gauges_for_single_site(fid="12345", file="comid_upstream_gauges.json")
immediate_sites = manager.get_immediate_sites(upstream_gauges, site_id="12345")
"""

import os
import json
import pynhd as nhd

from config import DATA_DIR
from config import STATION_UPSTREAM_GAGE_FILE, COMID_UPSTREAM_GAGE_FILE, DATA_DIR


class UpstreamGaugeManager:
    """
    A class to manage upstream gauge identification and data handling.

    Attributes:
    ----------
    data_dir : str
        Base directory for data files.
    comid_file : str
        File path for COMID upstream gauge data.
    station_file : str
        File path for station upstream gauge data.
    """

    def __init__(self, 
                 data_dir=DATA_DIR, 
                 comid_file_name=COMID_UPSTREAM_GAGE_FILE, 
                 station_file_name=STATION_UPSTREAM_GAGE_FILE):
        """
        Initialize the UpstreamGaugeManager.

        Args:
        -----
        data_dir : str
            Base directory for data files.
        comid_file_name : str
            File name for COMID upstream gauge data.
        station_file_name : str
            File name for station upstream gauge data.
        """
        self.data_dir = data_dir
        self.comid_file = os.path.join(data_dir, comid_file_name)
        self.station_file = os.path.join(data_dir, station_file_name)
        self.ensure_file_exists(self.comid_file)
        self.ensure_file_exists(self.station_file)

    @staticmethod
    def ensure_file_exists(file):
        """Ensure the file exists. If not, create it with empty JSON content."""
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                json.dump({}, f)

    def update_gauge_file(self, 
                          new_data, 
                          file, 
                          overwrite=False):
        """Update the gauge file with new data."""
        self.ensure_file_exists(file)
        with open(file, 'r') as f:
            current_data = json.load(f)

        for key, value in new_data.items():
            if overwrite or key not in current_data:
                current_data[key] = value

        with open(file, 'w') as f:
            json.dump(current_data, f)

    def is_station_in_file(self, 
                           fid, 
                           file):
        """Check if a station ID exists in the specified file."""
        self.ensure_file_exists(file)
        with open(file, 'r') as f:
            data = json.load(f)
        return fid in data

    def get_gauges_from_file(self, 
                             fid, 
                             file):
        """Retrieve the upstream gauges for a station from the file."""
        if not self.is_station_in_file(fid, file):
            return None
        with open(file, 'r') as f:
            data = json.load(f)
        return data[fid]

    def get_gauges_for_single_site(self, 
                                   fid, 
                                   fsource='comid', 
                                   overwrite=False,
                                   restrict_to_list=False,
                                   fid_list=None):
        """Retrieve or calculate the upstream gauges for a single site."""
        
        file = self.comid_file if fsource == 'comid' else self.station_file
        
        gauges = None if overwrite else self.get_gauges_from_file(fid, file)
        if gauges is not None:
            return gauges

        nldi = nhd.NLDI()
        usgs_fid = f'USGS-{fid}' if fsource == 'nwissite' and 'USGS-' not in fid else fid

        try:
            upstream_data = nldi.navigate_byid(
                fid=usgs_fid,
                fsource=fsource,
                source='nwissite',
                navigation='upstreamTributaries',
                distance=1000
            )
            if fsource == 'comid':
                gauges = list(set(upstream_data['comid'].values))
            elif fsource == 'nwissite':
                gauges = [str(c.split('-')[1]) for c in upstream_data['identifier'].values]
                gauges = list(set(gauges))

            gauges = [str(c) for c in gauges if c != fid]
        except Exception as e:
            print(f"Failed to find upstream gauges for {fsource} {fid}: {e}")
            return []

        # Restrict to the provided list of FIDs
        if restrict_to_list:
            if fid_list is None:
                raise ValueError("fid_list must be provided when restrict_to_list is True.")
            gauges = [g for g in gauges if g in fid_list]

        self.update_gauge_file({fid: gauges}, file, overwrite)
        return gauges

    def get_gauges_for_multiple_sites(self, 
                                      fids, 
                                      fsource='comid', 
                                      overwrite=False, 
                                      restrict_to_list=False):
        """Retrieve upstream gauges for multiple sites."""
        
        result = {}
        for fid in fids:
            gauges = self.get_gauges_for_single_site(fid=fid, 
                                                     fsource=fsource, 
                                                     overwrite=overwrite,
                                                     restrict_to_list=restrict_to_list,
                                                     fid_list=fids)
            if gauges is not None:
                result[fid] = [g for g in gauges if g in fids] if restrict_to_list else gauges
        return result

    @staticmethod
    def find_immediate_upstream_sites(upstream_gauges):
        """Identify immediate upstream sites."""
        immediate = {}
        for site, upstream_sites in upstream_gauges.items():
            immediate[site] = [
                upstream for upstream in upstream_sites
                if not any(upstream in upstream_gauges.get(intermediate, [])
                           for intermediate in upstream_sites if intermediate != upstream)
            ]
        return immediate

    @staticmethod
    def get_immediate_sites(upstream_gauges, site_id):
        """Get immediate upstream sites for a specific site."""
        if site_id not in upstream_gauges:
            raise ValueError(f"Site ID {site_id} not found.")
        all_sites = upstream_gauges[site_id]
        return [
            site for site in all_sites
            if not any(site in upstream_gauges.get(other, []) for other in all_sites if other != site)
        ]
