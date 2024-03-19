"""
This script is used to identify the absolute path to the pywrdrb project directory.

This is used to ensure stability in relative paths throughout the project, regardless of the 
current working directory in which a given script is run.
"""

import os

# Absolute directory to the pywrdrb folder
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))

DATA_DIR = os.path.realpath(os.path.join(ROOT_DIR, './data/')) + '/'
OUTPUT_DIR = os.path.realpath(os.path.join(ROOT_DIR, './outputs/')) + '/'
FIG_DIR = os.path.realpath(os.path.join(ROOT_DIR, './figures/')) + '/'

PYWRDRB_DIR = os.path.realpath(os.path.join(ROOT_DIR, '../Pywr-DRB/')) + '/'

path_to_nhm_data = os.path.realpath(os.path.join(ROOT_DIR, '../NHM-Data-Retrieval/datasets/NHMv10/')) + '/'
path_to_nwm_data = os.path.realpath(os.path.join(ROOT_DIR, '../NWMv21/')) + '/'

data_dir = DATA_DIR
output_dir = OUTPUT_DIR
pywrdrb_dir = PYWRDRB_DIR
fig_dir = FIG_DIR