"""
This script is used to identify the absolute path to the pywrdrb project directory.

This is used to ensure stability in relative paths throughout the project, regardless of the 
current working directory in which a given script is run.
"""

import os

# Absolute directory to the pywrdrb folder
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))

data_dir = os.path.realpath(os.path.join(ROOT_DIR, './data/')) + '/'
output_dir = os.path.realpath(os.path.join(ROOT_DIR, './outputs/')) + '/'
fig_dir = os.path.realpath(os.path.join(ROOT_DIR, './figures/')) + '/'

pywrdrb_dir = os.path.realpath(os.path.join(ROOT_DIR, '../Pywr-DRB/')) + '/'