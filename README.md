# DRB-Historic-Reconstruction

This repository contains code used to generate ensembles of probabilistic streamflow reconstructions from 1945-2023, and then run water resource systems simulations using the reconstructed historical streamflows.  The corresponding study is:
 
Amestoy, T.J., & P.M. Reed. (Submitted for Review). Integrated River Basin Assessment Framework Combing Probabilistic Streamflow Reconstruction, Bayesian Bias Correction, and Drought Storyline Analysis. Environmental Modeling and Software.

## Overview

The workflow consists of several sequential steps:
1. Data retrieval and preprocessing from various sources
2. Flow duration curve (FDC) bias correction analysis
3. Leave-one-out reconstructions at 85 diagnostic gauges for model validation
4. Generation of streamflow reconstructions at Pywr-DRB locations
5. Simulation of streamflow ensemble from 1945-2023 using Pywr-DRB


## Requirements

- Python 3.11+
- MPI for parallel processing
- Pywr-DRB (https://github.com/Pywr-DRB/Pywr-DRB)
- High-performance computing environment (100+ cores recommended)
- See `requirements.txt` for a complete list of dependencies

## Setup

Create and activate a virtual environment:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Execution

This workflow is designed to run on high-performance computing (HPC) environments. It has been run on Cornell University's Hopper HPC cluster. The complete workflow requires significant computational resources (100+ cores) to be completed in a reasonable timeframe.

To run the workflow on an HPC system with SLURM:
```bash
sbatch run_main.sh
```

Scripts are named with sequential numbering, for example:
```bash
python 01_get_source_data.py  # Retrieves data from USGS, NID, DayMet, etc.
python 02_preprocess_data.py  # Processes data and identifies diagnostic sites
# etc
```

## Data Sources

The workflow integrates data from multiple sources:
- USGS streamflow gauges
- National Hydrologic Model (NHM) v1.0
- National Inventory of Dams (NID)
- NLDI catchment characteristics
- DayMet climate data

## Output

The main outputs include:
- Bias-corrected streamflow ensembles at key Pywr-DRB nodes
- Diagnostic figures and error metrics
- Water management simulation results

## Data Access

Due to the large size of the ensemble output datasets and Pywr-DRB inputs/outputs, these files are not included in this repository. However, they can be accessed in a compressed format through [a separate Zenodo repository HERE](https://doi.org/10.5281/zenodo.15101163).
