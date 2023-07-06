# DRB-Historic-Reconstruction

This repository is part of used to generate a "historic reconstruction" of streamflow timeseries at  

The full historic reconstruction is generated for the period 1945-2023 at more than 30 locations in the basin. 

***
## Overview

*A detailed documentation of the method will be available in shortly...*



### Content

- `get_unmanaged_flow_data.py`

> This script uses the [Hyriver](https://docs.hyriver.io/) suite to query and retrieve streamflow timeseries at USGS NWIS gauge stations. Gauge stations are filtered to remove stations which are identified by the [NLDI](https://waterdata.usgs.gov/blog/nldi-intro/) as being impacted by upstream management, and to remove stations outside of the DRB.

- `make_reconstruction.py`

> This script is able to generate several variations of the historic reconstruction.  




- `./QPPQModel/`

> This module includes a model used to generate streamflow predictions at ungauged locations (or, generate *unmanaged* streamflows at managed gauge locations).  

- `


### Installation

`git clone https://github.com/Pywr-DRB/DRB-Historic-Reconstruction.git`


### Dependencies

Ensure that all dependencies listed in `requirements.txt` are installed. To install these within a virtual environment (`venv`) in the project directory:

```bash
cd DRB-Historic-Reconstruction
py -m virtualenv venv
venv\Scripts\Activate
py -m pip install -r requirements.txt
```





## Outputs

All generated timeseries are located in the `./output/` folder.  