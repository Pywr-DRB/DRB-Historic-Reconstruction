# DRB-Historic-Reconstruction

This repository is part of used to generate a "historic reconstructions" of naturalized streamflow timeseries at each of the [PywrDRB](https://pywr-drb.github.io/Pywr-DRB/intro.html) model nodes.  

The full historic reconstructions span the period 1945-2023, at more than 30 locations in the basin. 

At ungauged or historically managed locations, prediction of naturalized streamflow is done using the QPPQ method (Fennessey, 1994) with Flow Duration Curve (FDC) estimates derived from [NHMv1.0](https://www.usgs.gov/mission-areas/water-resources/science/national-hydrologic-model-infrastructure) or [NWMv2.1](https://water.noaa.gov/about/nwm) modeled streamflows. 


***
## Overview

Four alternative versions of the reconstruction are generated: two single-trace estimates and two ensemble-based estimates.  

In the single-trace estimates, non-exceedance probability (NEP) timeseries are estimated at ungauged locations using an aggregation of observed NEP timeseries at nearby gauge locations (standard QPPQ). The NEP timeseries are then converted to estimated flow timeseries using the estimated FDC. This is repeated using both NHM (`nhmv10`) and NWM (`nwmv21`) modeled FDCs.  

In the ensemble reconstruction, NEP timeseries are drawn from a *single* nearby gauge for each year. Rather than aggregating `K` NEP timeseries, we sample a single timeseries from the `K` nearest, where each gauge has a probability of being selected which is inversely proportional to square of the distance between the prediction location at the observed gague location (i.e., $p_i= \frac{1}{d_i}$ where $d_i$ is the distance between each gauge and the prediction location). By selecting a single gauge for each year, we are able to maintain the extreme-values present in the historic record, rather than reducing these events via aggregation.  By default, 30 samples of the reconstruction are generated. 


### Quick-Start

> ***Note:*** Use of code in this repository assumes that you have the [Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB) repository accessible in the same parent folder. 

```
.
└── folder/
    ├── Pywr-DRB
    └── DRB-Historic-Reconstruction
```

To replicate the reconstruction process, begin by cloning this repository:

```
git clone https://github.com/Pywr-DRB/DRB-Historic-Reconstruction.git
```

Ensure that all dependencies listed in `requirements.txt` are installed. To install these within a virtual environment (`venv`) in the project directory:

```bash
cd DRB-Historic-Reconstruction
py -m virtualenv venv
venv\Scripts\Activate
py -m pip install -r requirements.txt
```

Regenerate all historic reconstruction variations (including a 30-sample ensemble of the probablistic QPPQ reconstructions) by calling:
```
py generate_all_reconstructions.py
```

All reconstructions will be exported to the `./outputs/` folder.

## Content

### Scripts

- `get_usgs_flow_data.py`

> This script uses the [Hyriver](https://docs.hyriver.io/) suite to query and retrieve streamflow timeseries at USGS NWIS gauge stations. First, all gauge data for the PywrDRB node gauges are retrieved and exported as `streamflow_daily_usgs_cms.csv`.  Then, gauge stations are filtered to remove stations which are identified by the [NLDI](https://waterdata.usgs.gov/blog/nldi-intro/) as being impacted by upstream management, and to remove stations outside of the DRB.


- `generate_reconstruction.py`

> This script contains the `generate_reconstruction()` function which is used to generate several variations of the historic reconstruction.  


- `generate_all_reconstructions.py`

> Run this script to create the 4 alternative versions of the reconstruction.


- `./QPPQModel/`

> This module includes a model used to generate streamflow predictions at ungauged locations (or, generate *unmanaged* streamflows at managed gauge locations).  See this blog post for a description of the QPPQ method: [*QPPQ Method for Streamflow Prediction at Ungauged Sites*](https://waterprogramming.wordpress.com/2022/11/22/qppq-method-for-streamflow-prediction-at-ungauged-sites-python-demo/)


- `reconstruction_diagnostics.ipynb`

> A notebook used to generate figures and check the outputs of the reconstruction.



## References

Fennessey, N. M. (1994). *A hydro-climatological model of daily stream flow for the northeast United States*. Tufts University.