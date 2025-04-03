import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
import contextily as cx

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from methods.utils.constants import mg_to_mcm
from methods.utils.directories import fig_dir
from methods.plotting.styles import ax_tick_label_fsize, ax_label_fsize, legend_fsize
from methods.plotting.styles import fig_dpi

from config import GEO_CRS
from config import DATA_DIR

from methods.plotting.data_plots import load_nhd_data, get_mainstem_geometry_from_nhd, get_tributary_geometry_from_nhd

dpi = fig_dpi
spatial_data_dir = f"{DATA_DIR}/DRB_spatial"

### Map of major DRB nodes
def make_DRB_map(fig_dir=fig_dir, units="MCM"):


    ### set crs consistent with contextily basemap
    crs = "EPSG:3857"

    # # Reservoir data
    reservoir_data = pd.read_csv(
        "pywrdrb/model_data/drb_model_istarf_conus.csv", sep=","
    )

    ### Load general shapefiles
    drb_boundary = gpd.read_file(
        f"{spatial_data_dir}/DRB_shapefiles/drb_bnd_polygon.shp"
    ).to_crs(crs)
    states = gpd.read_file(f"{spatial_data_dir}/states/tl_2010_us_state10.shp").to_crs(
        crs
    )
    
    nhd = load_nhd_data()
    drc = gpd.read_file(f"{spatial_data_dir}/NHD_NJ/DRCanal.shp").to_crs(crs)

    ### load drb node info into geodataframes
    crs_nodedata = 4386
    major_nodes = gpd.read_file(
        f"{spatial_data_dir}/model_components/drb_model_major_nodes.csv", sep=","
    )
    reservoirs = major_nodes.loc[major_nodes["type"] == "reservoir"]
    reservoirs = gpd.GeoDataFrame(
        reservoirs.drop("geometry", axis=1),
        geometry=gpd.points_from_xy(reservoirs.long, reservoirs.lat, crs=crs_nodedata),
    ).to_crs(crs)
    flow_reqs = major_nodes.loc[major_nodes["type"] == "regulatory"]
    flow_reqs = gpd.GeoDataFrame(
        flow_reqs.drop("geometry", axis=1),
        geometry=gpd.points_from_xy(flow_reqs.long, flow_reqs.lat, crs=crs_nodedata),
    ).to_crs(crs)



    ### get river network from NHD
    mainstem = get_mainstem_geometry_from_nhd(nhd)
    
    ### get all other tributary streams containing a Pywr-DRB reservoir, or downstream of one. Note that 2 different regions have Tulpehocken Creek - use only the correct one.
    tribs = get_tributary_geometry_from_nhd(nhd)

    ### rough lines for NYC Delaware Aqueduct system
    ### cannonsville to rondout
    lines = [LineString([Point(-75.37462, 42.065872), Point(-74.4296, 41.79926)])]
    ### pepacton to rondout
    lines.append(LineString([Point(-74.965531, 42.073603), Point(-74.4296, 41.79926)]))
    ### neversink to rondout
    lines.append(LineString([Point(-74.643266, 41.821286), Point(-74.4296, 41.79926)]))
    ### rondout to west branch reservoir
    lines.append(LineString([Point(-74.4296, 41.79926), Point(-73.69541, 41.41176)]))
    ### west branch reservoir to kensico reservoir
    lines.append(
        LineString([Point(-73.69541, 41.41176), Point(-73.7659656, 41.0737078)])
    )
    ### kensico reservoir to hillside reservoir
    lines.append(
        LineString([Point(-73.7659656, 41.0737078), Point(-73.8693806, 40.90715556)])
    )

    ### convert projection
    crs_longlat = "EPSG:4326"
    aqueducts = gpd.GeoDataFrame({"geometry": lines}, crs=crs_longlat)
    aqueducts = aqueducts.to_crs(crs)

    ### create map figures
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    use_basemap = True

    ### plot drb boundary
    drb_boundary.plot(ax=ax, color="none", edgecolor="k", lw=1, zorder=0.9)

    ### plot river network
    mainstem.plot(ax=ax, color="navy", lw=3, zorder=1.1)
    for trib in tribs:
        trib.plot(ax=ax, color="cornflowerblue", lw=2, zorder=1)

    ### plot reservoirs & min flow locations
    list_nyc_reservoirs = (
        "reservoir_cannonsville",
        "reservoir_pepacton",
        "reservoir_neversink",
    )
    for r in reservoirs["name"]:
        color = "firebrick" if r in list_nyc_reservoirs else "sandybrown"
        r_abbrev = r.split("_")[1]
        if units == "MG":
            try:
                s = (
                    50
                    + reservoir_data["Adjusted_CAP_MG"]
                    .loc[reservoir_data["reservoir"] == r_abbrev]
                    .iloc[0]
                    / 1000
                    * 2
                )
            except:
                s = 50
        elif units == "MCM":
            try:
                s = (
                    50
                    + reservoir_data["Adjusted_CAP_MG"]
                    .loc[reservoir_data["reservoir"] == r_abbrev]
                    .iloc[0]
                    * mg_to_mcm
                    / 2
                )
            except:
                s = 50
        reservoirs.loc[reservoirs["name"] == r].plot(
            ax=ax, color=color, edgecolor="k", markersize=s, zorder=2
        )
    flow_reqs.plot(
        ax=ax,
        color="mediumseagreen",
        edgecolor="k",
        markersize=250,
        zorder=2.1,
        marker="*",
    )

    ### NYC tunnel systsem
    aqueducts.plot(ax=ax, color="darkmagenta", lw=2, zorder=1.2, ls=":")

    ### plot NJ diversion - Delaware & Raritan Canal
    drc.plot(ax=ax, color="darkmagenta", lw=2, zorder=1.2, ls=":")

    ### add state boundaries
    if use_basemap:
        states.plot(ax=ax, color="none", edgecolor="0.5", lw=0.7, zorder=0)
    else:
        states.plot(ax=ax, color="0.95", edgecolor="0.5", lw=0.7, zorder=0)

    ### map limits
    ax.set_xlim([-8.517e6, -8.197e6])  # -8.215e6])
    ax.set_ylim([4.75e6, 5.235e6])

    ### annotations
    fontsize = 10
    fontcolor = "0.5"
    plt.annotate(
        "New York",
        xy=(-8.46e6, 5.168e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "Pennsylvania",
        xy=(-8.46e6, 5.152e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "New York",
        xy=(-8.245e6, 5.032e6),
        rotation=-31,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "New Jersey",
        xy=(-8.257e6, 5.019e6),
        rotation=-31,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "Pennsylvania",
        xy=(-8.48e6, 4.833e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "Maryland",
        xy=(-8.48e6, 4.817e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "Delaware",
        xy=(-8.43e6, 4.8e6),
        rotation=-85,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "Pennsylvania",
        xy=(-8.359e6, 5.03e6),
        rotation=50,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )
    plt.annotate(
        "New Jersey",
        xy=(-8.337e6, 5.024e6),
        rotation=50,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
    )

    fontcolor = "firebrick"
    plt.annotate(
        "Cannonsville",
        xy=(-8.404e6, 5.1835e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    plt.annotate(
        "Pepacton",
        xy=(-8.306e6, 5.168e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    plt.annotate(
        "Neversink",
        xy=(-8.268e6, 5.143e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    fontcolor = "mediumseagreen"
    plt.annotate(
        "Montague",
        xy=(-8.284e6, 5.055e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    plt.annotate(
        "Trenton",
        xy=(-8.353e6, 4.894e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    fontcolor = "darkmagenta"
    plt.annotate(
        "NYC\nDiversion",
        xy=(-8.25e6, 5.085e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )
    plt.annotate(
        "NJ\nDiversion",
        xy=(-8.315e6, 4.959e6),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fontcolor,
        fontweight="bold",
    )

    ### legend
    axin = ax.inset_axes([0.58, 0.006, 0.41, 0.25])
    axin.set_xlim([0, 1])
    axin.set_ylim([0, 1])
    ### mainstem
    axin.plot([0.05, 0.15], [0.93, 0.93], color="navy", lw=3)
    axin.annotate(
        "Delaware River",
        xy=(0.18, 0.93),
        ha="left",
        va="center",
        color="k",
        fontsize=fontsize,
    )
    ### tributaries
    axin.plot([0.05, 0.15], [0.83, 0.83], color="cornflowerblue", lw=2)
    axin.annotate(
        "Tributary",
        xy=(0.18, 0.83),
        ha="left",
        va="center",
        color="k",
        fontsize=fontsize,
    )
    ### DRB boundary
    axin.plot([0.05, 0.15], [0.73, 0.73], color="k", lw=1)
    axin.annotate(
        "Basin Boundary",
        xy=(0.18, 0.73),
        ha="left",
        va="center",
        color="k",
        fontsize=fontsize,
    )
    ### Diversions
    axin.plot([0.05, 0.15], [0.63, 0.63], color="darkmagenta", lw=2, ls=":")
    axin.annotate(
        "Interbasin Transfer",
        xy=(0.18, 0.63),
        ha="left",
        va="center",
        color="darkmagenta",
        fontweight="bold",
        fontsize=fontsize,
    )
    ### Minimum flow targets
    axin.scatter(
        [0.1], [0.53], color="mediumseagreen", edgecolor="k", s=200, marker="*"
    )
    axin.annotate(
        "Flow Target",
        xy=(0.18, 0.53),
        ha="left",
        va="center",
        color="mediumseagreen",
        fontweight="bold",
        fontsize=fontsize,
    )
    ### NYC Reservoirs
    axin.scatter([0.1], [0.43], color="firebrick", edgecolor="k", s=100)
    axin.annotate(
        "NYC Reservoir",
        xy=(0.18, 0.43),
        ha="left",
        va="center",
        color="firebrick",
        fontweight="bold",
        fontsize=fontsize,
    )
    ### Non-NYC Reservoirs
    axin.scatter([0.1], [0.33], color="sandybrown", edgecolor="k", s=100)
    axin.annotate(
        "Non-NYC Reservoir",
        xy=(0.18, 0.33),
        ha="left",
        va="center",
        color="k",
        fontsize=fontsize,
    )
    ### marker size for reservoirs
    # axin.annotate('Reservoir Capacity', xy=(0.05, 0.3), ha='left', va='center', color='k', fontsize=fontsize)
    axin.scatter([0.15], [0.18], color="0.5", edgecolor="k", s=50 + 10 / 2)
    axin.scatter([0.45], [0.18], color="0.5", edgecolor="k", s=50 + 200 / 2)
    axin.scatter([0.8], [0.18], color="0.5", edgecolor="k", s=50 + 530 / 2)
    axin.annotate(
        "10", xy=(0.15, 0.05), ha="center", va="center", color="k", fontsize=fontsize
    )
    axin.annotate(
        "200", xy=(0.45, 0.05), ha="center", va="center", color="k", fontsize=fontsize
    )
    axin.annotate(
        "530 MCM",
        xy=(0.8, 0.05),
        ha="center",
        va="center",
        color="k",
        fontsize=fontsize,
    )
    ### clean up
    axin.set_xticks([])
    axin.set_yticks([])
    axin.patch.set_alpha(0.9)

    # ### basemap - this is slow and breaks sometimes, if so just try later
    if use_basemap:
        cx.add_basemap(ax=ax, alpha=0.5, attribution_size=6)
        figname = f"{fig_dir}/static_map_withbasemap"
    else:
        figname = f"{fig_dir}static_map"

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(rf"{figname}.svg", bbox_inches="tight", dpi=dpi)
    plt.savefig(rf"{figname}.png", bbox_inches="tight", dpi=dpi)
    return 


if __name__ == "__main__":
    
    # make and save map
    make_DRB_map()