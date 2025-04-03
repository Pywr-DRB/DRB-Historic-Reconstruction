
from methods.pywr_drb_node_data import upstream_nodes_dict, downstream_node_lags
from methods.pywr_drb_node_data import obs_pub_site_matches

def subtract_upstream_catchment_inflows(inflows):
    """
    Subtracts upstream catchment inflows from the input inflows timeseries.

    Inflow timeseries are cumulative. For each downstream node, this function subtracts the flow into all upstream nodes so
    that it represents only the direct catchment inflows into this node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows subtracted.
    """
    inflows = inflows.copy()
    for node, upstreams in upstream_nodes_dict.items():
        for upstream in upstreams:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows.loc[inflows.index[lag:], node] -= inflows.loc[inflows.index[:-lag], upstream].values
                ### subtract same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows.loc[inflows.index[:lag], node] -= inflows.loc[inflows.index[:lag], upstream].values
            else:
                inflows[node] -= inflows[upstream]

        ### if catchment inflow is negative after subtracting upstream, set to 0
        inflows.loc[inflows[node] < 0, node] = 0

        ### delTrenton node should have zero catchment inflow because coincident with DRCanal
        ### -> make sure that is still so after subtraction process
        inflows['delTrenton'] *= 0.
    return inflows


def add_upstream_catchment_inflows(inflows):
    """
    Adds upstream catchment inflows to get cumulative flow at downstream nodes. THis is inverse of subtract_upstream_catchment_inflows()

    Inflow timeseries are cumulative. For each downstream node, this function adds the flow into all upstream nodes so
    that it represents cumulative inflows into the downstream node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows added.
    """
    ### loop over upstream_nodes_dict in reverse direction to avoid double counting
    for node in list(upstream_nodes_dict.keys())[::-1]:
        for upstream in upstream_nodes_dict[node]:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows.loc[inflows.index[lag:], node] += inflows.loc[inflows.index[:-lag], upstream].values
                ### add same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows.loc[inflows.index[:lag], node] += inflows.loc[inflows.index[:lag], upstream].values
            else:
                inflows[node] += inflows[upstream]

        ### if catchment inflow is negative after adding upstream, set to 0 (note: this shouldnt happen)
        inflows.loc[inflows[node] < 0, node] = 0
    return inflows


def aggregate_node_flows(df):
    """Sums flows from different sites in site_matches for each node.

    Args:
        df (pandas.DataFrame): Reconstructed site flows, for each gauge and or PUB location.

    Returns:
        pandas.DataFrame: Reconstructed flows aggregated for Pywr-DRB nodes
    """
    for node, sites in obs_pub_site_matches.items():
        if sites:
            df.loc[:,node] = df.loc[:, sites].sum(axis=1)    
    return df