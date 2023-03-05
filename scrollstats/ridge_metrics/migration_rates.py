import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import *
from shapely.ops import split


def calc_dist_along_line(line: LineString) -> np.array:
    """Calculate an array of distance along a given line """

    x, y = line.xy

    dx = np.ediff1d(x)
    dy = np.ediff1d(y)

    arc_length = np.insert(np.sqrt(dx ** 2 + dy ** 2), 0, 0)

    return np.cumsum(arc_length)


def create_ridge_columns(transects, ridges):
    """Create an empty column for each ridge"""
    
    out_transects = transects.copy()
    
    for i in ridges["ridge_id"]:
        out_transects[i] = np.nan
        
    return out_transects


def create_vert_table(line, ridges, crs):
    """Create a table for all vertices along a trasect relating the transect distance to ridge id"""
    
    # Create dataframe for all verts on the line
    v_id = [f"v_{i}" for i, v in enumerate(line.coords)]
    points = [Point(i) for i in line.coords[:]]
    verts = gpd.GeoDataFrame({'v_id': v_id}, 
                             geometry=points,
                             crs=crs)  
    
    # Relate vertices to ridges with a spatial join
    verts = verts.sjoin_nearest(ridges)
    
    # Join along-transect distances
    verts['along_dist'] = calc_dist_along_line(line)
    
    # `sjoin_nearest` can assign centerline vertex with first ridge.
    # Results in the first ridge after the centerline being assigned 2 different distance values
    # So, filter for distance values greater than zero
    return verts[verts.along_dist > 0]


def verts_for_bend(transects, ridges):
    """Create a vert table for each transect in the bend"""
    vert_table_list = []
    
    # Loop through gdf rows to create vert table
    for i, row in transects.iterrows():
        vert_table = create_vert_table(row["geometry"], ridges, transects.crs)
        
        vert_table["transect_id"] = row["transect_id"]
        vert_table_list.append(vert_table)
    
    
    verts_gdf = pd.concat(vert_table_list)
    
    return verts_gdf
    

def calc_rel_mig_rates(transects, ridges):
    """Calculate the along transect distance for every vertex on every transect"""

    # Create a column for each ridge
    out_transects = create_ridge_columns(transects, ridges)
    
    # Relate ridges to transect vertices for the entire bend
    bend_vert_gdf = verts_for_bend(transects, ridges)
    
    # Assign migration rate values for every ridge, for every row in transects
    for i in out_transects["transect_id"]:
    
        # Get rows relevant to the transect
        transect_verts = bend_vert_gdf.loc[bend_vert_gdf.transect_id ==i]

        # Get ridge ids
        ridge_ids = transect_verts['ridge_id']

        # Get distances
        along_dists = transect_verts['along_dist'].values

        # Set ridge distances
        out_transects.loc[out_transects.transect_id==i, ridge_ids] = along_dists
    
    return out_transects


def split_transects(transects, ridges, ridge_id):
    """
    Split transects on the intersected ridge for plotting.
    
    This function takes advantage of the fact that all transects are created inwardly from the centerline.
    So, the first part of the split line will always be "before" the splitting ridge
    """
    
    # Subset the transects that actually intersect the ridge
    itx_transects = transects[~transects[ridge_id].isna()].copy()
    
    itx_transects.geometry = itx_transects.geometry.apply(lambda x: split(x, ridges.loc[ridge_id].geometry).geoms[0])
    
    return itx_transects


def norm_rel_mig_rate(transects, ridge_id):
    """Normalize the relative migration rate to the lowest value and give a unique name"""
    
    norm_rate = transects[ridge_id] / transects[ridge_id].min()
    norm_rate.name = f"{ridge_id}_norm"
    
    return norm_rate
