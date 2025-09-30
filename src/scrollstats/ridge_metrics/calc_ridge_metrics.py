from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from .data_extractors import BendDataExtractor


def calculate_ridge_metrics(
    in_transects: gpd.GeoDataFrame | Path,
    in_ridges: gpd.GeoDataFrame | Path,
    in_bin_raster: rasterio.io.DatasetReader | Path | None = None,
    in_dem: rasterio.io.DatasetReader | Path | None = None,
    in_packets: gpd.GeoDataFrame | Path | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Main function to calculate scroll metrics.

    If in_packets is specified, then all metrics for the rich_transects will be calculated for the transect fragment within each packet.

    All arguments can be provided as a file path or in-memory object (vector: GeoDataFrame, raster: rasterio dataset)
    """

    # Check if args are paths or objects in memory
    if isinstance(in_transects, gpd.GeoDataFrame):
        transects = in_transects.copy()
    else:
        transects = gpd.read_file(in_transects)

    if isinstance(in_ridges, gpd.GeoDataFrame):
        ridges = in_ridges
    else:
        ridges = gpd.read_file(in_ridges)

    if isinstance(in_bin_raster, rasterio.io.DatasetReader) or in_bin_raster is None:
        bin_raster = in_bin_raster
    else:
        bin_raster = rasterio.open(in_bin_raster)

    if isinstance(in_dem, rasterio.io.DatasetReader) or in_dem is None:
        dem = in_dem
    else:
        dem = rasterio.open(in_dem)

    if isinstance(in_packets, gpd.GeoDataFrame) or in_packets is None:
        packets = in_packets
    else:
        packets = gpd.read_file(in_packets)

    # If packets are provided, create intersection with packets and return MultiIndex DataFrame
    if isinstance(packets, gpd.GeoDataFrame):
        transects = transects.overlay(
            packets, how="intersection"
        )  # .set_index(["packet_id", "transect_id"])

    # Calculate sampled ridge metrics
    bde = BendDataExtractor(transects, bin_raster, dem, ridges)
    transects = bde.rich_transects
    itx = bde.itx_metrics

    return transects, itx
