import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine

from config import DB_PW


class BendDataset:
    """
    Convenience class to get all relevant data for a bend from the database.
    Returns the geometry in a projected CRS if the `proj` argument is set to True in the `.get_[geom_name]()` method
    """
    def __init__(self, bend_id):
        self.bend_id = bend_id
        self.river_abbrev = self.bend_id[:3]
        self.db_engine = self.create_db_engine()
        self.river_id = self.get_river_id()
        self.local_utm = None

    def create_db_engine(self):
        engine = create_engine(f"postgresql://postgres:{DB_PW}@localhost:5432/scroll")
        return engine
    
    def get_river_id(self):
        """Get the river id from the rivers table to use for other queries"""
        river = pd.read_sql(f"SELECT river_id FROM rivers WHERE river_abbrev='{self.river_abbrev}'", self.db_engine)
        river_id = river.loc[0, "river_id"]
        return river_id
    
    def get_local_utm(self, feature):
        """Get the local utm zone for a given feature"""
        if not self.local_utm:
            if self.bend_id.startswith("LBR"):
                crs = 'EPSG:32140'
            else:
                crs = feature.estimate_utm_crs()
            self.local_utm = crs
        
        return self.local_utm

    def get_ridges(self, proj=False):
        """Get the ridges from self.db_engine for the given bend id"""
        ridges = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM ridges WHERE bend_id='{self.bend_id}'", self.db_engine, geom_col="geometry")

        if proj:
            ridges = ridges.to_crs(self.get_local_utm(ridges))

        return ridges
    
    def get_transects(self, proj=False):
        """Get the transects from self.db_engine for the given bend id"""
        transects = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM transects WHERE bend_id='{self.bend_id}'", self.db_engine, geom_col="geometry")

        if proj:
            transects = transects.to_crs(self.get_local_utm(transects))
            
        return transects
    
    def get_packets(self, proj=False):
        """Get the packets from self.db_engine for the given bend id"""
        packets = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM packets WHERE bend_id='{self.bend_id}'", self.db_engine, geom_col="geometry")

        if proj:
            packets = packets.to_crs(self.get_local_utm(packets))
            
        return packets
    
    def get_bend(self, proj=False):
        """Get the bend from self.db_engine for the given bend id"""
        bend = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM bends WHERE bend_id='{self.bend_id}'", self.db_engine, geom_col="geometry")

        if proj:
            bend = bend.to_crs(self.get_local_utm(bend))
            
        return bend
    
    def get_clipper(self, proj=False):
        """
        Get the clipper from self.db_engine for the given bend id. 
        The clipper for a bend is a realtively simple polygon to clip a greater-than bend length centerline to a bend-length.
        """
        clipper = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM clippers WHERE bend_id='{self.bend_id}'", self.db_engine, geom_col="geometry")

        if proj:
            clipper = clipper.to_crs(self.get_local_utm(clipper))
            
        return clipper

    def get_centerline(self, proj=False):
        """Get the centerline from self.db_engine for the given bend id"""
        centerline = gpd.GeoDataFrame.from_postgis(f"SELECT * FROM centerlines WHERE river_id='{self.river_id}'", self.db_engine, geom_col="geometry")

        if self.bend_id.startswith("LBR"):
            clipper = self.get_clipper()
            centerline = centerline.clip(clipper)

        if proj:
            centerline = centerline.to_crs(self.get_local_utm(centerline))

        return centerline
