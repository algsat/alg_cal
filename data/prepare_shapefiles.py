# IMPORT
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shutil import copyfile
from zipfile import ZipFile
from glob import glob


# INPUT
s2_paths_file = 's2_paths.shp'
s2_track_acqday_file = 's2_track_acqday.csv'
s2_overlap_file = 's2_over_raw.shp'
l8_overlap_file = 'l8_over_raw.shp'

# BODY
# join my s2 acqday definitions with the s2 path shapefile
s2_paths = (
    gpd
    .read_file(s2_paths_file)
    .rename(columns={'Track': 'track'})
    .astype({'track': 'int64'})
)
s2_track_acqday = (
    pd
    .read_csv(s2_track_acqday_file)
    .astype({'track': 'int64'})
    .set_index('track')
)
s2_paths_acqday = (
    s2_paths
    .join(s2_track_acqday, on='track')
    .loc[:, ['track', 'acqday', 'geometry']]
)

# clean path overlap files
s2_over = (
    gpd
    .read_file(s2_overlap_file)
    .dropna(subset=['geometry'])
    .assign(tracks = lambda x: [id.replace('|', ',') for id in x.ID])
    .assign(n_tracks = lambda x: [len(id.split(',')) for id in x.tracks])
    .loc[:, ['tracks', 'n_tracks', 'geometry']]
)
l8_over = (
    gpd
    .read_file(l8_overlap_file)
    .dropna(subset=['geometry'])
    .assign(paths = lambda x: [','.join([p.split('.')[0] for p in id.split('|')]) for id in x.ID])
    .assign(n_paths = lambda x: [len(id.split(',')) for id in x.paths])
    .loc[:, ['paths', 'n_paths', 'geometry']]
)

# EXPORT
s2_paths_acqday.to_file('s2_paths_acqday.shp')
s2_over.to_file('s2_over.shp')
l8_over.to_file('l8_over.shp')
# copy prj files
copyfile('s2_paths.prj', 's2_paths_acqday.prj')
copyfile('s2_over_raw.prj', 's2_over.prj')
copyfile('l8_over_raw.prj', 'l8_over.prj')
