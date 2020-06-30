# IMPORT
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shutil import copyfile
from zipfile import ZipFile
from glob import glob


# INPUT
s2_paths_file = 's2_paths.shp'
s2_track_acqday_file = 's2_track_acqday.csv'
s2_overlap_file = 's2_over_raw.shp'
l8_paths_file = 'l8_paths.shp'
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
l8_path_acq = (
        gpd
        .read_file(l8_paths_file)
        .rename(columns={'acqdayl8': 'acqday'})
        .astype({'path': 'int64'})
        .loc[:, ['path', 'acqday']]
    )
l8_path_acq_dict = {str(pth): str(acq) for pth, acq in l8_path_acq.itertuples(index=False, name=None)}
l8_path_acq_dict[''] = ''
# s2_track_acq = pd.read_csv(s2_track_acq_file)
s2_track_acq_dict = {trk: acq for trk, acq in s2_track_acqday.reset_index().itertuples(index=False, name=None)}
# clean path overlap files
clip_poly = Polygon([(-180, 75), (180, 75), (180, -75), (-180, -75), (-180, 75)])
s2_over_raw = (
    gpd
    .read_file(s2_overlap_file)
    .dropna(subset=['geometry'])
    .assign(tracks = lambda x: [id.replace('|', ',') for id in x.ID])
    .assign(acq_id = lambda x: [str(sorted(set([int(s2_track_acq_dict[int(id)]) for id in ids.split(',')])))[1: -1].replace(', ', '_') for ids in x.tracks])
    .assign(n_tracks = lambda x: [len(id.split(',')) for id in x.tracks])
    .query('n_tracks<5')
    .loc[:, ['tracks', 'n_tracks', 'acq_id', 'geometry']]
    .assign(geometry = lambda x: x.buffer(0))
)
l8_over_raw = (
    gpd
    .read_file(l8_overlap_file)
    .dropna(subset=['geometry'])
    .assign(paths = lambda x: [','.join([p.split('.')[0] for p in id.split('|')]) for id in x.ID])
    .assign(acq_id = lambda x: [str(sorted(set([l8_path_acq_dict[id] for id in ids.split(',')])))[1: -1].replace(', ', '_').replace("'", '') for ids in x.paths])
    .assign(n_paths = lambda x: [len(id.split(',')) for id in x.paths])
    .query('n_paths<5')
    .loc[:, ['paths', 'n_paths', 'acq_id', 'geometry']]
    .assign(geometry = lambda x: x.buffer(0))
)
clip_poly_fix = Polygon([(-179, 75), (179, 75), (179, -75), (-179, -75), (-179, 75)])
fix_tracks = [
    "140,40,83"
    ,"47,32,133"
    ,"97,140"
    ,"97,140,40,83"
    ,"12"
    ,"88,131"
    ,"103"
    ,"103,60"
]
s2_over = gpd.GeoDataFrame(
    pd.concat([
        gpd.clip(s2_over_raw.query(f'tracks in {fix_tracks}'), clip_poly_fix)
        , gpd.clip(s2_over_raw.query(f'tracks not in {fix_tracks}'), clip_poly)
    ])
).assign(geometry = lambda x: [MultiPolygon([gg for gg in geom if gg.type == 'Polygon']) if geom.type=='GeometryCollection' else geom for geom in x.geometry])

l8_over = (
    gpd
    .clip(l8_over_raw, clip_poly)
    .assign(geometry = lambda x: [MultiPolygon([gg for gg in geom if gg.type == 'Polygon']) if geom.type=='GeometryCollection' else geom for geom in x.geometry])
    .assign(acq_id = lambda x: ['_'.join([str(i) for i in sorted([int(id) for id in ids.split('_')])]) for ids in x.acq_id])
)

# EXPORT
s2_paths_acqday.to_file('s2_paths_acqday.shp')
s2_over.to_file('s2_over.shp')
l8_over.to_file('l8_over.shp')
# copy prj files
copyfile('s2_paths.prj', 's2_paths_acqday.prj')
copyfile('s2_over_raw.prj', 's2_over.prj')
copyfile('l8_over_raw.prj', 'l8_over.prj')
