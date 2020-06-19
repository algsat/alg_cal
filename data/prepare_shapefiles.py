# IMPORT
import pandas as pd
import geopandas as gpd

# INPUT
s2_paths_file = 's2_paths.shp'
s2_track_acqday_file = 's2_track_acqday.csv'

# BODY
s2_track_acqday = (
    pd
    .read_csv(s2_track_acqday_file)
    .astype({'track': 'int64'})
    .set_index('track')
)
s2_paths = (
    gpd
    .read_file(s2_paths_file)
    .astype({'Track': 'int64'})
    .join(s2_track_acqday, on='Track')
)

# EXPORT
s2_paths.to_file('s2_paths_acqday.shp')
