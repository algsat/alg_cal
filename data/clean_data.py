# IMPORT
import pandas as pd
import geopandas as gpd

# INPUT
s2_dates_raw_file = 's2_track_start_dates.csv'
l8_l7_acq_date_file = 'l8_l7_acq_date_start.csv'

# BODY
# clean s2
s2_dates_raw = pd.read_csv(s2_dates_raw_file)
s2_dates = (
    s2_dates_raw
    .drop(['s2a'], axis=1)
    .rename(columns={'s2b': 'date'})
)
s2b_acqday_date = (
    s2_dates
    .date
    .drop_duplicates()
    .sort_values()
    .to_frame()
    .assign(acqday = lambda x: [1, 2, 3, 4, 5] * 2)
    .loc[:, ['acqday', 'date']]
)
s2_acqday_date = s2b_acqday_date.iloc[:5, :]
s2_track_acqday = (
    s2_dates
    .set_index('date')
    .join(s2b_acqday_date.set_index('date'))
)
# clean l8
l8_acqday_date = (
    pd
    .read_csv(l8_l7_acq_date_file)
    .rename(columns={'l8': 'date'})
    .astype({'acqday': 'int64'})
    .drop(['l7'], axis=1)
)
# l8_dates = (
#     gpd
#     .read_file(l8_path_file)
#     .rename(columns={'acqdayl8': 'acqday'})
#     .loc[:, ['path', 'acqday']]
#     .astype({'path': 'int64', 'acqday': 'int64'})
#     .set_index('acqday')
#     .join(l8_acqday_date)
# )

# EXPORT
# s2_dates.to_csv('s2_dates.csv', index=False)
# l8_dates.to_csv('l8_dates.csv', index=False)
l8_acqday_date.to_csv('l8_acqday_dates.csv', index=False)
s2_acqday_date.to_csv('s2_acqday_dates.csv', index=False)
s2_track_acqday.to_csv('s2_track_acqday.csv', index=False)
