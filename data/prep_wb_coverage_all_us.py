# IMPORT
import numpy as np
import pandas as pd
import geopandas as gpd

# INPUT
old_wb_file = 'wb_coverage.csv'
path_comb_file = 's2_and_l8_path_combinations.csv'
l8_paths_file = 'l8_paths.shp'
s2_acqday_file = 's2_track_acqday.csv'

# BODY
# get acqday dictionaries
# l8
l8_acq_df = (
    gpd
    .read_file(l8_paths_file)
    .loc[:, ['path', 'acqdayl8']]
    .astype('int64')
    .drop_duplicates()
)
l8_acq_dict = {str(path): str(acq) for path,acq in zip(l8_acq_df.path, l8_acq_df.acqdayl8)}
l8_acq_dict[''] = ''
# s2
s2_acq_df = (
    pd
    .read_csv(s2_acqday_file)
    .astype('int64')
)
s2_acq_dict = {str(track): str(acq) for track,acq in zip(s2_acq_df.track, s2_acq_df.acqday)}
s2_acq_dict[''] = ''
# format path combo file
path_combo_df = (
    pd
    .read_csv(path_comb_file)
    .fillna('')
)
s2_track_id_list = []
s2_acq_id_list = []
l8_path_id_list = []
l8_acq_id_list = []
# loop over rows
for i in range(len(path_combo_df)):
    # get values from df
    # s2
    s2_full = path_combo_df.s2_full.values[i].replace('.0', '')
    s2_part = path_combo_df.s2_partial.values[i].replace('.0', '')
    # l8
    l8_full = path_combo_df.l8_full.values[i].replace('.0', '')
    l8_part = path_combo_df.l8_partial.values[i].replace('.0', '')
    # get acquisition dates from paths
    # s2
    s2_full_acq_list = list(np.sort([s2_acq_dict[pth] for pth in s2_full.split(';')]))
    s2_full_acq = '_'.join(s2_full_acq_list)
    s2_part_acq_list = list(np.sort([s2_acq_dict[pth] for pth in s2_part.split(';')]))
    s2_part_acq = 'p_'.join(s2_part_acq_list) + 'p'
    if s2_part_acq == 'p':
        s2_acq = s2_full_acq
        s2_paths = s2_full.replace(';', '_')
    elif s2_full_acq == '':
        s2_acq = s2_part_acq
        s2_paths = s2_part.replace(';', 'p_') + 'p'
    else:
        s2_acq = s2_full_acq + '_' + s2_part_acq
        s2_paths = (
            s2_full.replace(';', '_') + '_'
            + s2_part.replace(';', 'p_') + 'p'
        )
    # l8
    l8_full_acq_list = list(np.sort([l8_acq_dict[pth] for pth in l8_full.split(';')]))
    l8_full_acq = '_'.join(l8_full_acq_list)
    l8_part_acq_list = list(np.sort([l8_acq_dict[pth] for pth in l8_part.split(';')]))
    l8_part_acq = 'p_'.join(l8_part_acq_list) + 'p'
    if l8_part_acq == 'p':
        l8_acq = l8_full_acq
        l8_paths = l8_full.replace(';', '_')
    elif l8_full_acq == '':
        l8_acq = l8_part_acq
        l8_paths = l8_part.replace(';', 'p_') + 'p'
    else:
        l8_acq = l8_full_acq + '_' + l8_part_acq
        l8_paths = (
            l8_full.replace(';', '_') + '_'
            + l8_part.replace(';', 'p_') + 'p'
        )
    # add to lists
    # s2
    s2_track_id_list.append(s2_paths)
    s2_acq_id_list.append(s2_acq)
    # l8
    l8_path_id_list.append(l8_paths)
    l8_acq_id_list.append(l8_acq)
# add dataframe
wb_cov = pd.DataFrame({
    's2_id': s2_track_id_list
    , 's2_acq_id': s2_acq_id_list
    , 'l8_id': l8_path_id_list
    , 'l8_acq_id': l8_acq_id_list
})

# EXPORT
wb_cov.drop_duplicates().to_csv('wb_coverage-all_us.csv', index=False)

