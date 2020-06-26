# IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns

# INPUT
# files
s2_acq_dates_file = 'data/s2_acqday_dates.csv'
l8_acq_dates_file = 'data/l8_acqday_dates.csv'
wb_coverage_file = 'data/wb_coverage.csv'

# settings
years = [2020, 2021]
colors = ['#ffffff', '#663399', '#003366', '#006400', '#ffcccb', '#ffae42', '#98fb98', '#ffffbf', '#d9d9d9', '#d3bda6']

# FUNCTIONS
def date_series(full_starts, return_period, partial_starts=None):
    if isinstance(full_starts, str):
        full_starts = [full_starts]
    year = full_starts[0][:4]
    idx = pd.date_range(f'{year}-01-01', end=f'{year}-12-31', freq='D')
    pass_series = pd.Series(0, index=idx)
    final_date = f'{year}-12-31'
    for full_start in full_starts:
        full_passes = pd.date_range(full_start, end=final_date, freq=f'{return_period}D')
        pass_series[full_passes] = 1.2
    if partial_starts is not None:
        if isinstance(partial_starts, str):
            partial_starts = [partial_starts]
        for partial_start in partial_starts:
            partial_passes = pd.date_range(partial_start, end=final_date, freq=f'{return_period}D')
            pass_series[partial_passes] = 0.45
    return pass_series

# Taken from SO_tourist answer at: https://stackoverflow.com/questions/32485907/matplotlib-and-numpy-create-a-calendar-heatmap
def split_months(df, year):
    """
    Take a df, slice by year, and produce a list of months,
    where each month is a 2D array in the shape of the calendar
    :param df: dataframe or series
    :return: matrix for daily values and numerals
    """
    df = df[df.index.year == year]


    # Empty matrices
    a = np.empty((6, 7))
    a[:] = np.nan

    day_nums = {m:np.copy(a) for m in range(1,13)}  # matrix for day numbers
    day_vals = {m:np.copy(a) for m in range(1,13)}  # matrix for day values

    # Logic to shape datetimes to matrices in calendar layout
    for d in df.iteritems():  # use iterrows if you have a DataFrame

        day = d[0].day
        month = d[0].month
        # col = d[0].dayofweek
        # todo: switch to sunday as beginning of week???
        col0 = d[0].dayofweek + 1
        if col0==7:
            col=0
        else:
            col=col0

        if d[0].is_month_start:
            row = 0

        day_nums[month][row, col] = day  # day number (0-31)
        day_vals[month][row, col] = d[1] # day value (the heatmap data)

        if col == 6:
            row += 1

    return day_nums, day_vals

# cmap1 = ListedColormap(['#ffffff', '#663399', '#003366', '#006400', '#60ff0000', '#60ff6500', '#60a5d610', '#60ffff00', '#60909090', '#90654321'], name='custom1')

# Taken from SO_tourist answer at: https://stackoverflow.com/questions/32485907/matplotlib-and-numpy-create-a-calendar-heatmap

from matplotlib.lines import Line2D
from matplotlib import gridspec
def create_year_calendar(day_nums, day_vals, file_prefix='example', color_dict=None, full_paths=None, part_paths=None, leg_title='track'):
    # fig, axes = plt.subplots(5, 3, figsize=(6.5, 9))
    fig = plt.figure(figsize=(6.5, 9))
    spec = gridspec.GridSpec(ncols=3, nrows=5, height_ratios=[2, 3, 3, 3, 3])
    axes = [fig.add_subplot(spec[i]) for i in range(3, 15)]
    ax_leg = fig.add_subplot(spec[2])
    # simplify legend axis
    ax_leg.tick_params(axis=u'both', which=u'both', length=0)  # remove tick marks
    ax_leg.xaxis.tick_top()
    # Despine
    for edge in ['left', 'right', 'bottom', 'top']:
        ax_leg.spines[edge].set_color('#FFFFFF')
    ax_leg.set_yticklabels([])
    ax_leg.set_xticklabels([])

    if (color_dict is not None) and (full_paths is not None):
        legend_elements = []
        i = 1
        for path in full_paths:
            elem = Line2D([0], [0], marker='s', color='w', markerfacecolor=color_dict[i], label=str(path), markersize=15)
            legend_elements.append(elem)
            i+=1
        i = 4
        for path in part_paths:
            elem = Line2D([0], [0], marker='s', color='w', markerfacecolor=color_dict[i], label=f'{path} (partial)', markersize=15)
            legend_elements.append(elem)
            i+=1
        ax_leg.legend(handles=legend_elements, loc=9, frameon=False, title=f'Explanation\n     {leg_title}', borderpad=0, borderaxespad=0, ncol=int(np.ceil(len(legend_elements) / 3)) )

    for i in range(0, 12):
        axs = axes[i]
        # get colormap
        if color_dict is None:
            axs.imshow(day_vals[i+1], cmap='Purples')
        else:
            un_vals = np.array([val for val in np.sort(np.unique(day_vals[i+1])) if not np.isnan(val)])
            color_list = [color_dict[val] for val in un_vals]
            # cmap = sns.color_palette(colors, n_colors=len(colors))
            # cmap = mpl.colors.Colormap(colors)
            cmap = ListedColormap(color_list)
            norm = mpl.colors.BoundaryNorm([-0.5] + list(un_vals + 0.5) ,cmap.N)
            # print(cmap)
            # axs.imshow(day_vals[i+1], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            axs.imshow(day_vals[i+1], cmap=cmap, norm=norm)
        # axs.imshow(day_vals[i+1], cmap=cmap, vmin=-0.05, vmax=1.25)  # heatmap
        # axs.set_title(month_names[i], fontsize=20)
        axs.set_title(month_names[i], fontsize=10)

        # Labels
        axs.set_xticks(np.arange(len(days)))
        # axs.set_xticklabels(days, fontsize=14, fontweight='bold', color='#555555')
        axs.set_xticklabels(days, fontsize=8, fontweight='bold', color='#555555')
        axs.set_yticklabels([])

        # Tick marks
        axs.tick_params(axis=u'both', which=u'both', length=0)  # remove tick marks
        axs.xaxis.tick_top()

        # Modify tick locations for proper grid placement
        axs.set_xticks(np.arange(-.5, 6, 1), minor=True)
        axs.set_yticks(np.arange(-.5, 5, 1), minor=True)
        axs.grid(which='minor', color='w', linestyle='-', linewidth=2.1)

        # Despine
        for edge in ['left', 'right', 'bottom', 'top']:
            axs.spines[edge].set_color('#FFFFFF')

        # Annotate
        for w in range(len(weeks)):
            for d in range(len(days)):
                day_val = day_vals[i+1][w, d]
                day_num = day_nums[i+1][w, d]
                # If day number is a valid calendar day, add an annotation
                if not np.isnan(day_num):
                    # axs.text(d+0.45, w-0.31, f"{day_num:0.0f}",
                    axs.text(d+0.42, w-0.27, f"{day_num:0.0f}",
                             ha="right", va="center",
                             # fontsize=14, color="#003333", alpha=0.8)  # day
                             fontsize=6, color="#003333", alpha=0.8)  # day

                # Aesthetic background for calendar day number
                patch_coords = ((d-0.4, w-0.5),
                                (d+0.5, w-0.5),
                                (d+0.5, w+0.4))

                triangle = Polygon(patch_coords, fc='w', alpha=0.7)
                axs.add_artist(triangle)

    # Final adjustments
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
    fig.tight_layout()
    # Save to file
    outname = f'images/{file_prefix}.pdf'
    plt.savefig(outname, dpi=120)

# def create_year_calendar(day_nums, day_vals, file_prefix='example', color_dict=None):
#     for i in range(0, 12):
#         # get colormap
#         if color_dict in None:
#             cmap='Purples'
#         else:
#             un_vals = np.sort(np.unique(day_vals[i+1]))
#             color_list = [color_dict[val] for val in un_vals]
#             cmap = ListedColormap(color_list)
#         fig, axs = plt.subplots(1, 1, figsize=(5, 4))
#         axs.imshow(day_vals[i+1], cmap='Purples', vmin=-0.05, vmax=1.25)  # heatmap
#         axs.set_title(month_names[i], fontsize=20)
#
#         # Labels
#         axs.set_xticks(np.arange(len(days)))
#         axs.set_xticklabels(days, fontsize=14, fontweight='bold', color='#555555')
#         axs.set_yticklabels([])
#
#         # Tick marks
#         axs.tick_params(axis=u'both', which=u'both', length=0)  # remove tick marks
#         axs.xaxis.tick_top()
#
#         # Modify tick locations for proper grid placement
#         axs.set_xticks(np.arange(-.5, 6, 1), minor=True)
#         axs.set_yticks(np.arange(-.5, 5, 1), minor=True)
#         axs.grid(which='minor', color='w', linestyle='-', linewidth=2.1)
#
#         # Despine
#         for edge in ['left', 'right', 'bottom', 'top']:
#             axs.spines[edge].set_color('#FFFFFF')
#
#         # Annotate
#         for w in range(len(weeks)):
#             for d in range(len(days)):
#                 day_val = day_vals[i+1][w, d]
#                 day_num = day_nums[i+1][w, d]
#                 # If day number is a valid calendar day, add an annotation
#                 if not np.isnan(day_num):
#                     # axs.text(d+0.45, w-0.31, f"{day_num:0.0f}",
#                     axs.text(d+0.42, w-0.27, f"{day_num:0.0f}",
#                              ha="right", va="center",
#                              fontsize=14, color="#003333", alpha=0.8)  # day
#
#                 # Aesthetic background for calendar day number
#                 patch_coords = ((d-0.4, w-0.5),
#                                 (d+0.5, w-0.5),
#                                 (d+0.5, w+0.4))
#
#                 triangle = Polygon(patch_coords, fc='w', alpha=0.7)
#                 axs.add_artist(triangle)
#
#         # Final adjustments
#         plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.04)
#         # plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
#         fig.tight_layout()
#
#         # Save to file
#         outname = f'images/{file_prefix}{i+1:02}.png'
#         plt.savefig(outname, dpi=120)

def first_pass_of_year(start_date, year, interval):
    year_start_dt = pd.to_datetime(f'{year}-01-01')
    offset = (pd.to_datetime(start_date) - year_start_dt).days % interval
    first_dt = year_start_dt + pd.to_timedelta(offset, 'D')
    return first_dt.strftime('%Y-%m-%d')

# BODY
# def main():
if 1:
    # load files
    wb_coverage = pd.read_csv(wb_coverage_file, dtype={x: 'str' for x in ['s2_full', 's2_part', 's2_id', 's2_acq_id', 'l8_full', 'l8_part', 'l8_id', 'l8_acq_id']})
    l8_dates = pd.read_csv(l8_acq_dates_file)
    s2_dates = pd.read_csv(s2_acq_dates_file)
    s2_ids = wb_coverage.loc[:, ['s2_id', 's2_acq_id']].drop_duplicates()
    l8_ids = wb_coverage.loc[:, ['l8_id', 'l8_acq_id']].drop_duplicates()
    # define some settings
    intervals = {'s2': 5, 'l8': 16}
    start_dates = {'s2': s2_dates, 'l8': l8_dates}
    cov_ids = {'s2': s2_ids, 'l8': l8_ids}
    col_names = {'s2': 'track', 'l8': 'path'}
    weeks = [1, 2, 3, 4, 5, 6]
    # days = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
    # todo: switch to sunday as beginning of week???
    days = ['S', 'M', 'T', 'W', 'T', 'F', 'S']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                   'September', 'October', 'November', 'December']
    color_dict = {i: col for i,col in enumerate(colors)}
    # loop over satellites
    for sat in ['s2', 'l8']:
        pass_int = intervals[sat]
        ids = cov_ids[sat]
        colname = col_names[sat]
        # loop over years
        for year in years:
            yr_starts = (
                start_dates[sat]
                .assign(start_date = lambda x: [first_pass_of_year(date, year, pass_int) for date in x.date])
            )
            yr_dates = pd.date_range(f'{year}-01-01', end=f'{year}-12-31', freq='D')
            # loop over ids
            for id, acq_id in ids.itertuples(index=False, name=None):
                acqs = acq_id.split('_')
                full_acqs = [int(x) for x in acqs if 'p' not in x]
                part_acqs = [int(x[:-1]) for x in acqs if 'p' in x]
                paths = id.split('_')
                full_paths = [int(x) for x in paths if 'p' not in x]
                part_paths = [int(x[:-1]) for x in paths if 'p' in x]
                full_start_dates = yr_starts.query(f'acqday in {full_acqs}').start_date.values.tolist()
                part_start_dates = yr_starts.query(f'acqday in {part_acqs}').start_date.values.tolist()
                basename = f'{sat}_{id}_{year}'
                # make csv of dates
                df_list = []
                i=1
                for start, path in zip(full_start_dates, full_paths):
                    dates = pd.date_range(start, end=f'{year}-12-31', freq=f'{pass_int}D')
                    i_df = pd.DataFrame({'date': dates, 'path': path, 'coverage': 'full', 'colorid': i})
                    df_list.append(i_df)
                    i+=1
                i=4
                for start, path in zip(part_start_dates, part_paths):
                    dates = pd.date_range(start, end=f'{year}-12-31', freq=f'{pass_int}D')
                    i_df = pd.DataFrame({'date': dates, 'path': path, 'coverage': 'partial', 'colorid': i})
                    df_list.append(i_df)
                    i+=1
                df_all = (
                    pd
                    .concat(df_list)
                    .rename(columns={'path': colname})
                    .sort_values(['date'])
                )
                df_all.drop(['colorid'], axis=1).to_csv(f'text_calendars/{basename}.csv', index=False)
                # make graphical calendars
                date_gen = pd.DataFrame({
                    'date': yr_dates
                    , 'colorid':0
                })
                color_ids = (
                    pd
                    .concat([date_gen, df_all.loc[:, ['date', 'colorid']]])
                    .sort_values(['date', 'colorid'], ascending=[True, False])
                    .drop_duplicates(['date'])
                    .colorid
                )
                df_dates = pd.Series(color_ids.values, index=yr_dates)
                day_nums, day_vals = split_months(df_dates, year)
                create_year_calendar(day_nums, day_vals, file_prefix=basename, color_dict=color_dict, full_paths=full_paths, part_paths=part_paths)
    #         # loop over acq_dates
    #         for acqday, yr_start in zip(yr_starts.acqday, yr_starts.start_date):
    #             file_prefix = f'{sat}_acqday{acqday}_{year}'
                # df = date_series(yr_start, pass_int)
                # day_nums, day_vals = split_months(df, year)
    #             create_month_calendar(day_nums, day_vals, file_prefix=file_prefix, color_dict=color_dict)
# if __name__ == "main":
#     main()
