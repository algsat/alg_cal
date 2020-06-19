import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Settings
years = [2020, 2021]
weeks = [1, 2, 3, 4, 5, 6]
days = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
               'September', 'October', 'November', 'December']

# FUNCTIONS

#
def date_series(full_starts, return_period, partial_starts=None):
    if isinstance(full_starts, str):
        full_starts = [full_starts]
    year = full_starts[0][:4]
    idx = pd.date_range(f'{year}-01-01', periods=365, freq='D')
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
        col = d[0].dayofweek

        if d[0].is_month_start:
            row = 0

        day_nums[month][row, col] = day  # day number (0-31)
        day_vals[month][row, col] = d[1] # day value (the heatmap data)

        if col == 6:
            row += 1

    return day_nums, day_vals

# Taken from SO_tourist answer at: https://stackoverflow.com/questions/32485907/matplotlib-and-numpy-create-a-calendar-heatmap
def create_year_calendar(day_nums, day_vals, file_prefix='example', year=2020):
    for i in range(0, 12):
        fig, axs = plt.subplots(1, 1, figsize=(5, 4))
        axs.imshow(day_vals[i+1], cmap='Purples', vmin=-0.05, vmax=1.25)  # heatmap
        axs.set_title(month_names[i], fontsize=20)

        # Labels
        axs.set_xticks(np.arange(len(days)))
        axs.set_xticklabels(days, fontsize=14, fontweight='bold', color='#555555')
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
                             fontsize=14, color="#003333", alpha=0.8)  # day

                # Aesthetic background for calendar day number
                patch_coords = ((d-0.4, w-0.5),
                                (d+0.5, w-0.5),
                                (d+0.5, w+0.4))

                triangle = Polygon(patch_coords, fc='w', alpha=0.7)
                axs.add_artist(triangle)

        # Final adjustments
        plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.04)
        # plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
        fig.tight_layout()

        # Save to file
        outname = f'../alg_sat_images/{file_prefix}_{year}{i+1:02}.png'
        plt.savefig(outname, dpi=120)

def main():
    for year in years:
        df = date_series('2020-01-04', 5)
        day_nums, day_vals = split_months(df, year)
        create_year_calendar(day_nums, day_vals, file_prefix='test', year=year)

if __name__ == "main":
    main()
