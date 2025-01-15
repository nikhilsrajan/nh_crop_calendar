import os
import argparse
import geopandas as gpd
import shapely
import rasterio
import numpy as np
import datetime
import numba
import tqdm
import time

import sys
sys.path.append('..')

import csu
import create_weather_data_catalogue as cwdc
import presets
import rsutils.utils
import rsutils.modify_images


# cluster paths
GEOGLAM_CPC_TMAX_FOLDERPATH = '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/cpc_tmax'
GEOGLAM_CPC_TMIN_FOLDERPATH = '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/cpc_tmin'

# # local debug
# GEOGLAM_CPC_TMAX_FOLDERPATH = '../data/cluster_files/cpc_tmax'
# GEOGLAM_CPC_TMIN_FOLDERPATH = '../data/cluster_files/cpc_tmin'

# Maize medium variety, source: FAO
## link: https://www.fao.org/land-water/databases-and-software/crop-information/maize/es/
T_BASE = 10
REQUIRED_GDD = 3000
MAX_TOLERABLE_TEMP = 45
MIN_TOLERABLE_TEMP = 0

COL_CROPPED_FILEPATH = 'cropped_filepath'


def get_bounds_gdf(shapes_gdf:gpd.GeoDataFrame, hardset_crs:str='epsg:4326'):
    shapes_gdf = rsutils.utils.get_bounds_gdf(shapes_gdf=shapes_gdf)

    minx, miny, maxx, maxy = shapes_gdf['geometry'][0].bounds
    crs = shapes_gdf.crs

    bounds_gdf = gpd.GeoDataFrame(
        data = {
            'geometry': [
                shapely.Polygon([
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                ])
            ]
        },
        crs = crs
    ).to_crs(hardset_crs)

    return bounds_gdf


def crop_weather_data_to_roi_bounds(
    weather_catalogue_df, bounds_gdf, working_dir, njobs,
):
    cropped_folderpath = os.path.join(working_dir, 'cropped')
    os.makedirs(cropped_folderpath, exist_ok=True)

    src_filepaths = []
    dst_filepaths = []

    for index, row in weather_catalogue_df.iterrows():
        filepath = row['tif_filepath']
        cropped_filepath = rsutils.utils.modify_filepath(
            filepath = filepath,
            new_folderpath = cropped_folderpath,
            prefix = 'cropped_',
        )
        if not os.path.exists(cropped_filepath):
            src_filepaths.append(filepath)
            dst_filepaths.append(cropped_filepath)
        weather_catalogue_df.loc[index, COL_CROPPED_FILEPATH] = cropped_filepath
    
    rsutils.modify_images.modify_images(
        src_filepaths = src_filepaths,
        dst_filepaths = dst_filepaths,
        sequence = [
            (rsutils.modify_images.crop, dict(shapes_gdf = bounds_gdf, all_touched = True))
        ],
        njobs = njobs,
        working_dir = working_dir,
    )
    
    return weather_catalogue_df


def get_tmean_stack(weather_catalogue_df, nodata = np.nan):
    t_mean_stack = []

    weather_catalogue_df = weather_catalogue_df.sort_values(by='date')

    start_date = weather_catalogue_df['date'].min()
    end_date = weather_catalogue_df['date'].max()

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    dates = [start_date + datetime.timedelta(days=d)
             for d in range((end_date - start_date).days + 1)]
    
    dates_filepaths_dict = weather_catalogue_df.groupby('date')[
        ['attribute', COL_CROPPED_FILEPATH]
    ].apply(
        lambda x: dict(zip(x['attribute'], x[COL_CROPPED_FILEPATH]))
    ).to_dict()

    template_filepath = weather_catalogue_df[COL_CROPPED_FILEPATH][0]
    with rasterio.open(template_filepath) as src:
        _, height, width = src.read().shape

    t_mean_stack = np.full(shape=(len(dates), height, width), fill_value=nodata, dtype=float)

    for i, date in tqdm.tqdm(enumerate(dates), total=len(dates)):
        date = date.strftime('%Y-%m-%d')
        if date in dates_filepaths_dict.keys():
            filepaths_dict = dates_filepaths_dict[date]
            if 'cpc-tmax' in filepaths_dict.keys():
                filepath = filepaths_dict['cpc-tmax']
                with rasterio.open(filepath) as src:
                    cpc_tmax = src.read()
            else:
                raise ValueError(f'Missing date for cpc-tmax: {date}')
            
            if 'cpc-tmin' in filepaths_dict.keys():
                filepath = filepaths_dict['cpc-tmin']
                with rasterio.open(filepath) as src:
                    cpc_tmin = src.read()
            else:
                raise ValueError(f'Missing date for cpc-tmin: {date}')
            
            t_mean_stack[i] = (cpc_tmax + cpc_tmin) / 2
        
        else:
            raise ValueError(f'Missing date: {date}')

    return t_mean_stack, dates


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python gdd_based_csu.py',
        description = (
            'Script to create GDD based CSU for a given shapefile.'
        ),
    )
    
    parser.add_argument('roi_shape_filepath', help='Path to ROI shapefile')
    parser.add_argument('--start-year', required=False, default=None, help='Start year (YYYY)')
    parser.add_argument('--end-year', required=False, default=None, help='End year (YYYY)')
    parser.add_argument('--working-dir', required=True, help='Folderpath where working files will be stored')
    parser.add_argument('--HM-cutoff-date', required=True, help='[YYYY-MM-DD] Date upto which "days to maturity" should be considered for calculating harmonic mean.')
    parser.add_argument('--njobs', required=False, default=1, help='Number of cores to run.')

    args = parser.parse_args()

    roi_gdf = gpd.read_file(str(args.roi_shape_filepath))
    bounds_gdf = get_bounds_gdf(shapes_gdf = roi_gdf, hardset_crs='epsg:4326')
    njobs = int(args.njobs)
    
    years = None
    if args.start_year is not None and args.end_year is not None:
        years = list(range(int(args.start_year), int(args.end_year) + 1))

    hm_cutoff_date = datetime.datetime.strptime(str(args.HM_cutoff_date), '%Y-%m-%d')

    working_dir = str(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

    t_mean_stack_filepath = os.path.join(working_dir, f'tmean-stack_{years[0]}_{years[-1]}.npy')
    gdd_dates_filepath = os.path.join(working_dir, f'gdd-dates_{years[0]}_{years[-1]}.npy')
    days_to_maturity_filepath = os.path.join(working_dir, f'days-to-maturity_{years[0]}_{years[-1]}.npy')
    gdd_at_maturity_filepath = os.path.join(working_dir, f'gdd-at-maturity_{years[0]}_{years[-1]}.npy')

    if os.path.exists(t_mean_stack_filepath) and os.path.exists(gdd_dates_filepath):
        print('Loading mean temperature stack')
        t_mean_stack = np.load(t_mean_stack_filepath)
        gdd_dates = np.load(gdd_dates_filepath, allow_pickle=True)
    else:
        weather_catalogue_df = cwdc.create_weather_data_catalogue_df(
            years = years,
            attribute_settings_dict = {
                presets.ATTR_CPCTMAX: cwdc.Settings(
                    attribute_folderpath = GEOGLAM_CPC_TMAX_FOLDERPATH,
                ),
                presets.ATTR_CPCTMIN: cwdc.Settings(
                    attribute_folderpath = GEOGLAM_CPC_TMIN_FOLDERPATH,
                ),
            }
        )

        print('Cropping weather data to roi bounds')
        weather_catalogue_df = crop_weather_data_to_roi_bounds(
            weather_catalogue_df = weather_catalogue_df,
            bounds_gdf = bounds_gdf, 
            working_dir = working_dir,
            njobs = njobs,
        )
        if years is None:
            print(weather_catalogue_df.columns)
            years = weather_catalogue_df['year'].unique().tolist()
            years.sort()

        print('Creating mean temperature stack')
        t_mean_stack, gdd_dates = get_tmean_stack(weather_catalogue_df=weather_catalogue_df)
        np.save(t_mean_stack_filepath, t_mean_stack)
        np.save(gdd_dates_filepath, gdd_dates)


    if os.path.exists(days_to_maturity_filepath) and os.path.exists(gdd_at_maturity_filepath):
        print('Loading days to maturity')
        days_to_maturity = np.load(days_to_maturity_filepath)
        gdd_at_maturity = np.load(gdd_at_maturity_filepath)
    else:
        print('Compute days to maturity')
        numba.set_num_threads(n = njobs)

        _start_time = time.time()
        days_to_maturity, gdd_at_maturity = \
        csu.calculate_days_to_maturity(
            temp_ts = t_mean_stack,
            t_base = T_BASE,
            required_gdd = REQUIRED_GDD,
            max_tolerable_temp = MAX_TOLERABLE_TEMP,
            min_tolerable_temp = MIN_TOLERABLE_TEMP,
        )
        _end_time = time.time()

        print(f't_elapsed: {_end_time - _start_time} s')
        np.save(days_to_maturity_filepath, days_to_maturity)
        np.save(gdd_at_maturity_filepath, gdd_at_maturity)

    print('Saving HM days to maturity')
    cutoff_index = np.where(np.array(gdd_dates) == hm_cutoff_date)[0][0]
    valid_days_to_maturity = days_to_maturity[:cutoff_index+1] # +1 to include the cutoff date
    hm_days_to_maturity = valid_days_to_maturity.shape[0] / (1 / valid_days_to_maturity).sum(axis=0)

    cropped_template_filepath = weather_catalogue_df[COL_CROPPED_FILEPATH][0]

    with rasterio.open(cropped_template_filepath) as src:
        out_meta = src.meta.copy()

    HM_days_to_maturity_filepath = os.path.join(working_dir, 
                                                f"HM-days-to-maturity_{gdd_dates[0].strftime('%Y-%m-%d')}_{gdd_dates[cutoff_index].strftime('%Y-%m-%d')}.tif")

    with rasterio.open(HM_days_to_maturity_filepath, 'w', **out_meta) as dst:
        dst.write(
            np.expand_dims(hm_days_to_maturity, axis=0)
        )
    print(f'Saved HM days to maturity: {os.path.abspath(HM_days_to_maturity_filepath)}')

    end_time = time.time()
    print(f'--- t_elapsed: {end_time - start_time} s ---')
