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
GEOGLAM_CHIRPS_FOLDERPATH = '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/chirps/global'

# # local debug
# GEOGLAM_CHIRPS_FOLDERPATH = '../data/cluster_files/chirps'

# Maize medium variety, source: FAO
## link: https://www.fao.org/land-water/databases-and-software/crop-information/maize/es/
## maize water needs source: https://www.fao.org/4/s2022e/s2022e07.htm#chapter%203:%20crop%20water%20needs -- 500-800 mm/total growing period
REQUIRED_PRECP = 500 # mm / total growing period

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


def get_prec_stack(weather_catalogue_df, nodata=0):
    # refactor to fill zeros for missing dates !!
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

    prec_stack = np.full(shape=(len(dates), height, width), fill_value=nodata, dtype=float)

    for i, date in tqdm.tqdm(enumerate(dates), total=len(dates)):
        date = date.strftime('%Y-%m-%d')
        if date in dates_filepaths_dict.keys():
            filepaths_dict = dates_filepaths_dict[date]
            if 'chirps' in filepaths_dict.keys():
                filepath = filepaths_dict['chirps']
                with rasterio.open(filepath) as src:
                    prec_stack[i] = src.read()
        else:
            raise ValueError(f'Missing date: {date}')
        
    prec_stack[prec_stack < 0] = np.nan
    prec_stack = prec_stack / 100

    return prec_stack, dates


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python prec_based_csu.py',
        description = (
            'Script to create Precipitation based CSU for a given shapefile.'
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

    prec_stack_filepath = os.path.join(working_dir, f'prec-stack_{years[0]}_{years[-1]}.npy')
    prec_dates_filepath = os.path.join(working_dir, f'prec-dates_{years[0]}_{years[-1]}.npy')
    days_to_req_prec_filepath = os.path.join(working_dir, f'days-to-req-prec_{years[0]}_{years[-1]}.npy')
    prec_at_req_prec_filepath = os.path.join(working_dir, f'prec-at-req-prec_{years[0]}_{years[-1]}.npy')

    if os.path.exists(prec_stack_filepath) and os.path.exists(prec_dates_filepath):
        print('Loading precipitation stack')
        prec_stack = np.load(prec_stack_filepath)
        prec_dates = np.load(prec_dates_filepath)
    else:
        weather_catalogue_df = cwdc.create_weather_data_catalogue_df(
            years = years,
            attribute_settings_dict = {
                presets.ATTR_CHIRPS: cwdc.Settings(
                    attribute_folderpath = GEOGLAM_CHIRPS_FOLDERPATH,
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

        print('Creating precipitation stack')
        prec_stack, prec_dates = get_prec_stack(weather_catalogue_df=weather_catalogue_df)
        np.save(prec_stack_filepath, prec_stack)
        np.save(prec_dates_filepath, prec_dates)

    
    if os.path.exists(days_to_req_prec_filepath) and os.path.exists(prec_at_req_prec_filepath):
        print('Loading days to required precipitation')
        days_to_req_prec = np.load(days_to_req_prec_filepath)
        prec_to_req_prec = np.load(prec_at_req_prec_filepath)
    else:
        print('Compute days to required precipitation')
        # need to refactor function to be more general
        numba.set_num_threads(n = njobs)
        _start_time = time.time()
        days_to_req_prec, prec_to_req_prec = \
        csu.calculate_days_to_maturity(
            temp_ts = prec_stack,
            t_base = 0,
            required_gdd = 800,
            max_tolerable_temp = 10000,
            min_tolerable_temp = -10000,
        )
        _end_time = time.time()

        print(f't_elapsed: {_end_time - _start_time} s')

        np.save(days_to_req_prec_filepath, days_to_req_prec)
        np.save(prec_at_req_prec_filepath, prec_to_req_prec)

    print('Saving HM days to required precipitation')
    cutoff_index = np.where(np.array(prec_dates) == hm_cutoff_date)[0][0]
    valid_days_to_req_prec = days_to_req_prec[:cutoff_index+1] # +1 to include the cutoff date
    hm_days_to_req_prec = valid_days_to_req_prec.shape[0] / (1 / valid_days_to_req_prec).sum(axis=0)

    cropped_template_filepath = weather_catalogue_df[COL_CROPPED_FILEPATH][0]

    with rasterio.open(cropped_template_filepath) as src:
        out_meta = src.meta.copy()

    HM_days_to_req_prec_filepath = os.path.join(
        working_dir, f"HM-days-to-req-prec_{prec_dates[0].strftime('%Y-%m-%d')}_{prec_dates[cutoff_index].strftime('%Y-%m-%d')}.tif"
    )

    with rasterio.open(HM_days_to_req_prec_filepath, 'w', **out_meta) as dst:
        dst.write(np.expand_dims(hm_days_to_req_prec, axis=0))

    print(f'Saved HM days to required precipitation: {os.path.abspath(HM_days_to_req_prec_filepath)}')

    end_time = time.time()
    print(f'--- t_elapsed: {_end_time - _start_time} s ---')