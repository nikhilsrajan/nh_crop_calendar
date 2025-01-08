import os
import argparse
import geopandas as gpd
import shapely
import tqdm
import rasterio
import numpy as np
import datetime

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
    weather_catalogue_df, bounds_gdf, working_dir
):
    cropped_folderpath = os.path.join(working_dir, 'cropped')
    os.makedirs(cropped_folderpath, exist_ok=True)

    for index, row in tqdm.tqdm(weather_catalogue_df.iterrows(), 
                                total=weather_catalogue_df.shape[0]):
        filepath = row['tif_filepath']
        cropped_filepath = rsutils.utils.modify_filepath(
            filepath = filepath,
            new_folderpath = cropped_folderpath,
            prefix = 'cropped_',
        )
        if not os.path.exists(cropped_filepath):
            rsutils.modify_images.crop(
                src_filepath = filepath,
                dst_filepath = cropped_filepath,
                shapes_gdf = bounds_gdf,
                all_touched = True,
            )
        weather_catalogue_df.loc[index, COL_CROPPED_FILEPATH] = cropped_filepath
    
    return weather_catalogue_df


def get_tmean_stack(weather_catalogue_df):
    cpc_tmax_stack = []
    cpc_tmin_stack = []

    weather_catalogue_df = weather_catalogue_df.sort_values(by='date')

    for index, row in weather_catalogue_df.iterrows():
        attribute = row['attribute']
        filepath = row[COL_CROPPED_FILEPATH]

        if attribute == 'cpc-tmax':
            with rasterio.open(filepath) as src:
                cpc_tmax_stack.append(src.read())

        elif attribute == 'cpc-tmin':
            with rasterio.open(filepath) as src:
                cpc_tmin_stack.append(src.read())
        
    cpc_tmax_stack = np.concatenate(cpc_tmax_stack, axis=0)
    cpc_tmin_stack = np.concatenate(cpc_tmin_stack, axis=0)

    t_mean_stack = (cpc_tmax_stack + cpc_tmin_stack) / 2

    del cpc_tmax_stack, cpc_tmin_stack

    return t_mean_stack


if __name__ == '__main__':
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

    args = parser.parse_args()

    roi_gdf = gpd.read_file(str(args.roi_shape_filepath))
    bounds_gdf = get_bounds_gdf(shapes_gdf = roi_gdf, hardset_crs='epsg:4326')
    
    years = None
    if args.start_year is not None and args.end_year is not None:
        years = list(range(int(args.start_year), int(args.end_year) + 1))

    hm_cutoff_date = datetime.datetime.strptime(str(args.HM_cutoff_date), '%Y-%m-%d')

    working_dir = str(args.working_dir)
    os.makedirs(working_dir, exist_ok=True)

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
    )
    if years is None:
        print(weather_catalogue_df.columns)
        years = weather_catalogue_df['year'].unique().tolist()
        years.sort()

    t_mean_stack_filepath = os.path.join(working_dir, f'tmean-stack_{years[0]}_{years[-1]}.npy')
    days_to_maturity_filepath = os.path.join(working_dir, f'days-to-maturity_{years[0]}_{years[-1]}.npy')
    gdd_at_maturity_filepath = os.path.join(working_dir, f'gdd-at-maturity_{years[0]}_{years[-1]}.npy')

    dates = weather_catalogue_df['date'].unique().tolist()
    dates = [datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in dates]
    dates.sort()

    print('Creating mean temperature stack')
    if os.path.exists(t_mean_stack_filepath):
        t_mean_stack = np.load(t_mean_stack_filepath)
    else:
        t_mean_stack = get_tmean_stack(weather_catalogue_df=weather_catalogue_df)
        np.save(t_mean_stack_filepath, t_mean_stack)

    print('Compute days to maturity')
    if os.path.exists(days_to_maturity_filepath) and os.path.exists(gdd_at_maturity_filepath):
        days_to_maturity = np.load(days_to_maturity_filepath)
        gdd_at_maturity = np.load(gdd_at_maturity_filepath)
    else:
        days_to_maturity, gdd_at_maturity = \
        csu.calculate_days_to_maturity(
            temp_ts = t_mean_stack,
            t_base = T_BASE,
            required_gdd = REQUIRED_GDD,
            max_tolerable_temp = MAX_TOLERABLE_TEMP,
            min_tolerable_temp = MIN_TOLERABLE_TEMP,
        )
        np.save(days_to_maturity_filepath, days_to_maturity)
        np.save(gdd_at_maturity_filepath, gdd_at_maturity)

    print('Saving HM days to maturity')
    cutoff_index = np.where(np.array(dates) == hm_cutoff_date)[0][0]
    valid_days_to_maturity = days_to_maturity[:cutoff_index+1] # +1 to include the cutoff date
    hm_days_to_maturity = valid_days_to_maturity.shape[0] / (1 / valid_days_to_maturity).sum(axis=0)

    cropped_template_filepath = weather_catalogue_df[COL_CROPPED_FILEPATH][0]

    with rasterio.open(cropped_template_filepath) as src:
        out_meta = src.meta.copy()

    HM_days_to_maturity_filepath = os.path.join(working_dir, 
                                                f"HM-days-to-maturity_{dates[0].strftime('%Y-%m-%d')}_{dates[cutoff_index].strftime('%Y-%m-%d')}.tif")

    with rasterio.open(HM_days_to_maturity_filepath, 'w', **out_meta) as dst:
        dst.write(
            np.expand_dims(hm_days_to_maturity, axis=0)
        )
    print(f'Saved HM days to maturity: {os.path.abspath(HM_days_to_maturity_filepath)}')
