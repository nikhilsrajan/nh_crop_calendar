import pandas as pd
import xarray as xr
import shapely
import tqdm
import geopandas as gpd
import multiprocessing as mp
import functools
import tqdm
import time
import numpy as np
import os
import argparse

import sys
sys.path.append('..')

import era5_utils
import csu


VAR_PREC = 'total_precipitation'
VAR_TEMP = '2m_temperature'


# Maize medium variety, source: FAO
## link: https://www.fao.org/land-water/databases-and-software/crop-information/maize/es/
T_BASE = 10
REQUIRED_GDD = 3000
MAX_TOLERABLE_TEMP = 45
MIN_TOLERABLE_TEMP = 0
REQUIRED_PRECP = 500 # mm / total growing period
MAX_CROP_DURATION = 300 # days


ERA5_CATALOG_FILEPATH = '/gpfs/data1/cmongp1/sasirajann/download_era5/data/era5/catalog.csv'
MALAWI_SHAPEFILEPATH = '/gpfs/data1/cmongp1/sasirajann/download_era5/data/shapefiles/mwi_adm_nso_hotosm_20230405_shp/mwi_admbnda_adm0_nso_hotosm_20230405.shp'

OUTPUT_FOLDERPATH = '/gpfs/data1/cmongp2/sasirajann/nh_crop_calendar/crop_calendar/data/era5_csu'


def get_filepath(
    var:str,
    date:str,
    catalog_df:pd.DataFrame,
):
    return catalog_df[
        (catalog_df['var'] == var) &
        (catalog_df['date'] == date)
    ]['filepath'].tolist()[0]


def clip_data(
    data:xr.DataArray,
    shapes:list[shapely.Geometry],
    crs:str,
    drop:bool = True,
    all_touched:bool = True,
):
    return data.rio.clip(shapes, crs, drop=drop, all_touched=all_touched)


def load_clipped_data(
    filepath:str,
    var:str,
    shapes:list[shapely.Geometry] = None,
    crs:str = None,
    drop:bool = True,
    all_touched:bool = True,
):
    if var == VAR_TEMP:
        era5_data = era5_utils.load_mean_temperature_nc_file(nc_filepath=filepath)
    elif var == VAR_PREC:
        era5_data = era5_utils.load_total_precipitation_nc_file(nc_filepath=filepath)
    else:
        raise ValueError(f'Invalid var={var}. var must be either {VAR_TEMP} or {VAR_PREC}')

    if shapes is None:
        return era5_data

    clipped_era5_data = clip_data(
        data = era5_data,
        shapes = shapes,
        crs = crs,
        drop = drop,
        all_touched = all_touched,
    )

    return clipped_era5_data


def load_clipped_data_by_daterange(
    startdate:str,
    enddate:str,
    catalog_df:pd.DataFrame,
    var:str,
    shapes:list[shapely.Geometry] = None,
    crs:str = None,
    drop:bool = True,
    all_touched:bool = True,
    njobs:int = 1,
):
    if var not in [VAR_TEMP, VAR_PREC]:
        raise ValueError(f'Invalid var={var}. var must be either {VAR_TEMP} or {VAR_PREC}')

    temp_catalog_df = catalog_df[catalog_df['var'] == var]
    temp_catalog_df = temp_catalog_df.sort_values(by='date')

    filepaths = temp_catalog_df[
        (temp_catalog_df['date'] >= startdate) &
        (temp_catalog_df['date'] <= enddate) 
    ]['filepath'].to_list()

    load_clipped_temp_data_partial = functools.partial(
        load_clipped_data,
        var = var,
        shapes = shapes,
        crs = crs,
        drop = drop,
        all_touched = all_touched,
    )

    with mp.Pool(njobs) as p:
        clipped_temp_datas = list(tqdm.tqdm(
            p.imap(load_clipped_temp_data_partial, filepaths), 
            total=len(filepaths)
        ))

    concat_clipped_temp_data = xr.concat(clipped_temp_datas, dim='valid_time')

    return concat_clipped_temp_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python scripts/era5_csu.py',
        description = (
            'Script to generate CSU using era5 temperature and precipitation data.'
        ),
    )


    for name, default, desc in [
        ('--t-base', T_BASE, 'Base temperature for GDD'),
        ('--req-gdd', REQUIRED_GDD, 'Required GDD for maturity'),
        ('--max-tol-temp', MAX_TOLERABLE_TEMP, 'Max tolerable temperature'),
        ('--min-tol-prec', MIN_TOLERABLE_TEMP, 'Min tolerable temperature'),
        ('--req-prec', REQUIRED_PRECP, 'Required precipitation for growth'),
        ('--max-tol-prec', None, 'Max tolerable precipitation'),
        ('--max-duration', None, 'Max duration for maturity'),
    ]:
        parser.add_argument(name, action='store', required=False, default=default, type=int, help=f'[default = {default}] {desc}')    

    parser.add_argument('-r', '--roi', action='store', required=False, default=None, help='path/to/shapefile')

    parser.add_argument('-s', '--startdate', action='store', required=True, help="[YYYY-MM-DD] Date to start data aggregation.")
    parser.add_argument('-e', '--enddate', action='store', required=True, help="[YYYY-MM-DD] Date to end data aggregation.")
    parser.add_argument('-c', '--cutoffdate', action='store', required=True, help="[YYYY-MM-DD] Date to cut-off before computing suitability.")

    parser.add_argument('-f', '--filepath', action='store', default=None, required=False, help="filepath where the file is to be downloaded to. Overrides -F.")
    parser.add_argument('-F', '--folderpath', action='store', default=None, required=False, help="folderpath where the file is to be downloaded to. Overriden by -f.")

    parser.add_argument('-j', '--njobs', action='store', required=False, default=1, type=int, help="[default = 1] Number of jobs to execute in parallel.")
    
    
    args = parser.parse_args()

    max_duration = args.max_duration
    if max_duration is None:
        max_duration = np.inf

    max_tol_prec = args.max_tol_prec
    if max_tol_prec is None:
        max_tol_prec = np.inf

    export_filepath = args.filepath
    if export_filepath is None:
        export_folderpath = args.folderpath

        if export_folderpath is None:
            raise ValueError('Either --filepath or --folderpath should be passed.')
        
        os.makedirs(export_folderpath, exist_ok=True)

        export_filepath = os.path.join(
            export_folderpath, 
            '_'.join([
                "sum-suitable-days",
                f"tbase={args.t_base}",
                f"reqgdd={args.req_gdd}",
                f"maxtoltemp={args.max_tol_temp}",
                f"mintoltemp={args.min_tol_temp}",
                f"reqprec={args.req_prec}",
                f"maxduration={max_duration}",
                f"maxtolprec={max_tol_prec}",
            ]) + ".tif"
        )

    print('== PARAMETERS ==')
    for name, value in [
        ("t_base", args.t_base),
        ("req_gdd", args.req_gdd),
        ("max_tol_temp", args.max_tol_temp),
        ("min_tol_temp", args.min_tol_temp),
        ("req_prec", args.req_prec),
        ("max_duration", max_duration),
        ("max_tol_prec", max_tol_prec),
    ]:
        print(name, '=', value)

    catalog_df = pd.read_csv(ERA5_CATALOG_FILEPATH)

    if args.roi is not None:
        _gdf = gpd.read_file(args.roi)
        shapes = [shapely.unary_union(_gdf['geometry']).envelope]
        crs = _gdf.crs
    else:
        shapes = None
        crs = None

    malawi_temp_data = load_clipped_data_by_daterange(
        startdate = args.startdate,
        enddate = args.enddate,
        var = VAR_TEMP,
        catalog_df = catalog_df,
        shapes = shapes,
        crs = crs,
        njobs = args.njobs,
    )

    malawi_prec_data = load_clipped_data_by_daterange(
        startdate = args.startdate,
        enddate = args.enddate,
        var = VAR_PREC,
        catalog_df = catalog_df,
        shapes = shapes,
        crs = crs,
        njobs = args.njobs,
    )

    # print(malawi_temp_data)

    dates = malawi_temp_data.valid_time.values

    # print(type(dates))

    # print(dates)

    cutoff_date = pd.Timestamp(args.cutoffdate)

    cutoff_index = np.where(np.array(dates) == cutoff_date)[0][0]

    print(f'cutoff_index = {cutoff_index}')

    print('START: computing days to maturity')

    start_time = time.time()

    days_to_maturity, gdd_at_maturity = \
    csu.calculate_days_to_maturity(
        temp_ts = malawi_temp_data.values,
        t_base = args.t_base,
        required_gdd = args.req_gdd,
        max_tolerable_temp = args.max_tol_temp,
        min_tolerable_temp = args.min_tol_temp,
    )

    days_to_req_prec, prec_to_req_prec = \
    csu.calculate_days_to_maturity(
        temp_ts = malawi_prec_data.values,
        t_base = 0,
        required_gdd = args.req_prec,
        max_tolerable_temp = max_tol_prec,
        min_tolerable_temp = -10000,
    )

    cum_prec = malawi_prec_data.values.cumsum(axis=0)


    end_time = time.time()

    valid_days_to_maturity = days_to_maturity[:cutoff_index]
    valid_days_to_req_prec = days_to_req_prec[:cutoff_index]

    print(f'END: computing days to maturity, t_elapsed: {round(end_time-start_time, 2)} secs')

    suitable_days = np.zeros(shape=valid_days_to_maturity.shape, dtype=np.uint8)
    suitable_days[np.where(
        (valid_days_to_req_prec <= valid_days_to_maturity)
        & (valid_days_to_maturity <= max_duration)
        & (valid_days_to_maturity != np.inf)
        & (valid_days_to_req_prec != np.inf)
    )] = 1
    sum_suitable_days = xr.DataArray(
        suitable_days.sum(axis=(0)).astype(np.uint16),
        coords = {
            'latitude': malawi_temp_data.latitude,
            'longitude': malawi_temp_data.longitude,
        },
        dims = ('latitude', 'longitude'),
    )

    # print(sum_suitable_days)
        
    sum_suitable_days = sum_suitable_days.rio.set_spatial_dims('longitude', 'latitude')
    sum_suitable_days = sum_suitable_days.rio.write_crs('epsg:4326')
    sum_suitable_days.rio.to_raster(export_filepath)

