import os
import pandas as pd
import argparse
import geopandas as gpd
import shutil
import datetime
import tqdm
import rasterio
import numpy as np
import xarray as xr

import sys
sys.path.append('..')

import rsutils.modify_images
import rsutils.utils


def create_datacube(
    catalog_df:pd.DataFrame,
    filepath_col:str,
    date_col:str,
    export_filepath:str,
):
    catalog_df = catalog_df.dropna()
    catalog_df = catalog_df.sort_values(by=date_col, ascending=True)
    nparray = []

    for index, row in tqdm.tqdm(catalog_df.iterrows(), total=catalog_df.shape[0]):
        with rasterio.open(row[filepath_col]) as src:
            nparray.append(src.read())

    nparray = np.concatenate(nparray, axis=0)

    if nparray.dtype == np.uint8:
        nparray = nparray.astype(np.int16)
    
    dataarray = xr.DataArray(
        data = nparray,
        dims = ('timestamps', 'height', 'width'),
        coords = {
            'timestamps': catalog_df['date'].to_list(),
        }
    )

    dataarray.to_netcdf(export_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python aggregate_tifs.py',
        description = (
            'Script to aggregate tifs to df.'
        ),
    )

    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('reference_filepath', help='/path/to/reference_tif')
    parser.add_argument('weather_catalog_csv_filepath', help='/path/to/weather_catalog_csv')
    parser.add_argument('export_folderpath', help='/path/to/export')
    parser.add_argument('-j', '--njobs', default=1, required=False, help='[default=1] Number of parallel jobs')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite dataarrays')
    
    args = parser.parse_args()

    shapes_gdf = gpd.read_file(args.roi_filepath)
    weather_catalog_df = pd.read_csv(args.weather_catalog_csv_filepath)

    export_folderpath = str(args.export_folderpath)
    njobs = int(args.njobs)

    working_dir = os.path.join(export_folderpath, 'temp')
    os.makedirs(working_dir, exist_ok=True)

    weather_catalog_df['cropped_tif_filepath'] = weather_catalog_df['tif_filepath'].apply(
        lambda x: rsutils.utils.modify_filepath(filepath=x, new_folderpath=working_dir, prefix='cropped_')
    )

    weather_catalog_df['success'] = rsutils.modify_images.modify_images(
        src_filepaths = weather_catalog_df['tif_filepath'],
        dst_filepaths = weather_catalog_df['cropped_tif_filepath'],
        sequence = [
            (rsutils.modify_images.crop, dict(shapes_gdf=shapes_gdf, all_touched=True)),
            (rsutils.modify_images.resample_by_ref, dict(ref_filepath=args.reference_filepath)),
        ],
        njobs = njobs,
        raise_error = False,
    )

    print('original counts:')
    print(weather_catalog_df['attribute'].value_counts())

    weather_catalog_df = weather_catalog_df[weather_catalog_df['success']]

    print('success counts:')
    print(weather_catalog_df['attribute'].value_counts())

    for attr in tqdm.tqdm(weather_catalog_df['attribute'].unique().tolist()):
        print('attribute:', attr)
        dataarray_path = os.path.join(export_folderpath, f'{attr}.nc')
        if args.overwrite or not os.path.exists(dataarray_path):
            create_datacube(
                catalog_df = weather_catalog_df[weather_catalog_df['attribute'] == attr],
                filepath_col = 'cropped_tif_filepath',
                date_col = 'date',
                export_filepath = dataarray_path,
            )

    shutil.rmtree(working_dir)
