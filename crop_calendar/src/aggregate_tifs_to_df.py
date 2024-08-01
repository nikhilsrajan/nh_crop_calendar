import rasterio
import rasterio.merge
import geopandas as gpd
import numpy as np
import tqdm
import os
import pandas as pd
import functools

import presets
import rsutils.utils as utils
import create_weather_data_catalogue as cwdc
import spatially_interpolate_files as sif


METHOD_COL = 'method'
CSV_FILEPATH_COL = 'csv_filepath'
VAL_COL = 'val'
X_COL = 'x'
Y_COL = 'y'


class LoadTIFMethod:
    READ_AND_CROP = 'read and crop'
    READ_NO_CROP = 'read no crop'
    COREGISTER_AND_CROP = 'coregister and crop'


def coregister_and_maybe_crop(
    tif_filepath:str,
    reference_tif_filepath:str,
    resampling=rasterio.merge.Resampling.nearest,
    nodata=None,
    shapes_gdf:gpd.GeoDataFrame = None,
):
    zero_tif_filepath = utils.add_epochs_prefix(
        filepath = reference_tif_filepath,
        prefix = 'zero_',
    )

    utils.create_zero_tif(
        reference_tif_filepath = reference_tif_filepath,
        zero_tif_filepath = zero_tif_filepath,
    )

    coregistered_tif_filepath = utils.add_epochs_prefix(
        filepath = tif_filepath,
        prefix = 'coregistered_',
    )

    utils.coregister(
        src_filepath = tif_filepath,
        dst_filepath = coregistered_tif_filepath,
        reference_zero_filepath = zero_tif_filepath,
        resampling = resampling,
        nodata = nodata,
    )

    if shapes_gdf is not None:
        out_image, out_meta = utils.crop_tif(
            src_filepath = coregistered_tif_filepath,
            shapes_gdf = shapes_gdf,
        )
    else:
        with rasterio.open(coregistered_tif_filepath) as src:
            out_image = src.read()
            out_meta = src.meta.copy()
    
    os.remove(zero_tif_filepath)
    os.remove(coregistered_tif_filepath)

    return out_image, out_meta


def load_tif(
    tif_filepath:str,
    bounds_gdf:gpd.GeoDataFrame = None,
    reference_tif_filepath:str = None,
    method:str = LoadTIFMethod.READ_NO_CROP,
    resampling = rasterio.merge.Resampling.nearest,
    nodata = None,
):
    if method == LoadTIFMethod.READ_NO_CROP:
        with rasterio.open(tif_filepath) as src:
            out_image = src.read()
            out_meta = src.meta.copy()

    elif method == LoadTIFMethod.READ_AND_CROP:
        if bounds_gdf is None:
            raise ValueError(f'bounds_gdf can not be None for method={method}')
        out_image, out_meta = utils.crop_tif(
            src_filepath=tif_filepath,
            shapes_gdf=bounds_gdf,
        )

    elif method == LoadTIFMethod.COREGISTER_AND_CROP:
        if bounds_gdf is None:
            raise ValueError(f'bounds_gdf can not be None for method={method}')
        if reference_tif_filepath is None:
            raise ValueError(f'reference_tif_filepath can not be None for method={method}')
        out_image, out_meta = coregister_and_maybe_crop(
            tif_filepath = tif_filepath,
            reference_tif_filepath = reference_tif_filepath,
            resampling = resampling,
            nodata = nodata,
            shapes_gdf = bounds_gdf,
        )
    return out_image, out_meta


def read_catalogue_and_create_csvs(
    mask_tif_filepaths:list[str],
    roi_geom_filepath:str,
    ref_tif_filepath:str,
    catalogue_df:pd.DataFrame,
    csvs_folderpath:str,
    overwrite:bool = False,
):
    mask_xs, mask_ys, _, _ = utils.get_mask_coords(
        mask_tif_filepaths = mask_tif_filepaths
    )

    roi_geom_gdf = gpd.read_file(roi_geom_filepath)

    bounds_gdf = utils.get_actual_bounds_gdf(
        src_filepath = ref_tif_filepath,
        shapes_gdf = roi_geom_gdf,
    )

    for index, row in tqdm.tqdm(
        catalogue_df.iterrows(), 
        total=catalogue_df.shape[0],
    ):
        attribute = row[cwdc.ATTRIBUTE_COL]
        filepath = row[sif.TIF_FILEPATH_COL]
        filetype = row[cwdc.FILETYPE_COL]
        method = row[METHOD_COL]
        year = row[presets.YEAR]
        day = row[presets.DAY]

        attribute_csv_folderpath = os.path.join(csvs_folderpath, attribute)
        os.makedirs(attribute_csv_folderpath, exist_ok=True)
        csv_filepath = os.path.join(attribute_csv_folderpath, f'{year}_{day}.csv')

        if not os.path.exists(csv_filepath) or overwrite:
            if filetype == cwdc.TIF_GZ_EXT:
                gzip_tif = utils.GZipTIF(filepath)
                filepath = gzip_tif.decompress_and_load()

            out_image, out_meta = load_tif(
                tif_filepath = filepath,
                bounds_gdf = bounds_gdf,
                reference_tif_filepath = ref_tif_filepath,
                method = method
            )

            data = {
                X_COL: mask_xs,
                Y_COL: mask_ys,
                VAL_COL: out_image[0, mask_xs, mask_ys]
            }
            _df = pd.DataFrame(data=data)
            _df.to_csv(csv_filepath, index=False)
            del _df

            del out_image, out_meta

            if filetype == cwdc.TIF_GZ_EXT:
                gzip_tif.delete_tif()
                del gzip_tif
        
        catalogue_df.loc[index, CSV_FILEPATH_COL] = csv_filepath
    
    return catalogue_df


def aggregate_csvs(
    catalogue_df:pd.DataFrame,
    attribute_col:str = cwdc.ATTRIBUTE_COL,
    csv_filepath_col:str = CSV_FILEPATH_COL,
    year_col:str = presets.YEAR,
    day_col:str = presets.DAY,
    x_col:str = X_COL,
    y_col:str = Y_COL,
    val_col:str = VAL_COL,
):
    attribute_wise_dfs = {}

    for attribute in tqdm.tqdm(catalogue_df[attribute_col].unique()):
        attribute_catalogue_df = catalogue_df[catalogue_df[attribute_col]==attribute]
        agg_attribute_df = None
        print(f'{attribute_col} =', attribute)
        for index, row in tqdm.tqdm(
            attribute_catalogue_df.iterrows(), 
            total=attribute_catalogue_df.shape[0],
        ):
            _filepath = row[csv_filepath_col]
            _year = row[year_col]
            _day = row[day_col]
            _df = pd.read_csv(_filepath)
            _df[year_col] = _year
            _df[day_col] = _day
            if agg_attribute_df is None:
                agg_attribute_df = _df
            else:
                agg_attribute_df = pd.concat([agg_attribute_df, _df])
            del _df
        attribute_wise_dfs[attribute] = agg_attribute_df
        del agg_attribute_df

    for key in attribute_wise_dfs.keys():
        _df = attribute_wise_dfs[key]
        attribute_wise_dfs[key] = _df.rename(columns={val_col:key})

    agg_df = functools.reduce(lambda  left, right: pd.merge(
        left, right, on=[x_col, y_col, year_col, day_col], how='outer',
    ), attribute_wise_dfs.values())

    attributes_cols = list(attribute_wise_dfs.keys())
    attributes_cols.sort()

    common_cols = [x_col, y_col, year_col, day_col]

    agg_df = agg_df[common_cols + attributes_cols]

    return agg_df


def aggregate_tifs_to_df(
    mask_tif_filepaths:list[str],
    roi_geom_filepath:str,
    ref_tif_filepath:str,
    catalogue_df:pd.DataFrame,
    csvs_folderpath:str,
    overwrite:bool = False,
):
    print('Converting tifs to csvs')
    catalogue_df = read_catalogue_and_create_csvs(
        mask_tif_filepaths = mask_tif_filepaths,
        roi_geom_filepath = roi_geom_filepath,
        ref_tif_filepath = ref_tif_filepath,
        catalogue_df = catalogue_df,
        csvs_folderpath = csvs_folderpath,
        overwrite = overwrite,
    )

    print('Aggregating csvs')
    aggregated_df = aggregate_csvs(
        catalogue_df = catalogue_df,
    )

    return aggregated_df


def create_xy_gdf(
    cropmask_tif_filepath:str,
    interp_tif_filepath:str,
    cropmask_col:str = 'crop',
):
    xy_gdf = utils.create_xy_gdf(
        mask_tif_filepaths=[
            cropmask_tif_filepath,
            interp_tif_filepath,
        ]
    )

    zero_interp_tif_filepath = utils.add_epochs_prefix(
        filepath = interp_tif_filepath,
        prefix = 'zero_',
    )
    utils.create_zero_tif(
        reference_tif_filepath = interp_tif_filepath,
        zero_tif_filepath = zero_interp_tif_filepath,
    )

    cropmask_xs, cropmask_ys, _, _ = \
    utils.get_mask_coords(
        mask_tif_filepaths=[
            cropmask_tif_filepath,
            zero_interp_tif_filepath,
        ]
    )

    os.remove(zero_interp_tif_filepath)
    
    xy_gdf[cropmask_col] = False

    xy_gdf.loc[
        xy_gdf.set_index(
            ['x', 'y']
        ).index.isin(
            pd.MultiIndex.from_arrays(
                [cropmask_xs, cropmask_ys]
            )
        ),
        cropmask_col
    ] = True

    return xy_gdf
