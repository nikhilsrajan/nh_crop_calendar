import pandas as pd
import os
import sys
import numpy as np
import geopandas as gpd
import tqdm
import rasterio
import skgstat as skg

sys.path.append('..')

import rsutils.utils as utils
import rsutils.rich_data_filter as rich_data_filter
import rsutils.modify_bands as modify_bands

import presets


ATTRIBUTE_COL = 'attribute'
TIF_FILEPATH_COL = 'tif_filepath'


def create_catalogue_df(
    folderpath:str, 
    filename_parser, 
    ignore_extensions:list[str] = None,
    keep_extensions:list[str]=['.tif'],
):
    data = {TIF_FILEPATH_COL: []}
    for filepath in utils.get_all_files_in_folder(
        folderpath=folderpath,
        ignore_extensions=ignore_extensions,
        keep_extensions=keep_extensions,
    ):
        data[TIF_FILEPATH_COL].append(filepath)
        filename = os.path.split(filepath)[1]
        parsed = filename_parser(filename=filename)
        for key, value in parsed.items():
            if key not in data.keys():
                data[key] = []
            data[key].append(value)
    catalogue_df = pd.DataFrame(data=data)
    return catalogue_df


def get_cropped_shapes(
    catalogue_df:pd.DataFrame,
    bounds_gdf:gpd.GeoDataFrame,
):
    shapes = set()
    for filepath in catalogue_df[TIF_FILEPATH_COL]:
        if filepath is None:
            continue
        out_image, out_meta = utils.crop_tif(
            src_filepath=filepath,
            shapes_gdf=bounds_gdf,
        )
        shapes.add(out_image.shape)
        del out_image, out_meta

    return list(shapes)


def get_all_shapes(catalogue_df:pd.DataFrame):
    shapes = set()
    for index, row in catalogue_df.iterrows():
        with rasterio.open(row[TIF_FILEPATH_COL]) as src:
            shapes.add(src.read().shape)
    return list(shapes)


def create_stack(
    catalogue_df:pd.DataFrame,
    nodata,
    bounds_gdf:gpd.GeoDataFrame = None,
    crop_and_read:bool = False,
):
    if crop_and_read and bounds_gdf is None:
        raise ValueError(f'bounds_gdf can not be None when crop_and_read is True.')
    
    years = catalogue_df[presets.YEAR].sort_values().unique()
    days = catalogue_df[presets.DAY].sort_values().unique()
    d_days, d_counts = np.unique(days[1:] - days[:-1], return_counts=True)
    max_occurring_d = d_days[np.argmax(d_counts)]
    
    all_days = np.arange(days.min(), days.max() + max_occurring_d, max_occurring_d)

    if crop_and_read:
        cropped_shapes = get_cropped_shapes(catalogue_df=catalogue_df, bounds_gdf=bounds_gdf)
    else:
        cropped_shapes = get_all_shapes(catalogue_df=catalogue_df)

    if len(cropped_shapes) != 1:
        raise ValueError(f'Shapes are not consistent. Got shapes: {cropped_shapes}')

    expected_shape = cropped_shapes[0]


    year_day_filepath = dict(zip(
        zip(catalogue_df[presets.YEAR], catalogue_df[presets.DAY]),
        catalogue_df[TIF_FILEPATH_COL]
    ))

    stack = []
    for year in years:
        yearwise_stack = []
        for day in all_days:
            if (year, day) not in year_day_filepath.keys():
                yearwise_stack.append(np.full(shape=expected_shape, fill_value=nodata))
            else:
                tif_filepath = year_day_filepath[(year, day)]
                if crop_and_read:
                    out_image, out_meta = utils.crop_tif(
                        src_filepath=tif_filepath,
                        shapes_gdf=bounds_gdf,
                    )
                else:
                    with rasterio.open(tif_filepath) as src:
                        out_image = src.read()
                        out_meta = src.meta.copy()
                yearwise_stack.append(out_image)
                del out_image
        yearwise_stack = np.concatenate(yearwise_stack, axis=0)
        stack.append(yearwise_stack)
        del yearwise_stack
    stack = np.stack(stack, axis=0)

    metadata = {
        'data_shape_desc': ('years', 'days', 'height', 'width'),
        'years': list(years),
        'days': list(days),
        'geotiff_metadata': out_meta,
    }

    return stack, metadata


def get_rich_data_stats_df(
    stack:np.ndarray,
    metadata:dict,
    lower_cap,
    upper_cap,
    select_mask_ndarray:np.ndarray=None,
):
    n_years, n_days, n_pixels_x, n_pixels_y = stack.shape

    if select_mask_ndarray is None:
        select_mask_ndarray = np.ones(shape=(n_pixels_x, n_pixels_y))

    select_xs, select_ys = np.where(select_mask_ndarray == 1)

    stack_ts = stack[:,:,select_xs,select_ys]
    del stack

    unavailable_data = np.zeros(shape=stack_ts.shape)
    unavailable_data[(stack_ts <= lower_cap) | (stack_ts >= upper_cap)] = 1

    n_years, n_days, n_pixels = unavailable_data.shape

    total_data, max_continuous_unavailable_data = rich_data_filter.get_rich_data_stats(
        unavailable_data=np.expand_dims(unavailable_data, axis=(3,4))
    )

    year_index = dict(zip(metadata['years'], range(len(metadata['years']))))
    unavailable_data_stats_df = pd.DataFrame(data={
        'year': [year for year in year_index.keys() for _ in range(n_pixels)],
        'x': np.tile(select_xs, reps=n_years),
        'y': np.tile(select_ys, reps=n_years),
        'total_data': total_data.flatten(),
        'max_continuous_unavailable_data': max_continuous_unavailable_data[:,0].flatten(),
        'start_index': max_continuous_unavailable_data[:,1].flatten(),
    })

    return unavailable_data_stats_df


def get_rich_data_mask(
    stack:np.ndarray,
    metadata:dict,
    lower_cap,
    upper_cap,
    min_total_data:int,
    max_max_continuous_unavailable_data:int,
    select_mask_ndarray:np.ndarray=None,
):
    n_years, n_days, n_pixels_x, n_pixels_y = stack.shape
    year_index = dict(zip(metadata['years'], range(len(metadata['years']))))

    unavailable_data_stats_df = get_rich_data_stats_df(
        stack = stack,
        metadata = metadata,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
        select_mask_ndarray = select_mask_ndarray,
    )

    rich_xy_df = unavailable_data_stats_df[
        (unavailable_data_stats_df['max_continuous_unavailable_data'] <= max_max_continuous_unavailable_data) &
        (unavailable_data_stats_df['total_data'] >= min_total_data)
    ][['year', 'x', 'y']]

    rich_data_mask = np.ones(shape=(n_years, n_pixels_x, n_pixels_y))
    for year, year_index in year_index.items():
        _xs = rich_xy_df[rich_xy_df['year']==year]['x']
        _ys = rich_xy_df[rich_xy_df['year']==year]['y']
        rich_data_mask[year_index, _xs, _ys] = 0

    return rich_data_mask


def temporally_interpolate_stack(
    stack:np.ndarray,
    metadata:dict,
    nodata,
    lower_cap,
    upper_cap,
):
    n_years, n_days, n_pixels_x, n_pixels_y = stack.shape
    year_index = dict(zip(metadata['years'], range(len(metadata['years']))))

    interp_stack = np.zeros(shape=stack.shape)

    for year, year_index in year_index.items():
        _interp_stack, _ = modify_bands.mask_invalid_and_interpolate(
            bands=np.expand_dims(stack[year_index], axis=(0, 4)),
            band_indices={'band': 0},
            upper_cap=upper_cap,
            lower_cap=lower_cap,
            mask_value=nodata,
        )
        _interp_stack = _interp_stack[0,:,:,:,0]
        interp_stack[year_index] = _interp_stack
    
    return interp_stack


def write_out_stack(
    stack:np.ndarray,
    metadata:dict,
    folderpath:str,
    attribute:str,
    attribute_col:str,
    tif_filepath_col:str,
    overwrite:bool=False,
):
    n_years, n_days, n_pixels_x, n_pixels_y = stack.shape

    years = metadata['years']
    days = metadata['days']
    out_meta = metadata['geotiff_metadata']
    
    if n_years != len(years):
        raise ValueError(f'n_years={n_years} != len(years)={len(years)}')

    if n_days != len(days):
        raise ValueError(f'n_days={n_days} != len(days)={len(days)}')
    
    data = {
        'year': [],
        'day': [],
        attribute_col: [],
        tif_filepath_col: [],
    }

    os.makedirs(folderpath, exist_ok=True)
    
    for year_index, year in enumerate(years):
        for day_index, day in enumerate(days):
            tif_filepath = os.path.join(folderpath, f'{attribute}_{year}_{day}.tif')
            if not os.path.exists(tif_filepath) or overwrite:
                with rasterio.open(tif_filepath, 'w', **out_meta) as dst:
                    dst.write(np.expand_dims(stack[year_index, day_index], axis=0))
            data['year'].append(year)
            data['day'].append(day)
            data[attribute_col].append(attribute)
            data[tif_filepath_col].append(tif_filepath)

    catalogue_df = pd.DataFrame(data=data)
    return catalogue_df


def raw_temporally_interpolate_files(
    catalogue_df:pd.DataFrame,
    crop_and_read:bool,
    bounds_gdf:gpd.GeoDataFrame,
    nodata,
    cropmask:np.ndarray,
    lower_cap,
    upper_cap,
    rich_data_min_total_data:int,
    rich_data_max_max_continuous_unavailable_data:int,
    out_folderpath:str,
    attribute:str,
    attribute_col:str,
    tif_filepath_col:str,
    overwrite:bool=False,
):
    # 1. create stack
    stack, metadata = create_stack(
        catalogue_df = catalogue_df,
        bounds_gdf = bounds_gdf,
        nodata = nodata,
        crop_and_read = crop_and_read,
    )
    n_years, n_days, n_pixels_x, n_pixels_y = stack.shape

    if cropmask.shape != (n_pixels_x, n_pixels_y):
        raise ValueError(
            "stack's height and width don't match with cropmask.\n"
            f"cropmask.shape = {cropmask.shape}\n"
            f"(n_pixels_x, n_pixels_y) = {(n_pixels_x, n_pixels_y)}"
        )
    
    # 2. create rich data masks // filter out not-rich pixels
    rich_data_mask = get_rich_data_mask(
        stack = stack,
        metadata = metadata,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
        min_total_data = rich_data_min_total_data,
        max_max_continuous_unavailable_data = rich_data_max_max_continuous_unavailable_data,
        select_mask_ndarray = cropmask,
    )

    rich_year_idx, rich_pixels_x_idx, rich_pixels_y_idx = np.where(rich_data_mask == 1)

    stack[rich_year_idx, :, rich_pixels_x_idx, rich_pixels_y_idx] = nodata

    # 3. temporally interpolate stack
    interp_stack = temporally_interpolate_stack(
        stack = stack,
        metadata = metadata,
        nodata = nodata,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
    )
    del stack

    # 4. write out interpolated tifs
    temp_interp_catalogue_df = write_out_stack(
        stack = interp_stack,
        metadata = metadata,
        folderpath = out_folderpath,
        attribute = attribute,
        overwrite = overwrite,
        attribute_col = attribute_col,
        tif_filepath_col = tif_filepath_col,
    )
    del interp_stack

    return temp_interp_catalogue_df


def compute_longlat(x:float, y:float, shift:float, transform:rasterio.Affine)->tuple[float]:
    long, lat = transform * (y +  shift, x + shift)
    return long, lat


def compute_longlat_for_df(
    df:pd.DataFrame,
    shift:float,
    transform:rasterio.Affine,
    x_col:str='x', 
    y_col:str='y', 
    long_col:str='longitude',
    lat_col:str='latitude',
):
    long_lat_df = pd.DataFrame(data=[
        compute_longlat(x=x, y=y, shift=shift, transform=transform) 
        for x, y in zip(df[x_col], df[y_col])
    ], columns=[long_col, lat_col], index=df.index)
    return pd.concat([df, long_lat_df], axis=1)


def reconstruct_tif_from_df(
    df:pd.DataFrame, 
    value_col:str, 
    out_meta:dict,
    x_col:str='x', 
    y_col:str='y',
):
    out_image = np.full(
        shape=(1, out_meta['height'], out_meta['width']), 
        fill_value=out_meta['nodata'],
        dtype=out_meta['dtype'],
    )
    out_image[:, df[x_col], df[y_col]] = df[value_col]

    out_meta['nodata'] = out_meta['nodata']
    out_meta['count'] = 1
    out_meta['dtype'] = out_meta['dtype']
    
    return out_image, out_meta


def read_spatially_interpolate_write(
    value_tif_filepath:str,
    cropmask_tif_filepath:str,
    interpmask_tif_filepath:str,
    out_tif_filepath:str,    
    attribute:str,
    val_nodata,
    upper_cap,
    lower_cap,
    model:str = 'spherical',
    n_lags:int = 20,
    bin_func:str = 'even',
    min_points:int = 5,
    max_points:int = 15,
    mode:str = 'exact',
    mask_nodata = 0,
    shift:float = 0.5,
):
    if attribute not in presets.VALID_NORMALISER_KEYS:
        raise ValueError(f'Invalid attribute={attribute}. attribute must be from {presets.VALID_NORMALISER_KEYS} for normalising.')
    
    if attribute not in presets.VALID_DENORMALISER_KEYS:
        raise ValueError(f'Invalid attribute={attribute}. attribute must be from {presets.VALID_DENORMALISER_KEYS} for denormalising.')

    with rasterio.open(cropmask_tif_filepath) as src:
        crop_mask = src.read()
        crop_mask_meta = src.meta.copy()
    
    with rasterio.open(interpmask_tif_filepath) as src:
        interp_mask = src.read()
        interp_mask_meta = src.meta.copy()
    
    with rasterio.open(value_tif_filepath) as src:
        val_ndarray = src.read()
        val_meta = src.meta.copy()

    for key in ['height', 'width', 'crs', 'transform']:
        if crop_mask_meta[key] != interp_mask_meta[key] or \
            interp_mask_meta[key] != val_meta[key]:
            raise ValueError(
                f'key {key} does not match across tifs.\n'
                f'crop_mask_meta[{key}] = {crop_mask_meta[key]}\n'
                f'interp_mask_meta[{key}] = {interp_mask_meta[key]}\n'
                f'val_meta[{key}] = {val_meta[key]}\n'
            )

    crop_xs, crop_ys = np.where(crop_mask[0] != mask_nodata)
    interp_xs, interp_ys = np.where(interp_mask[0] != mask_nodata)

    crop_value_df = pd.DataFrame(data={
        'x': crop_xs,
        'y': crop_ys,
        attribute: val_ndarray[0, crop_xs, crop_ys],
    })

    crop_value_df = crop_value_df[
        (crop_value_df[attribute] > lower_cap) &
        (crop_value_df[attribute] < upper_cap)
    ]
    
    interp_ndvi_df = pd.DataFrame(data={
        'x': interp_xs,
        'y': interp_ys,
    })

    transform = val_meta['transform']

    crop_value_df = compute_longlat_for_df(
        df=crop_value_df, shift=shift, transform=transform,
    )

    interp_ndvi_df = compute_longlat_for_df(
        df=interp_ndvi_df, shift=shift, transform=transform,
    )

    val_col = f'normalised_{attribute}'

    crop_value_df[val_col] = presets.NORMALISERS[attribute](x=crop_value_df[attribute])

    xy_cols = ['longitude', 'latitude']

    V = skg.Variogram(
        coordinates = crop_value_df[xy_cols].values,
        values = crop_value_df[val_col].values,
        model = model,
        n_lags = n_lags,
        bin_func = bin_func,
    )

    OK = skg.OrdinaryKriging(V, min_points=min_points, max_points=max_points, mode=mode)

    interp_ndvi_df[val_col] = OK.transform(
        interp_ndvi_df[xy_cols].values,
    )

    interp_ndvi_df[attribute] = presets.DENORMALISERS[attribute](x=interp_ndvi_df[val_col])
    interp_ndvi_df[attribute] = interp_ndvi_df[attribute].fillna(val_nodata).astype(int)

    out_image, out_meta = reconstruct_tif_from_df(
        df = interp_ndvi_df,
        value_col = attribute,
        out_meta = val_meta,
    )

    with rasterio.open(out_tif_filepath, 'w', **out_meta) as dst:
        dst.write(out_image)


def raw_spatially_interpolate_files(
    catalogue_df:pd.DataFrame,
    tif_filepath_col:str,
    attribute_col:str,
    out_folderpath:str,
    cropmask_tif_filepath:str,
    interpmask_tif_filepath:str,
    val_nodata,
    upper_cap,
    lower_cap,
    model:str = 'spherical',
    n_lags:int = 20,
    bin_func:str = 'even',
    min_points:int = 5,
    max_points:int = 15,
    mode:str = 'exact',
    mask_nodata = 0,
    shift:float = 0.5,
    overwrite:bool = False,
):
    os.makedirs(out_folderpath, exist_ok=True)

    data = { col : [] for col in catalogue_df.columns }

    for index, row in tqdm.tqdm(
        catalogue_df.iterrows(),
        total=catalogue_df.shape[0],
    ):
        value_tif_filepath = row[tif_filepath_col]
        attribute = row[attribute_col]
        out_tif_filepath = utils.modify_filepath(
            filepath = value_tif_filepath,
            new_folderpath = out_folderpath,
        )

        if not os.path.exists(out_tif_filepath) or overwrite:
            read_spatially_interpolate_write(
                value_tif_filepath = value_tif_filepath,
                cropmask_tif_filepath = cropmask_tif_filepath,
                interpmask_tif_filepath = interpmask_tif_filepath,
                out_tif_filepath = out_tif_filepath,
                attribute = attribute,
                val_nodata = val_nodata,
                upper_cap = upper_cap,
                lower_cap = lower_cap,
                model = model,
                n_lags = n_lags,
                bin_func = bin_func,
                min_points = min_points,
                max_points = max_points,
                mode = mode,
                mask_nodata = mask_nodata,
                shift = shift,
            )

        for col in catalogue_df.columns:
            if col not in [tif_filepath_col]:
                data[col].append(row[col])
        data[tif_filepath_col].append(out_tif_filepath)

    out_catalogue_df = pd.DataFrame(data=data)
    
    return out_catalogue_df


def spatially_interpolate_files(
    roi_geom_filepath:str,
    ref_tif_filepath:str,
    folderpath:str,
    attribute:str,
    nodata,
    cropmask_tif_filepath:str,
    interp_tif_filepath:str,
    mask_nodata,
    lower_cap,
    upper_cap,
    rich_data_min_total_data:int,
    rich_data_max_max_continuous_unavailable_data:int,
    t_interp_folderpath:str,
    st_interp_folderpath:str,
    tst_interp_folderpath:str,
    years:list[int] = None,
    overwrite_t_interp:bool = False,
    overwrite_st_interp:bool = False,
    overwrite_tst_interp:bool = False,
    model:str = 'spherical',
    n_lags:int = 20,
    bin_func:str = 'even',
    min_points:int = 5,
    max_points:int = 15,
    mode:str = 'exact',
    shift:float = 0.5,
):  
    if attribute not in presets.VALID_PARSER_KEYS:
        raise ValueError(
            f'Invalid attribute={attribute}. Must be from {presets.VALID_PARSER_KEYS}.'
        )
    
    with rasterio.open(cropmask_tif_filepath) as src:
        cropmask = src.read()

    if cropmask.shape[0] != 1:
        raise ValueError(f'cropmask_tif_filepath has more than 1 band.')

    with rasterio.open(interp_tif_filepath) as src:
        interp = src.read()

    if cropmask.shape != interp.shape:
        raise ValueError(f'shapes of cropmask_tif_filepath and interp_tif_filepath do not match.')

    cropmask = cropmask[0]
    interp = interp[0]

    roi_geom_gdf = gpd.read_file(roi_geom_filepath)

    bounds_gdf = utils.get_actual_bounds_gdf(
        src_filepath = ref_tif_filepath,
        shapes_gdf = roi_geom_gdf,
    )

    catalogue_df = create_catalogue_df(
        folderpath=folderpath,
        filename_parser = presets.PARSERS[attribute],
        keep_extensions=['.tif'],
    )

    if years is not None:
        catalogue_df = catalogue_df[catalogue_df[presets.YEAR].isin(years)]

    print('Temporally interpolating cropmask pixels')
    t_interp_catalogue_df = raw_temporally_interpolate_files(
        catalogue_df = catalogue_df,
        crop_and_read = True,
        bounds_gdf = bounds_gdf,
        nodata = nodata,
        cropmask = cropmask,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
        rich_data_min_total_data = rich_data_min_total_data,
        rich_data_max_max_continuous_unavailable_data = rich_data_max_max_continuous_unavailable_data,
        out_folderpath = t_interp_folderpath,
        attribute = attribute,
        tif_filepath_col = TIF_FILEPATH_COL,
        attribute_col = ATTRIBUTE_COL,
        overwrite = overwrite_t_interp,
    )

    print('Spatially interpolating from cropmask to interpmask')
    st_interp_catalogue_df = raw_spatially_interpolate_files(
        catalogue_df = t_interp_catalogue_df,
        tif_filepath_col = TIF_FILEPATH_COL,
        attribute_col = ATTRIBUTE_COL,
        out_folderpath = st_interp_folderpath,
        cropmask_tif_filepath = cropmask_tif_filepath,
        interpmask_tif_filepath = interp_tif_filepath,
        val_nodata = nodata,
        upper_cap = upper_cap,
        lower_cap = lower_cap,
        model = model,
        n_lags = n_lags,
        bin_func = bin_func,
        min_points = min_points,
        max_points = max_points,
        mode = mode,
        mask_nodata = mask_nodata,
        shift = shift,
        overwrite = overwrite_st_interp,
    )

    print('Temporally interpolating interpmask pixels')
    tst_interp_catalogue_df = raw_temporally_interpolate_files(
        catalogue_df = st_interp_catalogue_df,
        crop_and_read = False,
        bounds_gdf = None,
        nodata = nodata,
        cropmask = cropmask,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
        rich_data_min_total_data = rich_data_min_total_data,
        rich_data_max_max_continuous_unavailable_data = rich_data_max_max_continuous_unavailable_data,
        out_folderpath = tst_interp_folderpath,
        attribute = attribute,
        tif_filepath_col = TIF_FILEPATH_COL,
        attribute_col = ATTRIBUTE_COL,
        overwrite = overwrite_tst_interp,
    )

    return tst_interp_catalogue_df


if __name__ == '__main__':
    ...