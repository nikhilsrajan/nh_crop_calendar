import sys
import pandas as pd
import os
import datetime
import geopandas as gpd
import tqdm
import rasterio
import rasterio.merge
import rasterio.warp
import numpy as np

sys.path.append('..')

import rsutils.utils as utils
import rsutils.s2_grid_utils as s2_grid_utils


CROPMASK_NODATA = 255
CROPMASK_ISCROP = 100
CROPMASK_ISNOTCROP = 0


def get_aezs(aez_gdf:gpd.GeoDataFrame, bounds_gdf:gpd.GeoDataFrame):
    worldcereal_aez_in_bound_gdf = gpd.sjoin(
        bounds_gdf.to_crs(aez_gdf.crs), 
        aez_gdf[['geometry', 'aez_id']],
    ).reset_index(drop=True)
    aez_ids = worldcereal_aez_in_bound_gdf['aez_id'].astype(str).to_list()
    return aez_ids


def create_worldcereal_files_catalogue_df(
    worldcereal_folderpath:str,
    filter_aezs=None,
    filter_types=None,
    filter_products=None,
):
    worldcereal_tif_catalogue_data = {
        'aez': [],
        'product': [],
        'startdate': [],
        'enddate': [],
        'type': [],
        'filepath': [],
    }

    for filepath in utils.get_all_files_in_folder(folderpath=worldcereal_folderpath, keep_extensions=['.tif']):
        filename = os.path.split(filepath)[1]
        filename_wo_ext = filename[:-4]
        aez, season, product, startdate, enddate, ftype = filename_wo_ext.split('_')
        worldcereal_tif_catalogue_data['aez'].append(str(aez))
        """
        previously we saved season and product into different columns:
        > worldcereal_tif_catalogue_data['season'].append(season)
        > worldcereal_tif_catalogue_data['product'].append(product)

        but for ease of coding, and avoiding the special cases where product repeats
        for different seasons, the change is made for product column to be unique by
        cancatenating season and product:
        > worldcereal_tif_catalogue_data['product'].append(season + '_' + product)
        """
        worldcereal_tif_catalogue_data['product'].append(season + '_' + product)
        worldcereal_tif_catalogue_data['startdate'].append(datetime.datetime.strptime(startdate, '%Y-%m-%d'))
        worldcereal_tif_catalogue_data['enddate'].append(datetime.datetime.strptime(enddate, '%Y-%m-%d'))
        worldcereal_tif_catalogue_data['type'].append(ftype)
        worldcereal_tif_catalogue_data['filepath'].append(filepath)

    worldcereal_tif_catalogue_df = pd.DataFrame(data=worldcereal_tif_catalogue_data)

    if filter_aezs is not None:
        if not isinstance(filter_aezs, list):
            raise ValueError('filter_aezs must be list')
        worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df[
            worldcereal_tif_catalogue_df['aez'].isin(filter_aezs)
        ]
    if filter_types is not None:
        if not isinstance(filter_types, list):
            raise ValueError('filter_types must be list')
        worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df[
            worldcereal_tif_catalogue_df['type'].isin(filter_types)
        ]
    if filter_products is not None:
        if not isinstance(filter_products, list):
            raise ValueError('filter_products must be list')
        worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df[
            worldcereal_tif_catalogue_df['product'].isin(filter_products)
        ]

    worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df.sort_values(by=['startdate', 'product']).reset_index(drop=True)

    return worldcereal_tif_catalogue_df


def crop_tif_to_each_shape_and_save(
    src_filepath:str, 
    shapes_gdf:gpd.GeoDataFrame, 
    output_folderpath:str, 
    id_col:str,
    overwrite:bool=False,
):
    data = {
        id_col: [],
        'tif_filepath': [],
    }
    for index, row in tqdm.tqdm(shapes_gdf.iterrows(), total=shapes_gdf.shape[0]):
        _id = row[id_col]
        dst_folderpath = os.path.join(output_folderpath, _id)
        os.makedirs(dst_folderpath, exist_ok=True)
        dst_filepath = utils.modify_filepath(
            filepath=src_filepath,
            prefix=f'{_id}_',
            new_folderpath=dst_folderpath,
        )
        if os.path.exists(dst_filepath) and not overwrite:
            data['id'].append(_id)
            data['tif_filepath'].append(dst_filepath)
            continue
        try:
            out_image, out_meta = utils.crop_tif(
                src_filepath=src_filepath, 
                shapes_gdf=gpd.GeoDataFrame(data={'geometry':[row['geometry']]}, crs=shapes_gdf.crs),
            )
            with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
                dst.write(out_image)
            del out_image, out_meta
        except ValueError as e:
            dst_filepath = None

        data['id'].append(_id)
        data['tif_filepath'].append(dst_filepath)

    tif_filepaths_df = pd.DataFrame(data=data)
    return tif_filepaths_df


def _divide_each_aez_into_s2grids(
    worldcereal_tif_catalogue_df:pd.DataFrame,
    bounds_s2_grids_gdf:gpd.GeoDataFrame,
    output_folderpath:str,
    overwrite:bool=False,
):
    s2_grid_cropped_tifs_dfs = {}

    for index, row in tqdm.tqdm(
        worldcereal_tif_catalogue_df.iterrows(), 
        total=worldcereal_tif_catalogue_df.shape[0],
    ):
        product = row['product']
        aez = row['aez']

        print(product, aez)

        tif_filepath = row['filepath']

        cropped_tif_filepaths_df = crop_tif_to_each_shape_and_save(
            src_filepath=tif_filepath,
            shapes_gdf=bounds_s2_grids_gdf,
            output_folderpath=output_folderpath,
            id_col='id',
            overwrite=overwrite,
        )

        s2_grid_cropped_tifs_dfs[(product, aez)] = cropped_tif_filepaths_df

    s2_grid_cropped_tifs_dfs_list = []
    for key, _df in s2_grid_cropped_tifs_dfs.items():
        product, aez = key
        _df['product'] = product
        _df['aez'] = str(aez)
        s2_grid_cropped_tifs_dfs_list.append(_df.dropna())
    
    cropmask_s2grid_tifs_df = pd.concat(s2_grid_cropped_tifs_dfs_list).reset_index(drop=True)

    return cropmask_s2grid_tifs_df


def _merge_divided_aez_into_s2grids(
    cropmask_s2grid_tifs_df:pd.DataFrame,
    aggregated_cropmask_folderpath:str,
    overwrite:bool=False,
):
    data = {
        'id': [],
        'product': [],
        'tif_filepath': [],
    }

    resampling = rasterio.merge.Resampling.nearest

    os.makedirs(aggregated_cropmask_folderpath, exist_ok=True)

    for _id, _product in tqdm.tqdm(set(zip(cropmask_s2grid_tifs_df['id'], cropmask_s2grid_tifs_df['product']))):
        _tif_filepaths = cropmask_s2grid_tifs_df[
            (cropmask_s2grid_tifs_df['id'] == _id) &
            (cropmask_s2grid_tifs_df['product'] == _product)
        ]['tif_filepath'].to_list()

        dst_filepath = os.path.join(aggregated_cropmask_folderpath, f'{_id}_{_product}.tif')

        if not os.path.exists(dst_filepath) or overwrite:
            out_image, out_transform = rasterio.merge.merge(
                _tif_filepaths,
                method=rasterio.merge.copy_max,
                resampling=resampling,
                nodata=CROPMASK_NODATA,
            )

            out_image[out_image != CROPMASK_ISCROP] = CROPMASK_ISNOTCROP

            with rasterio.open(_tif_filepaths[0]) as ref:
                out_meta = ref.meta.copy()

            out_meta.update({
                'count': out_image.shape[0],
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'nodata': CROPMASK_NODATA,
                'compress':'lzw',
            })

            with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
                dst.write(out_image)

            del out_image
        
        data['id'].append(_id)
        data['product'].append(_product)
        data['tif_filepath'].append(dst_filepath)

    aggregated_cropmask_tif_filepaths_df = pd.DataFrame(data=data)
    
    return aggregated_cropmask_tif_filepaths_df


def divide_worldcereal_cropmasks_into_s2grids(
    worldcereal_tif_catalogue_df:pd.DataFrame,
    bounds_s2_grids_gdf:gpd.GeoDataFrame,
    aez_s2_grid_folderpath:str,
    s2_grid_level_folderpath:str,
    overwrite:bool=False,
):
    cropmask_s2grid_tifs_df = _divide_each_aez_into_s2grids(
        worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df,
        bounds_s2_grids_gdf = bounds_s2_grids_gdf,
        output_folderpath = aez_s2_grid_folderpath,
        overwrite = overwrite,
    )

    print('Merging each cropmask aez-s2grid into s2grids:')
    aggregated_cropmask_tif_filepaths_df = _merge_divided_aez_into_s2grids(
        cropmask_s2grid_tifs_df = cropmask_s2grid_tifs_df,
        aggregated_cropmask_folderpath = s2_grid_level_folderpath,
        overwrite = overwrite,
    )

    return aggregated_cropmask_tif_filepaths_df


def merge_worldcereal_products(
    cropmask_tif_filepaths_df:pd.DataFrame,
    products_to_merge:list[str],
    merged_product_name:str,
    merged_product_folderpath:str,
    resampling = rasterio.merge.Resampling.nearest,
    overwrite:bool = False,
):
    pivoted_aggregated_cropmask_tif_filepaths_df = cropmask_tif_filepaths_df.pivot(
        index=['id'], columns=['product'], values=['tif_filepath']
    )

    os.makedirs(merged_product_folderpath, exist_ok=True)

    for index, row in tqdm.tqdm(
        pivoted_aggregated_cropmask_tif_filepaths_df.iterrows(),
        total=pivoted_aggregated_cropmask_tif_filepaths_df.shape[0],
    ):
        _tif_filepaths = []
        for _product in products_to_merge:
            _tif_filepaths.append(row[('tif_filepath', _product)])
        _merged_tif_filepath = os.path.join(merged_product_folderpath, f'{index}_{merged_product_name}.tif')

        if overwrite or not os.path.exists(_merged_tif_filepath):
            with rasterio.open(_tif_filepaths[0]) as ref:
                out_meta = ref.meta.copy()

            out_image, out_transform = rasterio.merge.merge(
                _tif_filepaths,
                method=rasterio.merge.copy_max,
                resampling=resampling,
                nodata=CROPMASK_NODATA,
            )

            out_meta.update({
                'count': out_image.shape[0],
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'nodata': CROPMASK_NODATA,
                'compress':'lzw',
            })

            with rasterio.open(_merged_tif_filepath, 'w', **out_meta) as dst:
                dst.write(out_image)

            del out_image
        
        pivoted_aggregated_cropmask_tif_filepaths_df.loc[index, ('tif_filepath', merged_product_name)] = _merged_tif_filepath

    cropmask_tif_filepaths_df = \
    pivoted_aggregated_cropmask_tif_filepaths_df.reset_index().melt(
        id_vars=[('id', '')]
    ).drop(
        columns=[None]
    ).rename(
        columns={('id',''): 'id', 'value': 'tif_filepath'}
    )

    return cropmask_tif_filepaths_df


def resample_to_ref_and_save(
    ref_filepath:str,
    src_filepath:str,
    dst_filepath:str,
    resampling = rasterio.warp.Resampling.average,
):
    with rasterio.open(ref_filepath) as ref:
        ref_meta = ref.meta.copy()
    
    with rasterio.open(src_filepath) as src:
        src_meta = src.meta.copy()
        src_image = src.read(1)

    src_image[src_image!=100] = 0
    dst_image = np.zeros((ref_meta['height'], ref_meta['width']))
    
    rasterio.warp.reproject(
        source = src_image,
        destination = dst_image,
        src_transform = src_meta['transform'],
        dst_transform = ref_meta['transform'],
        src_nodata = src_meta['nodata'],
        dst_nodata = CROPMASK_ISNOTCROP,
        src_crs = src_meta['crs'],
        dst_crs = ref_meta['crs'],
        resampling = resampling,
    )

    ref_meta['nodata'] = CROPMASK_ISNOTCROP

    with rasterio.open(dst_filepath, 'w', **ref_meta) as dst:
        dst.write(np.expand_dims(dst_image, axis=0))


def resample_cropmasks_to_ref(
    cropmask_tif_filepaths_df:pd.DataFrame,
    resampled_cropmasks_folderpath:str,
    ref_tif_filepath:str,
    overwrite:bool=False,
    resampling=rasterio.merge.Resampling.average
):
    data = {
        'id': [],
        'product': [],
        'tif_filepath': [],
    }

    os.makedirs(resampled_cropmasks_folderpath, exist_ok=True)

    for index, row in tqdm.tqdm(
        cropmask_tif_filepaths_df.iterrows(), 
        total=cropmask_tif_filepaths_df.shape[0],
    ):
        _id = row['id']
        product = row['product']
        cropmask_filepath = row['tif_filepath']

        resampled_cropmask_filepath = os.path.join(resampled_cropmasks_folderpath, f'resampled_{_id}_{product}.tif')

        if not os.path.exists(resampled_cropmask_filepath) or overwrite:
            resample_to_ref_and_save(
                ref_filepath=ref_tif_filepath,
                src_filepath=cropmask_filepath,
                dst_filepath=resampled_cropmask_filepath,
                resampling=resampling,
            )
        
        data['id'].append(_id)
        data['product'].append(product)
        data['tif_filepath'].append(resampled_cropmask_filepath)

    resampled_cropmask_tifs_df = pd.DataFrame(data=data)

    return resampled_cropmask_tifs_df


def merge_tifs(
    tif_filepaths:list[str], 
    dst_filepath:str, 
    bounds:tuple=None, 
    method=rasterio.merge.copy_max,
):
    with rasterio.open(tif_filepaths[0]) as src:
        out_meta = src.meta.copy()
    out_image, out_transform = rasterio.merge.merge(
        datasets=tif_filepaths,
        nodata=out_meta['nodata'],
        bounds=bounds,
        method=method,
    )
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
    })
    with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
        dst.write(out_image)


def merge_s2grid_tifs_by_product(
    products:list[str],
    cropmask_s2grid_tifs_df:pd.DataFrame,
    merged_cropmask_folderpath:str,
    bounds:tuple,
):
    data = {
        'product': [],
        'tif_filepath': [],
    }

    os.makedirs(merged_cropmask_folderpath, exist_ok=True)

    for product in tqdm.tqdm(products):
        _tif_filepaths = cropmask_s2grid_tifs_df[
            (cropmask_s2grid_tifs_df['product']==product)
        ]['tif_filepath'].to_list()
        merged_tif_filepath = os.path.join(merged_cropmask_folderpath, f'{product}.tif')
        merge_tifs(
            tif_filepaths=_tif_filepaths,
            dst_filepath=merged_tif_filepath,
            bounds=bounds,
        )
        data['product'].append(product)
        data['tif_filepath'].append(merged_tif_filepath)

    merged_cropmask_catalogue_df = pd.DataFrame(data=data)

    return merged_cropmask_catalogue_df


def inplace_coregister_with_cropped_ref_tif(
    tif_filepaths:list[str],
    bounds_gdf:gpd.GeoDataFrame,
    ref_tif_filepath:str,
    new_folderpath:str = None,
):
    cropped_ref_ndarray, cropped_ref_meta = utils.crop_tif(
        src_filepath = ref_tif_filepath,
        shapes_gdf = bounds_gdf,
    )

    cropped_ref_ndarray = np.zeros(shape=cropped_ref_ndarray.shape)
    cropped_ref_meta['dtype'] = rasterio.uint8
    cropped_ref_meta['nodata'] = 0

    zero_cropped_ref_tif_filepath = utils.modify_filepath(
        filepath = ref_tif_filepath,
        prefix = 'cropped_zero_',
    )

    with rasterio.open(zero_cropped_ref_tif_filepath, 'w', **cropped_ref_meta) as dst:
        dst.write(cropped_ref_ndarray)

    # for index, row in catalogue_df.iterrows():
    #     tif_filepath = row[tif_filepath_col]
    out_tif_filepaths = []
    for tif_filepath in tif_filepaths:
        out_tif_filepath = utils.modify_filepath(
            filepath = tif_filepath,
            new_folderpath = new_folderpath,
        )
        utils.coregister(
            src_filepath = tif_filepath,
            dst_filepath = out_tif_filepath,
            reference_zero_filepath = zero_cropped_ref_tif_filepath,
        )
        out_tif_filepaths.append(out_tif_filepath)
    
    return out_tif_filepaths


def resample_worldcereal_cropmasks(
    roi_geom_filepath:str,
    worldcereal_folderpath:str,
    worldcereal_aez_filepath:str,
    ref_tif_filepath:str,
    aez_s2_grid_folderpath:str,
    s2_grid_level_folderpath:str,
    merged_product_folderpath:str,
    resampled_cropmasks_folderpath:str,
    output_folderpath:str,
    merge_products_dict:dict[str,list[str]], # merged_product_name: products_to_merge
    out_products:list[str],
    s2_grid_res:int = 4,
    overwrite:bool = False,
):
    roi_geom_gdf = gpd.read_file(roi_geom_filepath)

    bounds_gdf = utils.get_actual_bounds_gdf(
        src_filepath = ref_tif_filepath,
        shapes_gdf = roi_geom_gdf,
    )

    bounds = tuple(bounds_gdf.bounds.iloc[0])
    ref_cropped_ndarray, ref_cropped_meta = utils.crop_tif(src_filepath=ref_tif_filepath, shapes_gdf=bounds_gdf)
    outshape = ref_cropped_ndarray.shape[1], ref_cropped_ndarray.shape[2]

    aez_gdf = gpd.read_file(worldcereal_aez_filepath)

    aez_ids = get_aezs(aez_gdf=aez_gdf, bounds_gdf=bounds_gdf)

    _products_to_merge = []
    _merged_product_names = []
    for _merged_product_name, __products_to_merge in merge_products_dict.items():
        _products_to_merge += __products_to_merge
        _merged_product_names.append(_merged_product_name)

    products_to_work_with = list((set(out_products) | set(_products_to_merge)) - set(_merged_product_names))

    worldcereal_tif_catalogue_df = create_worldcereal_files_catalogue_df(
        worldcereal_folderpath=worldcereal_folderpath,
        filter_aezs=aez_ids,
        filter_types=['classification'],
        filter_products=products_to_work_with,
    )

    # divide bounds_gdf into s2_grids of res=s2_grid_res
    bounds_s2_grids_gdf = s2_grid_utils.get_s2_grids_gdf(
        geojson_epsg_4326=bounds_gdf['geometry'][0],
        res=s2_grid_res,
    )
    bounds_s2_grids_gdf = gpd.overlay(bounds_gdf, bounds_s2_grids_gdf)

    print(f"Dividing each aez's cropmask for products={products_to_work_with} by s2 grids:")
    cropmask_tif_filepaths_df = divide_worldcereal_cropmasks_into_s2grids(
        worldcereal_tif_catalogue_df = worldcereal_tif_catalogue_df,
        bounds_s2_grids_gdf = bounds_s2_grids_gdf,
        aez_s2_grid_folderpath = aez_s2_grid_folderpath,
        s2_grid_level_folderpath = s2_grid_level_folderpath,
        overwrite = overwrite,
    )

    # merge list of products into single tif
    for merged_product_name, products_to_merge in merge_products_dict.items():
        print(f"Merging different product -- {products_to_merge} -> {merged_product_name}")
        cropmask_tif_filepaths_df = merge_worldcereal_products(
            cropmask_tif_filepaths_df = cropmask_tif_filepaths_df,
            merged_product_folderpath = merged_product_folderpath,
            products_to_merge = products_to_merge,
            merged_product_name = merged_product_name,
            overwrite = overwrite,
        )

    cropmask_tif_filepaths_df = cropmask_tif_filepaths_df[
        cropmask_tif_filepaths_df['product'].isin(out_products)
    ]

    print(f"Resampling cropmasks for products={out_products} to target resolution")
    resampled_cropmask_tifs_df = resample_cropmasks_to_ref(
        cropmask_tif_filepaths_df = cropmask_tif_filepaths_df,
        resampled_cropmasks_folderpath = resampled_cropmasks_folderpath,
        ref_tif_filepath = ref_tif_filepath,
        overwrite = overwrite,
    )

    print(f"Merging {out_products} to bounds:")
    merged_cropmask_catalogue_df = merge_s2grid_tifs_by_product(
        products = out_products,
        cropmask_s2grid_tifs_df = resampled_cropmask_tifs_df,
        merged_cropmask_folderpath = output_folderpath,
        bounds = bounds,
    )

    print(f"Co-register merged tifs with cropped ref_tif_filepath:")
    inplace_coregister_with_cropped_ref_tif(
        tif_filepaths = merged_cropmask_catalogue_df['tif_filepath'],
        bounds_gdf = bounds_gdf,
        ref_tif_filepath = ref_tif_filepath,
    )
    
    return merged_cropmask_catalogue_df


if __name__ == '__main__':
    bounding_filepath = '../../data/outputs/france_modis_bounding.geojson'
    aez_geojson_filepath = '../../data/worldcereal/WorldCereal_AEZ.geojson'
    reference_geotiff = '../../data/GEOGLAM-BACS_v1.0.0/Percent_Spring_Wheat.tif'

    worldcereal_folderpath = '../../data/worldcereal/'
    aez_s2_grid_folderpath = '../../data/outputs/s2_grid_level/worldcereal/'
    s2_grid_level_folderpath = '../../data/outputs/s2_grid_level/aggregated_worldcereal/'
    merged_product_folderpath = '../../data/outputs/s2_grid_level/aggregated_worldcereal'
    resampled_cropmasks_folderpath = '../../data/outputs/s2_grid_level/resampled_cropmasks/'
    output_folderpath = '../../data/outputs/merged_WC_cropmask/'

    merged_cropmask_catalogue_df = resample_worldcereal_cropmasks(
        roi_geom_filepath = bounding_filepath,
        worldcereal_folderpath = worldcereal_folderpath,
        worldcereal_aez_filepath = aez_geojson_filepath,
        ref_tif_filepath = reference_geotiff,
        aez_s2_grid_folderpath = aez_s2_grid_folderpath,
        s2_grid_level_folderpath = s2_grid_level_folderpath,
        merged_product_folderpath = merged_product_folderpath,
        resampled_cropmasks_folderpath = resampled_cropmasks_folderpath,
        output_folderpath = output_folderpath,
        products_to_merge = ['springcereals', 'wintercereals'],
        merged_product_name = 'cereals',
        out_products = ['cereals', 'temporarycrops'],
        s2_grid_res = 4,
        overwrite = False,
    )
