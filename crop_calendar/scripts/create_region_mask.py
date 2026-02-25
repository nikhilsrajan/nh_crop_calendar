import numpy as np
import rasterio
import rasterio.dtypes
import rasterio.mask
import geopandas as gpd

import sys
sys.path.append('..')

import rsutils.utils


def create_region_mask(
    mask_tif_filepath:str,
    ref_tif_filepath:str, 
    shapes_gdf:gpd.GeoDataFrame, 
    all_touched:bool=True,
):
    rsutils.utils.create_fill_tif(
        reference_tif_filepath = ref_tif_filepath,
        out_tif_filepath = mask_tif_filepath,
        fill_value = 1,
        nodata = 0,
    )

    with rasterio.open(mask_tif_filepath) as src:
        out_meta = src.meta.copy()
        src_crs_shapes_gdf = shapes_gdf.to_crs(src.crs)
        shapes = src_crs_shapes_gdf['geometry'].to_list()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=False, nodata=0, all_touched=all_touched,
        )

    out_meta['dtype'] = 'uint8'
    out_meta['nodata'] = 0
    out_meta['count'] = 1
    with rasterio.open(mask_tif_filepath, 'w', **out_meta) as dst:
        dst.write(out_image)


for project in [
    # 'Morocco', 'Latvia', 'Estonia', 'Portugal', 'south-east-africa'
    'Malawi'
]:
    shapefilename = project + '.geojson'
    if project == 'south-east-africa':
        shapefilename = 'bounds-2012-01-01-nc.geojson'
    elif project == 'Malawi':
        shapefilename = 'mwi_adm_nso_hotosm_20230405_shp/mwi_admbnda_adm0_nso_hotosm_20230405.shp'
    create_region_mask(
        mask_tif_filepath = f'../data/outputs/{project}/region_mask.tif',
        ref_tif_filepath = f'../data/outputs/{project}/resampled_worldcereal/maize.tif',
        shapes_gdf = gpd.read_file(f'../data/outputs/{project}/{shapefilename}'),
    )