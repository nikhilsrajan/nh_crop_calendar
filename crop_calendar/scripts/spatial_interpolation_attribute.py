import os
import argparse

import sys
sys.path.append('..')

import spatially_interpolate_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python spatial_interpolation_attribute.py',
        description = (
            'Script to interpolate attributes spatially.'
        ),
    )

    nodata = 0
    mask_nodata = 0
    lower_cap = 0
    upper_cap = 250
    rich_data_min_total_data = 23
    rich_data_max_max_continuous_unavailable_data = 7


    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('ref_tif_filepath', help='/path/to/reference_raster')
    parser.add_argument('cropmask_filepath', help='/path/to/cropmask')
    parser.add_argument('interpmask_filepath', help='/path/to/interpmask')
    parser.add_argument('years', help='comma separated years (YYYY). Example: 2019,2020,2018')
    parser.add_argument('attribute_folderpath', help='/path/to/attribute_rasters')
    parser.add_argument('attribute', help='attribute name')
    parser.add_argument('t_interp_folderpath', help='/path/to/t_interp_outputs')
    parser.add_argument('st_interp_folderpath', help='/path/to/st_interp_outputs')
    parser.add_argument('tst_interp_folderpath', help='/path/to/tst_interp_outputs')
    parser.add_argument('--overwrite-t-interp', action='store_true')
    parser.add_argument('--overwrite-st-interp', action='store_true')
    parser.add_argument('--overwrite-tst-interp', action='store_true')

    args = parser.parse_args()

    years = [int(year_str) for year_str in args.years.split(',')]

    t_interp_catalogue_df, st_interp_catalogue_df, tst_interp_catalogue_df \
    = spatially_interpolate_files.spatially_interpolate_files(
        roi_geom_filepath = args.roi_filepath,
        ref_tif_filepath = args.ref_tif_filepath,
        attribute_folderpath = args.attribute_folderpath,
        attribute = args.attribute,
        nodata = nodata,
        cropmask_tif_filepath = args.cropmask_filepath,
        interp_tif_filepath = args.interpmask_filepath,
        mask_nodata = mask_nodata,
        lower_cap = lower_cap,
        upper_cap = upper_cap,
        rich_data_min_total_data = rich_data_min_total_data,
        rich_data_max_max_continuous_unavailable_data = rich_data_max_max_continuous_unavailable_data,
        years = years,
        t_interp_folderpath = args.t_interp_folderpath,
        overwrite_t_interp = args.overwrite_t_interp,
        st_interp_folderpath = args.st_interp_folderpath,
        overwrite_st_interp = args.overwrite_st_interp,
        tst_interp_folderpath = args.tst_interp_folderpath,
        overwrite_tst_interp = args.overwrite_tst_interp,
    )