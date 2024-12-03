import os
import pandas as pd
import argparse

import sys
sys.path.append('..')

import aggregate_tifs_to_df as at2d
import create_weather_data_catalogue as cwdc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python aggregate_tifs.py',
        description = (
            'Script to aggregate tifs to df.'
        ),
    )

    parser.add_argument('prefix')
    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('ref_tif_filepath', help='/path/to/reference_raster')
    parser.add_argument('cropmask_filepath', help='/path/to/cropmask')
    parser.add_argument('interpmask_filepath', help='/path/to/interpmask')
    parser.add_argument('years', help='comma separated years (YYYY). Example: 2019,2020,2018')
    parser.add_argument('weather_catalog', help='/path/to/weather_catalog')
    parser.add_argument('csv_folderpath', help='/path/to/csvs')
    parser.add_argument('export_folderpath', help='/path/to/export')
    
    args = parser.parse_args()

    years = [int(year_str) for year_str in args.years.split(',')]

    expected_output_filepaths = [
        os.path.join(
            args.export_folderpath,
            f'{args.prefix}_{year}.pickle',
        )
        for year in years
    ]

    if not all([os.path.exists(fp) for fp in expected_output_filepaths]):
        weather_catalogue_df = pd.read_csv(args.weather_catalog)

        weather_catalogue_df[at2d.METHOD_COL] \
        = at2d.LoadTIFMethod.READ_AND_CROP
        weather_catalogue_df.loc[
            weather_catalogue_df[cwdc.FILETYPE_COL] == cwdc.TIF_GZ_EXT,
            at2d.METHOD_COL
        ] = at2d.LoadTIFMethod.COREGISTER_AND_CROP
        weather_catalogue_df.loc[
            weather_catalogue_df[cwdc.ATTRIBUTE_COL] == 'ndvi-interp', 
            at2d.METHOD_COL
        ] = at2d.LoadTIFMethod.READ_NO_CROP

        aggregated_df = at2d.aggregate_tifs_to_df(
            catalogue_df = weather_catalogue_df,
            mask_tif_filepaths = [
                args.cropmask_filepath,
                args.interpmask_filepath,
            ],
            roi_geom_filepath = args.roi_filepath,
            ref_tif_filepath = args.ref_tif_filepath,
            csvs_folderpath = args.csv_folderpath,
        )

        os.makedirs(args.export_folderpath, exist_ok=True)

        for year in years:
            aggregated_df[
                aggregated_df['year'] == year
            ].to_pickle(os.path.join(
                args.export_folderpath,
                f'{args.prefix}_{year}.pickle',
            ))
    else:
        print('All aggregated dfs already created.')

    xy_gdf = at2d.create_xy_gdf(
        cropmask_tif_filepath = args.cropmask_filepath,
        interp_tif_filepath = args.interpmask_filepath,
    )

    xy_gdf.to_file(os.path.join(
        args.export_folderpath,
        f'{args.prefix}_xy.geojson',
    ))
