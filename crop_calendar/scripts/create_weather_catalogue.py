import os
import argparse

import sys
sys.path.append('..')

import create_weather_data_catalogue as cwdc
import presets


CLUSTER_PATHS = {
    presets.ATTR_CHIRPS: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/chirps/global/',
    presets.ATTR_CPCTMAX: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/cpc_tmax/',
    presets.ATTR_CPCTMIN: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/cpc_tmin/',
    presets.ATTR_ESI4WK: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/esi_4wk/',
    presets.ATTR_NSIDC: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/nsidc/daily/',
    presets.ATTR_NSIDC_SURFACE: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/nsidc/daily/surface/',
    presets.ATTR_NSIDC_ROOTZONE: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/nsidc/daily/rootzone/',
    presets.ATTR_FPAR: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/fpar/',
    presets.ATTR_GCVI: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/gcvi/',
    presets.ATTR_NDVI: '/gpfs/data1/cmongp1/GEOGLAM/Input/intermed/ndvi/'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_weather_catalogue.py',
        description = (
            'Script to create weather data catalogue.'
        ),
    )

    parser.add_argument('--chirps', action='store_true',)
    parser.add_argument('--cpctmax', action='store_true',)
    parser.add_argument('--cpctmin', action='store_true',)
    parser.add_argument('--esi4wk', action='store_true',)
    parser.add_argument('--nsidc-surface', action='store_true',)
    parser.add_argument('--nsidc-rootzone', action='store_true',)
    parser.add_argument('--fpar', action='store_true',)
    parser.add_argument('--gcvi', action='store_true',)
    parser.add_argument('--ndvi', action='store_true',)
    # parser.add_argument('--interp-ndvi', help='/path/to/interp_ndvi', required=False, default=None)
    parser.add_argument('--years', help='comma separated years (YYYY). Example: 2019,2020,2018')
    parser.add_argument('--export', help='/path/to/export')

    args = parser.parse_args()

    years = [int(year_str) for year_str in args.years.split(',')]

    attribute_settings_dict = {}

    if args.chirps:
        attribute_settings_dict[presets.ATTR_CHIRPS] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_CHIRPS]
        )
    if args.cpctmax:
        attribute_settings_dict[presets.ATTR_CPCTMAX] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_CPCTMAX]
        )
    if args.cpctmin:
        attribute_settings_dict[presets.ATTR_CPCTMIN] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_CPCTMIN]
        )
    if args.esi4wk:
        attribute_settings_dict[presets.ATTR_ESI4WK] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_ESI4WK]
        )
    if args.nsidc_surface:
        attribute_settings_dict[presets.ATTR_NSIDC_SURFACE] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_NSIDC_SURFACE]
        )
    if args.nsidc_surface:
        attribute_settings_dict[presets.ATTR_NSIDC_ROOTZONE] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_NSIDC_ROOTZONE]
        )
    if args.fpar:
        attribute_settings_dict[presets.ATTR_FPAR] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_FPAR]
        )
    if args.gcvi:
        attribute_settings_dict[presets.ATTR_GCVI] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_GCVI]
        )
    if args.ndvi:
        attribute_settings_dict[presets.ATTR_NDVI] = cwdc.Settings(
            attribute_folderpath = CLUSTER_PATHS[presets.ATTR_NDVI]
        )
    
    weather_catalogue_df = cwdc.create_weather_data_catalogue_df(
        years = years,
        attribute_settings_dict = attribute_settings_dict
    )

    weather_catalogue_df.to_csv(args.export, index=False)
