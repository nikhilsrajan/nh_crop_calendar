import os
import argparse

import sys
sys.path.append('..')

import create_weather_data_catalogue as cwdc
import presets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_weather_catalogue.py',
        description = (
            'Script to create weather data catalogue.'
        ),
    )

    parser.add_argument('--chirps', help='/path/to/chirps')
    parser.add_argument('--cpctmax', help='/path/to/cpctmax')
    parser.add_argument('--cpctmin', help='/path/to/cpctmin')
    parser.add_argument('--esi4wk', help='/path/to/esi4wk')
    parser.add_argument('--nsidc', help='/path/to/nsidc')
    parser.add_argument('--interp-ndvi', help='/path/to/interp_ndvi')
    parser.add_argument('--years', help='comma separated years (YYYY). Example: 2019,2020,2018')
    parser.add_argument('--export', help='/path/to/export')

    args = parser.parse_args()

    years = [int(year_str) for year_str in args.years.split(',')]

    weather_catalogue_df = cwdc.create_weather_data_catalogue_df(
        years = years,
        attribute_settings_dict = {
            presets.ATTR_CHIRPS: cwdc.Settings(
                attribute_folderpath = args.chirps,
            ),
            presets.ATTR_CPCTMAX: cwdc.Settings(
                attribute_folderpath = args.cpctmax,
            ),
            presets.ATTR_CPCTMIN: cwdc.Settings(
                attribute_folderpath = args.cpctmin,
            ),
            presets.ATTR_ESI4WK: cwdc.Settings(
                attribute_folderpath = args.esi4wk,
            ),
            presets.ATTR_NSIDC: cwdc.Settings(
                attribute_folderpath = args.nsidc,
            ),
            presets.ATTR_NDVI_INTERP: cwdc.Settings(
                attribute_folderpath = args.interp_ndvi,
            ),
        }
    )

    weather_catalogue_df.to_csv(args.export)
