import datetime


# parsers

YEAR = 'year'
DAY = 'day'

# attributes must not have '_' character
# This restriction is placed because of the file name convention
# f'{attribute}_{year}_{day}.tif' used in spatially_interpolate_files.py
ATTR_NDVI = 'ndvi'
ATTR_NSIDC = 'nsidc'
ATTR_CHIRPS = 'chirps'
ATTR_ESI4WK = 'esi-4wk'
ATTR_CPCTMAX = 'cpc-tmax'
ATTR_CPCTMIN = 'cpc-tmin'
ATTR_NDVI_INTERP = 'ndvi-interp'
ATTR_GCVI = 'gcvi'
ATTR_FPAR = 'fpar'


def interp_filename_parser(filename:str):
    _, year, day = filename.split('.')[0].split('_')
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def ndvi_filename_parser(filename:str):
    _, _, _, _, year, day, _, _, _ = filename.split('.')
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def soilmoisture_filename_parser(filename:str):
    _, _, _, _, year, day, _, _ = filename.split('_')
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def chirps_filename_parser(filename:str):
    year_day_str = filename.split('_')[1].split('.')[-1]
    year = int(year_day_str[:4])
    day = int(year_day_str[4:])
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def esi4wk_filename_parser(filename:str):
    year_day_str = filename.split('_')[-1].split('.')[0]
    year = int(year_day_str[:4])
    day = int(year_day_str[4:])
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def cpc_filename_parser(filename:str):
    year_day_str = filename.split('_')[1]
    year = int(year_day_str[:4])
    day = int(year_day_str[4:])
    return {
        YEAR: int(year),
        DAY: int(day),
    }


def fpar_filename_parser(filename:str):
    date_str = filename.split('_')[-1].split('.')[0]
    date = datetime.datetime.strptime(date_str, '%Y%m%d')
    year = date.year
    day = (date - datetime.datetime(year, 1, 1)).days + 1
    return {
        YEAR: year,
        DAY: day,
    }


def ndvi_normalise(x):
    return (x - 50) / 200


def ndvi_denormalise(x):
    return x * 200 + 50


PARSERS = {
    ATTR_NDVI : ndvi_filename_parser,
    ATTR_NSIDC : soilmoisture_filename_parser,
    ATTR_CHIRPS : chirps_filename_parser,
    ATTR_ESI4WK : esi4wk_filename_parser,
    ATTR_CPCTMAX : cpc_filename_parser,
    ATTR_CPCTMIN : cpc_filename_parser,
    ATTR_NDVI_INTERP : interp_filename_parser,
    ATTR_GCVI : ndvi_filename_parser,
    ATTR_FPAR : fpar_filename_parser,
}

VALID_PARSER_KEYS = list(PARSERS.keys())


NORMALISERS = {
    ATTR_NDVI: ndvi_normalise,
    ATTR_NDVI_INTERP: ndvi_normalise,
}

VALID_NORMALISER_KEYS = list(NORMALISERS.keys())


DENORMALISERS = {
    ATTR_NDVI: ndvi_denormalise,
    ATTR_NDVI_INTERP: ndvi_denormalise,
}

VALID_DENORMALISER_KEYS = list(DENORMALISERS.keys())
