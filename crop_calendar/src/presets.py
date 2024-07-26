# parsers

YEAR = 'year'
DAY = 'day'


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


def ndvi_normalise(x):
    return (x - 50) / 200


def ndvi_denormalise(x):
    return x * 200 + 50


PARSERS = {
    'ndvi' : ndvi_filename_parser,
    'nsidc' :  soilmoisture_filename_parser,
    'chirps' : chirps_filename_parser,
    'esi_4wk' : esi4wk_filename_parser,
    'cpc_tmax' : cpc_filename_parser,
    'cpc_tmin' : cpc_filename_parser,
}

VALID_PARSER_KEYS = list(PARSERS.keys())


NORMALISERS = {
    'ndvi': ndvi_normalise
}

VALID_NORMALISER_KEYS = list(NORMALISERS.keys())


DENORMALISERS = {
    'ndvi': ndvi_denormalise
}

VALID_DENORMALISER_KEYS = list(DENORMALISERS.keys())
