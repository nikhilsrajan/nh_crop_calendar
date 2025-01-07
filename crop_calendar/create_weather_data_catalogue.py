import pandas as pd
import datetime
import tqdm

import spatially_interpolate_files as sif
import chcfetch.chcfetch as chcfetch

import presets


# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
tqdm.tqdm.pandas()

ATTRIBUTE_COL = 'attribute'
FILETYPE_COL = 'filetype'
DATE_COL = 'date'
TIF_EXT = '.tif'
TIF_GZ_EXT = '.tif.gz'


def generate_catalogue_df(
    attribute:str,
    attribute_folderpath:str,
    years:list[int] = None,
    tif_filepath_col:str = sif.TIF_FILEPATH_COL,
):
    if attribute not in presets.VALID_PARSER_KEYS:
        raise ValueError(
            f'Invalid attribute={attribute}. Must be from {presets.VALID_PARSER_KEYS}.'
        )

    attribute_catalogue_df = sif.create_catalogue_df(
        folderpath = attribute_folderpath,
        filename_parser = presets.PARSERS[attribute],
        keep_extensions = ['.tif'],
        tif_filepath_col = tif_filepath_col,
    )

    attribute_catalogue_df[ATTRIBUTE_COL] = attribute
    attribute_catalogue_df[FILETYPE_COL] = TIF_EXT

    attribute_catalogue_df = \
    attribute_catalogue_df.sort_values(
        by=[presets.YEAR, presets.DAY]
    ).reset_index(drop=True)

    if years is not None:
        attribute_catalogue_df = \
        attribute_catalogue_df[attribute_catalogue_df[presets.YEAR].isin(years)]

    attribute_catalogue_df[DATE_COL] = attribute_catalogue_df.apply(
        lambda row: (datetime.datetime(year=row[presets.YEAR], month=1, day=1) \
            + datetime.timedelta(days=row[presets.DAY] - 1)).strftime('%Y-%m-%d'),
        axis=1
    )

    return attribute_catalogue_df


def add_year_day_from_date(
    row, 
    date_col:str = DATE_COL,
    year_col:str = presets.YEAR,
    day_col:str = presets.DAY,
):
    date = row[date_col]
    year = date.year
    day = (date - datetime.datetime(year, 1, 1)).days + 1
    row[year_col] = year
    row[day_col] = day
    return row


def fetch_missing_chirps_v2p0_p05_files(
    years:list[int],
    geoglam_chirps_data_folderpath:str,
    chc_chirp_v2_0_p05_download_folderpath:str,
    njobs:int = 8,
    overwrite:bool = False,
    tif_filepath_col:str = sif.TIF_FILEPATH_COL,
):
    print('Creating CHIRPS local catalogue.')
    chirps_downloaded_catalogue_df = generate_catalogue_df(
        attribute = presets.ATTR_CHIRPS,
        attribute_folderpath = geoglam_chirps_data_folderpath,
        tif_filepath_col = tif_filepath_col,
        years = years,
    )

    print('Checking how many files in the local CHIRPS catalogue are corrupted.')
    chirps_downloaded_catalogue_df = \
    chirps_downloaded_catalogue_df.progress_apply(
        lambda row: sif.add_tif_corruption_cols(
            row=row, 
            tif_filepath_col=tif_filepath_col,
        ), 
        axis=1,
    )
    n_corrupted = chirps_downloaded_catalogue_df[sif.IS_CORRUPTED_COL].sum()
    print(f'Number of corrupted tifs: {n_corrupted}')

    print(f"Querying CHC for p05 CHIRPS files for years={years}")
    chc_fetch_paths_dfs = []
    for _year in tqdm.tqdm(years):
        _res_df = chcfetch.query_chirps_v2_global_daily(
            product = chcfetch.Products.CHIRPS.P05,
            startdate = datetime.datetime(_year, 1, 1),
            enddate = datetime.datetime(_year, 12, 31),
            show_progress = False,
        )
        chc_fetch_paths_dfs.append(_res_df)
        del _res_df
    chc_fetch_paths_df = pd.concat(chc_fetch_paths_dfs).reset_index(drop=True)

    chc_fetch_paths_df = chc_fetch_paths_df.apply(add_year_day_from_date, axis=1)
    chc_fetch_paths_df[ATTRIBUTE_COL] = presets.ATTR_CHIRPS

    valid_downloads_df = chirps_downloaded_catalogue_df[~chirps_downloaded_catalogue_df[sif.IS_CORRUPTED_COL]]

    pending_downloads_df = chc_fetch_paths_df[
        ~chc_fetch_paths_df[DATE_COL].isin(chirps_downloaded_catalogue_df[
            ~chirps_downloaded_catalogue_df[sif.IS_CORRUPTED_COL]
        ][DATE_COL])
    ]

    print(f'Number of files that need to be downloaded: {pending_downloads_df.shape[0]}')

    pending_downloads_df = chcfetch.download_files_from_paths_df(
        paths_df = pending_downloads_df,
        download_folderpath = chc_chirp_v2_0_p05_download_folderpath,
        njobs = njobs,
        download_filepath_col = tif_filepath_col,
        overwrite = overwrite,
    )

    pending_downloads_df[FILETYPE_COL] = TIF_GZ_EXT

    keep_cols = [DATE_COL, presets.YEAR, presets.DAY, tif_filepath_col, FILETYPE_COL, ATTRIBUTE_COL]
    merged_catalogue_df = pd.concat([
        pending_downloads_df[keep_cols],
        valid_downloads_df[keep_cols],
    ]).sort_values(by=DATE_COL, ascending=True).reset_index(drop=True)

    return merged_catalogue_df


class Settings(object):
    def __init__(
            self,
            attribute_folderpath:str = None,
            download_folderpath:str = None,
            njobs:int = 8,
            overwrite:bool = False,
        ):
        self.attribute_folderpath = attribute_folderpath
        self.download_folderpath = download_folderpath
        self.njobs = njobs
        self.overwrite = overwrite


def create_weather_data_catalogue_df(
    attribute_settings_dict:dict[str, Settings],
    years:list[int] = None,
    tif_filepath_col:str = sif.TIF_FILEPATH_COL,
):
    weather_data_catalogue_dfs = []
    for attribute, settings in tqdm.tqdm(attribute_settings_dict.items()):
        if settings.attribute_folderpath is None:
            raise ValueError(f'settings.attribute_folderpath for attribute={attribute} can not be None.')

        _catalogue_df = generate_catalogue_df(
            attribute = attribute,
            attribute_folderpath = settings.attribute_folderpath,
            years = years,
            tif_filepath_col = tif_filepath_col,
        )
        
        weather_data_catalogue_dfs.append(_catalogue_df)
        del _catalogue_df
    
    weather_data_catalogue_df = pd.concat(weather_data_catalogue_dfs).reset_index(drop=True)
    del weather_data_catalogue_dfs

    return weather_data_catalogue_df


if __name__ == '__main__':
    pass