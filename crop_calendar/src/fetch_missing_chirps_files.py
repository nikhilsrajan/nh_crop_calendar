import pandas as pd
import datetime
import tqdm

import spatially_interpolate_files as sif
import chcfetch.chcfetch as chcfetch

import presets


# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
tqdm.tqdm.pandas()


FILETYPE_COL = 'filetype'
DATE_COL = 'date'
TIF_EXT = '.tif'
TIF_GZ_EXT = '.tif.gz'


def fetch_missing_chirps_v2p0_p05_files(
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    geoglam_chirps_data_folderpath:str,
    chc_chirp_v2_0_p05_download_folderpath:str,
    njobs:int = 8,
    overwrite:bool = False,
):
    print('Creating CHIRPS local catalogue.')
    chirps_downloaded_catalogue_df = sif.create_catalogue_df(
        folderpath = geoglam_chirps_data_folderpath,
        filename_parser = presets.chirps_filename_parser,
        keep_extensions = ['.tif'],
        tif_filepath_col = sif.TIF_FILEPATH_COL,
    )

    chirps_downloaded_catalogue_df = \
    chirps_downloaded_catalogue_df.sort_values(
        by=[presets.YEAR, presets.DAY]
    ).reset_index(drop=True)

    chirps_downloaded_catalogue_df[DATE_COL] = chirps_downloaded_catalogue_df.apply(
        lambda row: datetime.datetime(year=row[presets.YEAR], month=1, day=1) \
            + datetime.timedelta(days=row[presets.DAY] - 1),
        axis=1
    )

    chirps_downloaded_catalogue_df = chirps_downloaded_catalogue_df[
        (chirps_downloaded_catalogue_df[DATE_COL] >= startdate)
        & (chirps_downloaded_catalogue_df[DATE_COL] <= enddate)
    ]

    print('Checking how many files in the local CHIRPS catalogue are corrupted.')
    chirps_downloaded_catalogue_df = \
    chirps_downloaded_catalogue_df.progress_apply(sif.add_tif_corruption_cols, axis=1)
    n_corrupted = chirps_downloaded_catalogue_df[sif.IS_CORRUPTED_COL].sum()
    chirps_downloaded_catalogue_df[FILETYPE_COL] = TIF_EXT
    print(f'Number of corrupted tifs: {n_corrupted}')

    print(f"Querying CHC for p05 CHIRPS files for daterange {startdate.strftime('%Y-%m-%d')} to {enddate.strftime('%Y-%m-%d')}")
    chc_fetch_paths_df = chcfetch.query_chirps_v2_global_daily(
        product = chcfetch.Products.CHIRPS.P05,
        startdate = startdate,
        enddate = enddate,
    )

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
        download_filepath_col = sif.TIF_FILEPATH_COL,
        overwrite = overwrite,
    )
    pending_downloads_df[FILETYPE_COL] = TIF_GZ_EXT

    merged_catalogue_df = pd.concat([
        pending_downloads_df[[DATE_COL, sif.TIF_FILEPATH_COL, FILETYPE_COL]],
        valid_downloads_df[[DATE_COL, sif.TIF_FILEPATH_COL, FILETYPE_COL]],
    ]).sort_values(by=DATE_COL, ascending=True).reset_index(drop=True)

    return merged_catalogue_df


if __name__ == '__main__':
    pass