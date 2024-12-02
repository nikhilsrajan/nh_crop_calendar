import os


# JOB_NAME = 'France'
JOB_NAME = 'south-east-africa'

# paths
FOLDERPATH_DATA = '../data'

# FILEPATH_ROI_SHAPE = os.path.join(FOLDERPATH_DATA, 'shapefiles/France_Regions.gpkg')
FILEPATH_ROI_SHAPE = os.path.join(FOLDERPATH_DATA, 'shapefiles/bounds-2012-01-01-nc.geojson')


FOLDERPATH_OUTPUTS = os.path.join(FOLDERPATH_DATA, f'outputs/{JOB_NAME}')

FOLDERPATH_CLUSTERFILES = os.path.join(FOLDERPATH_DATA, 'cluster_files')
FOLDERPATH_CLUSTERFILES_CHIRPS = os.path.join(FOLDERPATH_DATA, 'cluster_files/chirps')
FOLDERPATH_CLUSTERFILES_CPCTMAX = os.path.join(FOLDERPATH_DATA, 'cluster_files/cpc_tmax')
FOLDERPATH_CLUSTERFILES_CPCTMIN = os.path.join(FOLDERPATH_DATA, 'cluster_files/cpc_tmin')
FOLDERPATH_CLUSTERFILES_ESI4WK = os.path.join(FOLDERPATH_DATA, 'cluster_files/esi_4wk')
FOLDERPATH_CLUSTERFILES_NDVI = os.path.join(FOLDERPATH_DATA, 'cluster_files/ndvi')
FOLDERPATH_CLUSTERFILES_NSIDC = os.path.join(FOLDERPATH_DATA, 'cluster_files/nsidc')
FOLDERPATH_CHC_CHIRP_V2P0_P05 = os.path.join(FOLDERPATH_DATA, 'chc/chirps-v2.0/p05')

FOLDERPATH_WORLDCEREAL = os.path.join(FOLDERPATH_DATA, 'worldcereal')
FILEPATH_WORLDCEREAL_AEZ = os.path.join(FOLDERPATH_WORLDCEREAL, 'WorldCereal_AEZ.geojson')
FILEPATH_REFERENCE_GEOTIFF = os.path.join(FOLDERPATH_DATA, 'ref_mod09.ndvi.global_0.05_degree.2019.001.c6.v1.tif')


# create cropmasks
FOLDERPATH_MASKS = os.path.join(FOLDERPATH_OUTPUTS, 'masks')

# aggregate tif to df
FOLDERPATH_CSVS = os.path.join(FOLDERPATH_OUTPUTS, 'csvs')
FOLDERPATH_AGGREGRATE_DFS = os.path.join(FOLDERPATH_OUTPUTS, 'aggregated_dfs')

# clustered plots
FOLDERPATH_PLOTS = os.path.join(FOLDERPATH_OUTPUTS, 'plots')
