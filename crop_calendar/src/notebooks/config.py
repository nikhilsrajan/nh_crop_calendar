import os


JOB_NAME = 'France'

# paths
FILEPATH_ROI_SHAPE = '../../data/shapefiles/France_Regions.gpkg'

FOLDERPATH_DATA = '../../data'
FOLDERPATH_OUTPUTS = f'../../data/outputs/{JOB_NAME}'

FOLDERPATH_CLUSTERFILES = '../../data/cluster_files'
FOLDERPATH_CLUSTERFILES_CHIRPS = '../../data/cluster_files/chirps'
FOLDERPATH_CLUSTERFILES_CPCTMAX = '../../data/cluster_files/cpc_tmax'
FOLDERPATH_CLUSTERFILES_CPCTMIN = '../../data/cluster_files/cpc_tmin'
FOLDERPATH_CLUSTERFILES_ESI4WK = '../../data/cluster_files/esi_4wk'
FOLDERPATH_CLUSTERFILES_NDVI = '../../data/cluster_files/ndvi'
FOLDERPATH_CLUSTERFILES_NSIDC = '../../data/cluster_files/nsidc'
FOLDERPATH_CHC_CHIRP_V2P0_P05 = '../../data/chc/chirps-v2.0/p05'

FOLDERPATH_WORLDCEREAL = '../../data/worldcereal'
FILEPATH_WORLDCEREAL_AEZ = '../../data/worldcereal/WorldCereal_AEZ.geojson'
FILEPATH_REFERENCE_GEOTIFF = '../../data/ref_mod09.ndvi.global_0.05_degree.2019.001.c6.v1.tif'


# settings
## create cropmasks
WC_S2_GRID_RES = 4
WC_PRODUCTS_TO_MERGE = ['springcereals', 'wintercereals']
WC_MERGED_PRODUCT_NAME = 'cereals'
WC_OUTPRODUCTS = ['cereals', 'temporarycrops']
FOLDERPATH_MASKS = os.path.join(FOLDERPATH_OUTPUTS, 'masks')
BINARY_MASK_SETTINGS = {
    'cereals': {
        'threshold': 10,
        'how': 'gte',
    },
    'temporarycrops': {
        'threshold': 0,
        'how': 'gt',
    }
}


## aggregate tif to df
FOLDERPATH_CSVS = os.path.join(FOLDERPATH_OUTPUTS, 'csvs')
FOLDERPATH_AGGREGRATE_DFS = os.path.join(FOLDERPATH_OUTPUTS, 'aggregated_dfs')
