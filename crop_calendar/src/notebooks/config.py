import os


JOB_NAME = 'France'

# paths
FILEPATH_ROI_SHAPE = '../../data/shapefiles/France_Regions.gpkg'

FOLDERPATH_DATA = '../../data'
FOLDERPATH_OUTPUTS = f'../../data/outputs/{JOB_NAME}'

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
