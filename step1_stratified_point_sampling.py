"""  Stratified Sampling based on WorldCover / Wetland Mask """
# https://code.earthengine.google.com/78760ae7e9c432d436e061fdda5da7d8
# https://code.earthengine.google.com/03c19c63be6e933681c46573cdfe7fee # landMask applied

import ee
import geemap
ee.Initialize()


landMask = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification').neq(200)
WorldCover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').mask(landMask).rename('WorldCover')
wetland_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").unmask(-1).rename('wetland_label')
wetland_mask = wetland_label.gte(0).rename('wetland_mask')

fused_label = wetland_label.where(wetland_label.eq(-1), WorldCover).unmask().rename('fused_label')
fused_label = fused_label.mask(fused_label.gte(0)).int8()

# Add the top 10 labels and pixel longitude-latitude bands to the WorldCover image
image = wetland_label.int8().addBands(fused_label).addBands(wetland_mask).addBands(WorldCover).addBands(ee.Image.pixelLonLat())


# Define the region of interest. Replace `country` with your actual region.
# For example, let's define a simple geometry around a specific area (this should be replaced with your region of interest)
FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))


classBand = 'fused_label'
# Perform stratified sampling on the image
pntCol = image.stratifiedSample(
    numPoints=1000,
    classBand=classBand,
    region=country,
    scale=10,
    projection='EPSG:4326',
    seed=42,
    # classValues=[0],
    # classPoints=[10000],
    dropNulls=True,
    tileScale=4,
    geometries=True
)

# df = geemap.ee_to_gdf(pntCol) # TODO: very slow

df = geemap.ee_to_gdf(pntCol)
df.to_csv(f'data/{classBand}_Stratified_1k_per_cls.csv')
