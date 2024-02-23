"""  Stratified Sampling based on WorldCover / Wetland Mask """
# https://code.earthengine.google.com/3bfb267aa6b4409bba53f41d772543df

import ee
import geemap
ee.Initialize()

# Load the first image from the ESA WorldCover dataset
WorldCover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('WorldCover')
top10_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").add(1).rename('wetland_label').unmask()
wetland_mask = top10_label.gt(0).rename('wetland_mask')

# Add the top 10 labels and pixel longitude-latitude bands to the WorldCover image
image = top10_label.addBands(wetland_mask).addBands(WorldCover).addBands(ee.Image.pixelLonLat())


# Define the region of interest. Replace `country` with your actual region.
# For example, let's define a simple geometry around a specific area (this should be replaced with your region of interest)
FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

# Perform stratified sampling on the image
pntCol = image.stratifiedSample(
    numPoints=10000,
    classBand='wetland_mask',
    region=country,
    scale=10,
    projection='EPSG:4326',
    seed=42,
    dropNulls=True,
    tileScale=4,
    geometries=True
)

df = geemap.ee_to_df(pntCol) # TODO: very slow
df.to_csv('data/gdf_wetMask_Stratified_1k_per_cls.csv')
