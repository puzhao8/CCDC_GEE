
#%% Start Configuration
import ee
import math
ee.Initialize()
from ccdc import getCcdcFit, syntheticImage, millis_to_fractionOfYear
from prettyprinter import pprint


FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

point = ee.Geometry.Point([-72.19594365263514, 4.556298580150745])

#%% CCDC features

check_date = "2020-09-01"
t = millis_to_fractionOfYear(ee.Date(check_date).millis()) # Fraction of Year (FoY)

""" Sentinel-2 """
ccdc_s2 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's2-sr'))\
        .filter(ee.Filter.eq('lambda', 0.005))\
        .mosaic()
        
ccdc_s2_t = getCcdcFit(t, ccdc_s2)

# *_coefs, *_magnitude, *_rmse, 
# 'tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb',
coefs_s2_t = ccdc_s2_t.select('.*_coefs')

print("---------------> Sentinel-2 <-----------------")
pprint(ccdc_s2_t.bandNames().getInfo())


""" Sentinel-1 """
ccdc_s1 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's1'))\
        .filter(ee.Filter.eq('lambda', 1))\
        # .filterBounds(point)
        
ccdc_s1_t = getCcdcFit(t, ccdc_s1)

# *_coefs, *_magnitude, *_rmse, 
# 'tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb',
coefs_s1_t = ccdc_s1_t.select('.*_coefs')

print("---------------> Sentinel-1 <-----------------")
pprint(ccdc_s1_t.bandNames().getInfo())

""" Synthetic Image Predicted by CCDC """
syn_s2_t = syntheticImage(t, ccdc_s2)
syn_s1_t = syntheticImage(t, ccdc_s1)

#%% REM, DEM, Slope, TWI etc.

# Relative Elevation Model
rem = ee.ImageCollection("projects/global-wetland-watch/assets/features/REM_MERIT_SWORD")\
        .filterBounds(country).mosaic().rename('rem')

# Digital Elevation Model
dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")\
        .filterBounds(country)\
        .select('DEM').mosaic().rename('elevation')\
        .setDefaultProjection(crs="EPSG:4326", scale=30)

# Calculate Terrain Based on DEM: slope, aspect, hillshade
# Terrain: https://code.earthengine.google.com/2e0a145c5bb298add69e97ed24854db3
terrain = ee.Algorithms.Terrain(dem)
        
# Topographic Wetness Index (TWI)
flowAccumulation = ee.Image('WWF/HydroSHEDS/15ACC').select('b1')
slope = ee.Terrain.slope(dem).multiply(math.pi / 180)

# TWI in GEE: https://code.earthengine.google.com/17a95828c7e9e6a4e06115eb737ee235
twi = flowAccumulation.divide(slope.tan()).log().rename('twi')

""" labels """
world_cover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('world_cover')
wetland_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").add(1).rename('wetland_label').unmask()
wetland_mask = wetland_label.gt(0).rename('wetland_mask')

""" stack all feature into a single image """
# rename water and water_coefs band names for Sentinel-1
stack_s2 = coefs_s2_t.addBands(syn_s2_t)
stack_s1 = coefs_s1_t.addBands(syn_s1_t).select(['water', 'water_coefs']).rename(['water_s1', 'water_coefs_s1'])
stack = (stack_s2.addBands([stack_s1, twi, rem, terrain, world_cover, wetland_label])
            .set('system:time_start', ee.Date(check_date).millis())
    )

print("---------------> Stacked Bands <-----------------")
pprint(list(stack.bandNames().getInfo()))

#%% Export
import xarray as xr

local_proj = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(point)
                .first().select('B8').projection()
        )
crs = local_proj.crs().getInfo()
stack = stack.setDefaultProjection(crs=local_proj, scale=10)
print(f"local crs: {crs}")

ds = xr.open_dataset(ee.ImageCollection([stack]), 
                         engine='ee', 
                        #  projection=local_proj,
                         crs='EPSG:32618',
                         geometry=point.buffer(5e4).bounds(),
                         scale=100,
                    )

# ds['world_cover'].T.plot()
#%% Visualization

# import geemap
# Map = geemap.Map()
# Map

# #%%

# Map.addLayer(syn_s1_t, {'bands': ['VH'], 'min':-15, 'max': 0}, 's1')
# Map.addLayer(syn_s2_t, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.2}, 's2')