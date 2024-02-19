
#%% Start Configuration
import ee
import xarray as xr
import math
ee.Initialize()
from ccdc import getCcdcFit, ccdcStats, syntheticImage, ccdc_phase, millis_to_fractionOfYear
from prettyprinter import pprint
import matplotlib.pyplot as plt


FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

point = ee.Geometry.Point([-72.19594365263514, 4.556298580150745])

#%% CCDC features

check_date = "2020-07-01"
t = millis_to_fractionOfYear(ee.Date(check_date).millis()) # Fraction of Year (FoY)

""" Sentinel-2 """
ccdc_s2 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's2-sr'))\
        .filter(ee.Filter.eq('lambda', 0.005))\
        .mosaic()
        
ccdc_s2_t = getCcdcFit(t, ccdc_s2)

stats_s2 = ccdcStats(t, ccdc_s2_t)
phase_s2 = ccdc_phase(ccdc_s2_t)

# *_coefs, *_magnitude, *_rmse, 
# 'tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb',
# coefs_s2_t = ccdc_s2_t.select('.*_coefs')

print("---------------> Sentinel-2 <-----------------")
pprint(stats_s2.bandNames().getInfo())


""" Sentinel-1 """
ccdc_s1 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's1'))\
        .filter(ee.Filter.eq('lambda', 1))\
        # .filterBounds(point)
        
ccdc_s1_t = getCcdcFit(t, ccdc_s1)

stats_s1 = ccdcStats(t, ccdc_s1_t)
phase_s1 = ccdc_phase(ccdc_s1_t)

# *_coefs, *_magnitude, *_rmse, 
# 'tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb',
# coefs_s1_t = ccdc_s1_t.select('.*_coefs')

print("---------------> Sentinel-1 <-----------------")
pprint(stats_s1.bandNames().getInfo())

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
twi = flowAccumulation.divide(slope.tan()).log().unmask(500000).rename('twi')
# Map.addLayer(twi, {'min':0, 'max':20, 'palette':['blue', 'white', 'green']}, 'twi')

""" labels """
world_cover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('world_cover')
wetland_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").add(1).rename('wetland_label').unmask()
wetland_mask = wetland_label.gt(0).rename('wetland_mask')

""" stack all feature into a single image """
# rename water and water_coefs band names for Sentinel-1
# stack_s2 = stats_s2.addBands(syn_s2_t)
# stack_s1 = stats_s1.addBands(syn_s1_t)
s1_scaled_100 = stats_s1.select("V.*_mean|V.*_max|V.*_min").multiply(100)
stack = (stats_s2.addBands([stats_s1, phase_s2, phase_s1]).multiply(1e4)
                .addBands(s1_scaled_100, s1_scaled_100.bandNames(), True)
                .addBands([twi, rem, world_cover, wetland_label, terrain])
                .regexpRename("_1", "_s1")
                .set('system:time_start', ee.Date(check_date).millis())
    )

print("---------------> Stacked Bands <-----------------")
bandList = stack.bandNames().getInfo()
pprint(bandList)

#%% Export
import xarray as xr

mgrs = ee.FeatureCollection("users/omegazhangpzh/MGRS")
mgrs_tile = mgrs.filter(ee.Filter.eq("MGRS_UTM", "18N")).first()

# 1x1 or 5x5 degree grids

scale = 10
point = ee.Geometry.Point([-70.13463383818201,3.3623537053117514])
# point = ee.Geometry.Point([-70.3635, 3.469])


local_proj = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(point)
                .first().select('B8').projection()
        )
crs = local_proj.crs().getInfo()
stack = stack.setDefaultProjection(crs=local_proj, scale=scale).toInt16()
print(f"local crs: {crs}")


COMP_LEVEL = 7
PATCH_SIZE = 128

band_encoding = { "zlib": True,
                "complevel": COMP_LEVEL,
                "fletcher32": True,
                "chunksizes": (1,PATCH_SIZE,PATCH_SIZE)
            }


encoding = {band: band_encoding for band in bandList}


# bands2export = ['VH_coefs', 'VH'] # 'VH_coefs', 'VH', 'B12_coefs', 'B12'
# # bands2export = ['rem', 'twi', 'world_cover', 'wetland_label']
# encoding = {band: band_encoding for band in bands2export}
# stack_ = stack.select(bands2export)

# TODO: 1) twi empty? 2) *_coefs?
ds = xr.open_dataset(ee.ImageCollection([stack]), 
                         engine='ee', 
                        #  projection=local_proj,
                         crs=crs,
                         geometry=point.buffer(5000).bounds(), # 256 x 256
                        # geometry=mgrs_tile,
                         scale=scale, # 26,897 x 932,901 per tile at 10m
                    )

# # ds['world_cover'].T.plot()
ds

#%%
from retry import retry
from requests.exceptions import HTTPError 
@retry(HTTPError, tries=3, delay=2, max_delay=60)
def export(ds, scale):
        try: 
                ds.to_netcdf(f'outputs/Colombia_features_scale_{scale}m.h5', engine='h5netcdf', encoding=encoding)


        except Exception as e:
                print("----> ", e)
                if str(e) == "User memory limit exceeded.": return 
        else: 
                return ds.to_netcdf(f'outputs/Colombia_features_scale_{scale}m.h5', engine='h5netcdf', encoding=encoding)

        ds.close()

export(ds, scale)
#%% Visualization

# import geemap
# Map = geemap.Map()
# Map

# #%%

# Map.addLayer(syn_s1_t, {'bands': ['VH'], 'min':-15, 'max': 0}, 's1')
# Map.addLayer(syn_s2_t, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.2}, 's2')

#%% save figures

import matplotlib.pyplot as plt
import time 
from matplotlib.colors import LinearSegmentedColormap as linSegCol
colors = ['#c21515','#ffffff','#19a11d']
my_colormap = linSegCol.from_list("my_colormap", colors)

bands_selected = [band for band in bandList if band.endswith('_rmse') or band.endswith('_mean')]
for band in bands_selected:
    vmin, vmax = ds[band].quantile([0.02, 0.98]).values
    print(f"band: {band}, range: ({vmin}, {vmax})")

    if 'phase' in band:
        ds[band].T.plot(vmin=-3, vmax=3, cmap=my_colormap)

    elif '_slope' in band:
        ds[band].T.plot(vmin=-0.01, vmax=0.01, cmap=my_colormap)

    elif 'rmse' in band:
        ds[band].T.plot(vmin=0, vmax=vmax, cmap=linSegCol.from_list("my_colormap", ['#ffffff','#c21515']))

    else: 
        ds[band].T.plot(vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.savefig(f'figures/{band}.png', dpi=100)
    plt.close()

    time.sleep(3)
