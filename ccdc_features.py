
#%% Start Configuration
import json
import xarray as xr
import math
import pandas as pd

from ccdc import getCcdcFit, ccdcStats, syntheticImage, ccdc_phase, millis_to_fractionOfYear
from prettyprinter import pprint
import matplotlib.pyplot as plt
import ee
ee.Initialize()

FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

"""#%% CCDC features"""

pntfc = pd.read_csv("data/wetland_mask_Stratified_5k_per_cls.csv")

check_date = "2020-07-01"
from ccdc import get_stacked_ccdc_features
stack = get_stacked_ccdc_features(check_date=check_date)

# print("---------------> Stacked Bands <-----------------")
# bandList = stack.bandNames().getInfo()
# pprint(bandList)

#%%
# sample point one by one
row = pntfc.loc[0]


point = ee.Geometry.Point([row.longitude, row.latitude])
fc = stack.toInt16().reduceRegions(
                collection=ee.FeatureCollection([point]), 
                reducer=ee.Reducer.mean(), scale=10).map(
                                                lambda x: x.set({
                                                        # 'pnt_idx': row['system:index'],
                                                        'lon': row.longitude,
                                                        'lat': row.latitude
                                                }))

import geemap                    
df = geemap.ee_to_gdf(fc)
df

#%% Stratified Sampling based on Wetland Mask
import geemap

pntCol = stack.stratifiedSample(
    numPoints=1000,
    classBand='wetland_mask',
    region=country,
    scale=100,
    projection='EPSG:4326',
    seed=42,
    dropNulls=True,
    tileScale=4,
    geometries=True
)

df = geemap.ee_to_df(pntCol) # TODO: very slow
df.to_csv('data/training_points_stratified_by_wetland_mask_1k_per_cls.csv')


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
    try: ds.to_netcdf(f'outputs/Colombia_features_scale_{scale}m.h5', engine='h5netcdf', encoding=encoding)
    except Exception as e: 
        print("----> ", e)
        if str(e) == "User memory limit exceeded.": return
    else: return ds.to_netcdf(f'outputs/Colombia_features_scale_{scale}m.h5', engine='h5netcdf', encoding=encoding)
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

bands_selected = [band for band in bandList if band.endswith('_phase3')]
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
    plt.savefig(f'outputs/features/{band}.png', dpi=100)
    plt.close()

    time.sleep(3)

