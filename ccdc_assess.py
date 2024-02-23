
#%% Start Configuration
import json
import xarray as xr
import math

from ccdc import getCcdcFit, ccdcStats, syntheticImage, ccdc_phase, millis_to_fractionOfYear
from prettyprinter import pprint
import matplotlib.pyplot as plt
import ee
ee.Initialize()

FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

#%% CCDC features

check_date = "2018-07-01"
t = millis_to_fractionOfYear(ee.Date(check_date).millis()) # Fraction of Year (FoY)

""" Sentinel-2 """
ccdc_s2 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's2-sr'))\
        .filter(ee.Filter.eq('lambda', 0.005))\
        .mosaic()
        
ccdc_s2_t = getCcdcFit(t, ccdc_s2)

stats_s2 = ccdcStats(t, ccdc_s2_t)
phase_s2 = ccdc_phase(ccdc_s2_t)


""" Sentinel-1 """
ccdc_s1 = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')\
        .filter(ee.Filter.eq('sensor', 's1'))\
        .filter(ee.Filter.eq('lambda', 1))\
        .mosaic()
        
ccdc_s1_t = getCcdcFit(t, ccdc_s1)

stats_s1 = ccdcStats(t, ccdc_s1_t)
phase_s1 = ccdc_phase(ccdc_s1_t)



#%% REM, DEM, Slope, TWI etc.

""" labels """
world_cover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('world_cover')
wetland_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").add(1).rename('wetland_label').unmask()
wetland_mask = wetland_label.gt(0).rename('wetland_mask')

""" stack all feature into a single image """
stack_raw = (stats_s2.addBands([stats_s1, phase_s2, phase_s1, world_cover, wetland_label])
                .regexpRename("_1", "_s1")
                .set('system:time_start', ee.Date(check_date).millis())
        )


print("---------------> Stacked Bands <-----------------")
bandList = stack_raw.bandNames().getInfo()
pprint(bandList)


#%% RMSE grouped by Wetland Types
from utils.ccdc_stats import get_and_plot_stats_group_by

wetland_type_names = {
  0: "non wetlands",
  1: "Transformed wetlands",
  2: "Varzeas and/or Igapós",
  3: "Wetlands in small depressions", # supplied by rain and/or floodable or waterlogged savannas and/or Zurales and/or estuaries
  4: "Flooded forests",
  5: "Overflow Forests",
  6: "Interfluvial flooded forests",
  7: "Floodable grasslands",
  8: "Rivers",
  9: "Wetlands in the process of transformation",
  10: "Swamps"
}

if False:
        selected_bands = [band for band in bandList if '_mean' in band]
        for band in selected_bands:
                df = get_and_plot_stats_group_by(
                                stack_raw=stack_raw, 
                                band=band, 
                                group_by_name='wetland_label', 
                                region=country, 
                                groupby_name_dict=wetland_type_names,
                                scale=100,
                                title_note=check_date
                        )



#%% RSE grouped by World Cover

if True:
        from utils.ccdc_stats import get_and_plot_stats_group_by
        worldcover_type_names = {
                10: "tree cover",
                20: "Shrubland",
                30: "Grassland",
                40: "Cropland", # supplied by rain and/or floodable or waterlogged savannas and/or Zurales and/or estuaries
                50: "Built-up",
                60: "Bare/sparse vegetation",
                70: "Snow and ice",
                80: "Permanent water",
                90: "Herbaceour wetland",
                95: "Mangroves",
                100: "Moss and lichen"
        }

        selected_bands = [band for band in bandList if band.endswith('_rmse')]
        selected_bands = ['ndvi_rmse', 'ndvi_rmse', 'water_rmse']
        for band in selected_bands:
                df = get_and_plot_stats_group_by(
                        stack_raw=stack_raw, 
                        band=band, 
                        group_by_name='world_cover', 
                        region=country, 
                        groupby_name_dict=worldcover_type_names,
                        scale=100,
                        check_date=check_date
                )