
import pandas as pd
import numpy as np
from prettyprinter import pprint
import math
import ee
ee.Initialize()

def maskL8sr(image):
    # // Bit 0 - Fill
    # // Bit 1 - Dilated Cloud
    # // Bit 2 - Cirrus
    # // Bit 3 - Cloud
    # // Bit 4 - Cloud Shadow

    # Bitwise representation of QA_PIXEL flags
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    # Replace the original bands with the scaled ones and apply the masks
    return (image.addBands(opticalBands, None, True)
                .addBands(thermalBands, None, True)
                .updateMask(qaMask)
                .updateMask(saturationMask))

# /** Calculate Phase Based on Flatten CCDC Coefficients **/
def ccdc_phase(ccdc_seg):
    def iterate_function(name, result):
        name = ee.String(name)
        flatten_coefs = flatten_segment(ccdc_seg.select(name))

        # tan(\theta) = COS / SIN, where COS and SIN denote the coeficients of cos(2*\pi*x) and sin(2*\pi*x), respectively
        # tan(x) = a2/a3 => x = arctan(a2/a3) => x= a3.atan2(a2)
        phase = flatten_coefs.select(".*_SIN.*").atan2(flatten_coefs.select(".*_COS.*")).regexpRename('_coefs_SIN', '_phase')
        # phase = flatten_coefs.select(name.cat('_3')).atan2(flatten_coefs.select(name.cat('_2'))).regexpRename('_coefs_2', '_phase')
        # phase2 = flatten_coefs.select(name.cat('_5')).atan2(flatten_coefs.select(name.cat('_4'))).regexpRename('_coefs_4', '_phase2')
        # phase3 = flatten_coefs.select(name.cat('_7')).atan2(flatten_coefs.select(name.cat('_6'))).regexpRename('_coefs_6', '_phase3')
        return ee.Image(result).addBands(phase)

    result = ccdc_seg.select('.*coefs').bandNames().iterate(iterate_function, ee.Image().select())
    return ee.Image(result)


def ccdcStats(t, ccdc_seg):
    # do a 1-year range
    start = ee.Number(t).subtract(0.5)
    end = ee.Number(t).add(0.5)
    seq = ee.List.sequence(start, end, 0.05)
    
    synth_collection = seq.map(lambda n: syntheticImage(ee.Number(n), ccdc_seg))
    synth_collection = ee.ImageCollection(synth_collection)

    # mean = synth_collection.mean()
    mean = synth_collection.reduce(ee.Reducer.mean())
    minmax = synth_collection.reduce(ee.Reducer.minMax())
    amp_names = mean.bandNames().map(lambda i: ee.String(i).replace('_mean', '_amp'))
    amp = minmax.select(".*max").subtract(minmax.select(".*min")).select(minmax.select(".*max").bandNames(), amp_names)

    rmse = ccdc_seg.select('.*_rmse').arrayGet(0)

    # coefficients ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3", "RMSE"]
    # slope = ccdc_seg.select('.*coefs').arrayGet([0, 1]).regexpRename('_coefs', '_slope')
    
    return rmse.addBands([mean, minmax, amp])




def getCcdcFit(time, ccdc):
    c = ee.ImageCollection(ccdc).mosaic()
    segment = dateToSegment(time, c)
    return c.arraySlice(0, segment, segment.add(1))

def dateToSegment(t, ccdc):
    tStart = ee.Image(ccdc.select('tStart'))
    tEnd = ee.Image(ccdc.select('tEnd'))
    segment = tStart.lte(t).And(tEnd.gte(t)).arrayArgmax().arrayGet([0])
    last_seg = tEnd.arrayArgmax().arrayGet([0])
    replacement = segment.where(tEnd.arrayGet([-1]).lte(t), last_seg)
    return replacement

def syntheticImage(t, ccdc):
    t = ee.Number(t)
    coefs = getCcdcFit(t, ccdc).select('.*_coefs')
    names = coefs.bandNames().map(lambda name: [ee.String(name).split("_").get(0)])
    w = t.multiply(ee.Number(math.pi).multiply(2))
    ts = ee.Array([[1, t, w.cos(), w.sin(), w.multiply(2).cos(), w.multiply(2).sin(), w.multiply(3).cos(), w.multiply(3).sin()]])
    image = coefs.multiply(ts).arrayReduce(ee.Reducer.sum(), [1]).arrayProject([0]).arrayGet([0]).rename(names.flatten())
    year = t.floor()
    date = ee.Date.fromYMD(year, 1, 1).advance(t.subtract(year), 'year')
    return image.set('system:time_start', date.millis())

def flatten_segment(segment):
    # This function is used to flatten CCDC segment coefs array into multiple bands in a single ee.Image
    # e.g., B2_coefs [Array: 1x8] -> B2_coefs_[0-7], denoting the 8 coeficients for CCDC model
    # INTP (intercept), Slope, COS, SIN, COS2, SIN2, COS3, SIN3'
    coefs_names = ee.List(["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"])

    numbers = ee.List.sequence(0, 7).map(lambda n: coefs_names.get(n)) #ee.Number(n).int().format()
    result = segment.select('.*_coefs').bandNames().iterate(
        lambda name, result: ee.Image(result).addBands(segment.select([name]).arrayFlatten([[name], numbers])), ee.Image().select()
    )
    return ee.Image(result)

# Convert Millis into the fraction of year: 2020-07-01 -> 2020.5
def ms_to_foY(t):
    date = ee.Date(t)
    dayOfYear = date.getRelative('day', 'year')
    year = date.get('year')
    year_str = year.format()
    isLeapYear = ee.Date.parse('YYYY', year_str).advance(1, 'year').difference(ee.Date.parse('YYYY', year_str), 'day').eq(366)
    daysInYear = ee.Algorithms.If(isLeapYear, 366, 365)
    fractionOfYear = dayOfYear.divide(daysInYear)
    return ee.Number(year).add(fractionOfYear)


def get_preprocessed_Sentinel2(params):
    maskCloud = params['maskCloud']
    start_date = params['start_date']
    end_date = params['end_date']
    region = params['region']
    
    s2_harmonized = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    
    QA_BAND = 'cs_cdf'
    s2_linked = s2_harmonized.filterDate(start_date, end_date).filterBounds(region).linkCollection(csPlus, [QA_BAND])
    # .map(lambda img: ee.Image(img).addBands(csPlus.filterBounds(region).filterDate(start_date, end_date).mean().select(QA_BAND)))
    
    if maskCloud:
        s2_linked = s2_linked.map(lambda img: img.updateMask(img.select(QA_BAND).gte(0.6)))
        
    def add_normalizedDiff(img):
        # ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        # ndwi = img.normalizedDifference(['B3', 'B8']).rename('ndwi')

        ndvi = img.expression('float(b("B8") - b("B4")) / (b("B8") + b("B4"))').rename('ndvi').float().unitScale(-1,1)
        ndwi = img.expression('float(b("B3") - b("B8")) / (b("B3") + b("B8"))').rename('ndwi').float().unitScale(-1,1)
        stack = img.select("B.*").divide(5000).addBands(ndvi).addBands(ndwi)
        water = stack.expression('1 / (1 + exp(- (0.133 + (-5.92 * b("ndvi")) + (14.82 * b("ndwi")))))').rename('water').float()

        return ee.Image(stack.addBands(water).copyProperties(img, img.propertyNames()))
    
    return s2_linked.map(add_normalizedDiff)




def add_ccdc_lambda(sensor, lambda_key, region=None):
    ccdc_local = ee.ImageCollection('projects/unep-sdg661/GWW/ccdc').filter(ee.Filter.eq('sensor', sensor))
    if region is not None: ccdc_local = ccdc_local.filterBounds(region)

    lambda_dict = {
        # Sentinel-2
        'l005': 0.005,
        'l010': 0.010,
        'l025': 0.025,
        'l050': 0.050,

        # Sentinel-1
        'l200': 0.2,
        'l300': 0.3,
        'l500': 0.5,
        'l1000': 1,
    }
    lambda_value = lambda_dict[lambda_key]
    
    def process_img(img):
        n = ms_to_foY(img.date().millis())
        ccdc_lambda = ccdc_local.filter(ee.Filter.eq('lambda', lambda_value))
        syn_img = syntheticImage(n, ccdc_lambda)
        syn_img = add_band_postfix(syn_img, lambda_key)
        return ee.Image(img.toFloat().addBands(syn_img)
                            # .copyProperties(img, img.propertyNames())
                            .set('FoY', n)
                        )
    
    return process_img


def add_band_postfix(img, postfix):
    old_band_names = img.bandNames()
    new_band_names = old_band_names.map(lambda band_name: ee.String(band_name).cat('_').cat(postfix))
    return img.select(old_band_names).rename(new_band_names)


def add_landsat_ccdc(postfix, region=None):
    ccdc_landsat = ee.ImageCollection("GOOGLE/GLOBAL_CCDC/V1")
    if region is not None: ccdc_landsat = ccdc_landsat.filterBounds(region)

    def process_img(img):
        n = ms_to_foY(img.date())  # Assuming ms_to_foY is defined
        
        img = (img.toFloat().select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'])
                           .rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'])
            )
        
        # Assuming syntheticImage and ccdc_landsat are defined
        syn_img = syntheticImage(ee.Number(n), ccdc_landsat)  # ccdc_landsat should be defined elsewhere
        nbr = img.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
        ndvi = img.normalizedDifference(['NIR', 'RED']).rename('NDVI')
        
        nbr_ccdc = syn_img.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
        ndvi_ccdc = syn_img.normalizedDifference(['NIR', 'RED']).rename('NDVI')
        
        img_with_ccdc = img.addBands(syn_img.addBands(nbr).addBands(ndvi))
        img_with_ccdc_and_postfix = add_band_postfix(img_with_ccdc.addBands(nbr_ccdc).addBands(ndvi_ccdc), postfix)
        
        return img_with_ccdc_and_postfix.set('FoY', n)
    
    return process_img


# /*
#   This function is used to obtain raw CCDC coeficients and stats (rmse, min, max, mean).
#   @params.ccdc: input ccdc parameters, use this by default if available
#   @params.sensor: specify sensor parameter
#   @params.lambda: specify lambda parameter
#   @params.t: specify the time t to obtain the corresponding coefs and stats 
#   Return: Stacked features
#   example: 
#     get_coefs_and_stats({ccdc: ccdc (ee.ImageCollection), t: t (ms)}) or 
#     get_coefs_and_stats({sensor: 's1', lambda: 0.3, t: t (ms)})
# */
def get_coefs_stats_phase(params):
    if 'ccdc' in params:
        ccdc = params['ccdc']
        print("input ccdc (get_coefs_and_stats)!")
    else:
        sensor = params['sensor']
        lambda_val = params['lambda']
        print(f"sensor: {sensor}, lambda = {lambda_val}")
        
        ccdc = (ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')
            .filter(ee.Filter.eq('sensor', sensor))
            .filter(ee.Filter.eq('lambda', lambda_val))
            .mosaic())
          
  
    t = params['t']        
    ccdc_t = getCcdcFit(t, ccdc)
  
    stats = ccdcStats(t, ccdc_t)
    coefs = flatten_segment(ccdc_t).regexpRename("_coefs_", "_")
    phase = ccdc_phase(ccdc_t) # phase info can be derived from coefs, we don't have to export them.

    return ee.Image(coefs).addBands([stats, phase])

# /*
#   This function is used to fetch seasonal synthetic images in April, July, and Oct.
#   @params.ccdc: input ccdc parameters, use this by default if available
#   @params.sensor: specify sensor parameter
#   @params.lambda: specify lambda parameter
#   @params.year: specify year for fetching seasonal synthetic images
#   Return: Stacked seasonal synthetic images
# */
def get_seasonal_synImgs(params):
    if 'ccdc' in params:
        ccdc = params['ccdc']
        print("input ccdc (get_seasonal_synImgs)!")
    else:
        sensor = params['sensor']
        lambda_val = params['lambda']
        print(f"sensor: {sensor}, lambda = {lambda_val}")
        
        ccdc = (
            ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')
            .filter(ee.Filter.eq('sensor', sensor))
            .filter(ee.Filter.eq('lambda', lambda_val))
            .mosaic()
        )
    
    year = params['year']
  
    synIm_0401 = syntheticImage(ms_to_foY(f"{year}-04-01"), ccdc)  # syntheticImage function should be defined
    synIm_0701 = syntheticImage(ms_to_foY(f"{year}-07-01"), ccdc)
    synIm_1001 = syntheticImage(ms_to_foY(f"{year}-10-01"), ccdc)
  
    synIm_0401 = add_band_postfix(synIm_0401, 'apr')
    synIm_0701 = add_band_postfix(synIm_0701, 'jul')
    synIm_1001 = add_band_postfix(synIm_1001, 'oct')
  
    return ee.Image(synIm_0401).addBands([synIm_0701, synIm_1001])

""" Stack features from various sources """
def stack_features(check_date="2020-07-01"):

    year = check_date[:4]
    t = ms_to_foY(ee.Date(check_date).millis()) # Fraction of Year (FoY)

    """ Sentinel-2 """
    print("---------------> Sentinel-2 <-----------------")
    ccdc_s2 = (
        ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')
            .filter(ee.Filter.eq('sensor', 's2-sr'))
            .filter(ee.Filter.eq('lambda', 0.005))
            
        ).mosaic()
            
    stack_s2 = get_coefs_stats_phase({'ccdc': ccdc_s2, 't': t})
    synImg_s2_seasonal = get_seasonal_synImgs({'ccdc': ccdc_s2, 'year': year})

    # pprint(stack_s2.bandNames().getInfo())


    """ Sentinel-1 """
    print("---------------> Sentinel-1 <-----------------")
    ccdc_s1 = (
        ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')
            .filter(ee.Filter.eq('sensor', 's1'))
            .filter(ee.Filter.eq('lambda', 1))
            # .filter(ee.Filter.eq('lambda', 0.3))
        ).mosaic()
            
    stack_s1 = get_coefs_stats_phase({'ccdc': ccdc_s1, 't': t})
    synImg_s1_seasonal = get_seasonal_synImgs({'ccdc': ccdc_s1, 'year': year})

    # pprint(stack_s1.bandNames().getInfo())

    """ PALSAR-2 Scansar """      
    ccdc_ss = (
        ee.ImageCollection('projects/unep-sdg661/GWW/ccdc')
        .filter(ee.Filter([
                    ee.Filter.eq('sensor', 'scansar'), 
                    ee.Filter.eq('lambda', 0.5), 
                    # ee.Filter.eq('lambda', 0.3), 
                    # ee.Filter.stringContains('system:index', 'v3')
                ]))
        ).mosaic()
    stack_ss = get_coefs_stats_phase({'ccdc': ccdc_ss, 't': t})
    synImg_ss_seasonal = get_seasonal_synImgs({'ccdc': ccdc_ss, 'year': year})


    # // ===========================> Topography-based Features <===========================
    """ # %% REM, DEM, Slope, TWI etc. """
    # Relative Elevation Model
    rem = ee.ImageCollection("projects/global-wetland-watch/assets/features/REM")\
            .mosaic().rename('rem')

    # Digital Elevation Model
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")\
            .select('DEM').mosaic().rename('elevation')\
            .setDefaultProjection(crs="EPSG:4326", scale=30)

    # Calculate Terrain Based on DEM: slope, aspect, hillshade
    # Terrain: https://code.earthengine.google.com/2e0a145c5bb298add69e97ed24854db3
    terrain = ee.Algorithms.Terrain(dem)
            
    # Topographic Wetness Index (TWI)
    flowAccumulation = ee.Image('WWF/HydroSHEDS/15ACC').select('b1')
    slope = ee.Terrain.slope(dem).multiply(math.pi / 180)

    # TWI in GEE: https://code.earthengine.google.com/17a95828c7e9e6a4e06115eb737ee235
    twi = flowAccumulation.divide(slope.tan()).log().unmask(500000).rename('twi').setDefaultProjection(crs="EPSG:4326", scale=10).toInt16()
    # Map.addLayer(twi, {'min':0, 'max':20, 'palette':['blue', 'white', 'green']}, 'twi')

    # // HAND Product 30m
    hand30_1000 = ee.Image("users/gena/GlobalHAND/30m/hand-1000").rename('hand30_100')
    hand30_100 = ee.ImageCollection("users/gena/global-hand/hand-100").mosaic().rename('hand30_1000')

    # // ===========================> Climate related Features <===========================
    canopy_height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').unmask().rename('canopy_height')
    canopy_height_std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').rename('canopy_height_std')

    # Ecoregion
    ecoregion = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").reduceToImage(
        properties=['ECO_ID'], reducer=ee.Reducer.first()).rename('ECO_ID')

    biome = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").reduceToImage(
        properties=['BIOME_NUM'], reducer=ee.Reducer.first()).rename('BIOME_NUM')

    # OpenLandMap: Soil Water Content
    soil_water = ee.Image('OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01').regexpRename('b', 'soil_water_')

    # Yearly Average Precipitation (ERA5 or CHIRPS?)
    year_filter = ee.Filter.date(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-01-01').advance(1, 'year'))

    chirps_yearly_precip = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .select('precipitation')
            .filter(year_filter)
            .sum()
            .rename('chirps_precip')
        )

    era5_yearly_precip = (
        ee.ImageCollection("ECMWF/ERA5/MONTHLY")
            .filter(year_filter)
            .select('total_precipitation')
            .sum()
            .multiply(1e3)
            .rename('era5_precip')
    )
    

    """ labels """
    world_cover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('world_cover')
    wetland_label = ee.Image("projects/global-wetland-watch/assets/labels/COL/top10_label").add(1).rename('wetland_label').unmask()
    wetland_mask = wetland_label.gt(0).rename('wetland_mask')

    GWL_FCS30 = ee.ImageCollection("projects/global-wetland-watch/assets/training-data/global/GWL_FCS30_2020").mosaic().rename('GWL_FCS30')

    # RFW: Reguarly Flooded Wetlands (500m) vs. CIFOR 2016 Global wetlands Map (231m)
    rfw = ee.Image("projects/global-wetland-watch/assets/labels/global/Regularly_Flooded_Wetlands").rename('rfw')
    cifor = ee.Image("projects/global-wetland-watch/assets/labels/global/CIFOR_2016_TROP_SUBTROP_Wetland_V3b").unmask().divide(10).int8().rename('cifor')


    """ rescale bands """
    synImg_s2_seasonal = synImg_s2_seasonal.multiply(1e4)
    synImg_s1_seasonal = synImg_s1_seasonal.unitScale(-30, 5).multiply(1e4)
    synImg_ss_seasonal = synImg_ss_seasonal.unitScale(-30, 5).multiply(1e4)

    """ stack all feature into a single image """
    # rename water and water_coefs band names for Sentinel-1
    stack = (stack_s2.addBands([stack_s1, stack_ss])
                    .addBands([synImg_s2_seasonal, synImg_s1_seasonal, synImg_ss_seasonal]) #// synthetic images in April, July, and October
                    .regexpRename("_1_1", "_ss") #// rename some Scansar bands
                    .regexpRename("_1", "_s1") #// rename some Sentinel-1 bands
                    .addBands([
                        rem, twi, terrain, hand30_1000, hand30_100, #// topography-based features
                        canopy_height, canopy_height_std,  #// canopy height features
                        ecoregion, biome, chirps_yearly_precip, era5_yearly_precip, soil_water, #// ecoregion, biome, chirps, era5
                        world_cover, GWL_FCS30, wetland_label, wetland_mask, rfw, cifor #// labels
                    ]).set('system:time_start', ee.Date(check_date).millis())
    )

    return stack




# TODO: ipygee not working, need to fix here
# /** boxplot time series **/
def chart_boxplot_time_series_cmp(imgCol, region, band1, band2):
    imgCol = imgCol.sort('FoY')
    # region = params['region']
    # band1 = params['band1']
    # band2 = params['band2']

    # Function to apply to each image
    def reduce_region(img):
        stats = img.reduceRegion(reducer=ee.Reducer.percentile([25, 50, 75]), geometry=region, scale=10, maxPixels=1e20)
        fraction_of_year = ms_to_foY(img.date())
        return {
            'Date': fraction_of_year,
            # 'Date': img.date().format(),

            band1 + '_p25': stats.get(band1 + '_p25'),
            band1 + '_p50': stats.get(band1 + '_p50'),
            band1 + '_p75': stats.get(band1 + '_p75'),

            band2 + '_p25': stats.get(band2 + '_p25'),
            band2 + '_p50': stats.get(band2 + '_p50'),
            band2 + '_p75': stats.get(band2 + '_p75'),
        }

    # Map over the image collection
    data = imgCol.map(lambda img: ee.Feature(None, reduce_region(img))).aggregate_array('properties').getInfo()

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Now df can be used with seaborn or matplotlib for plotting
    return df


if __name__ == "__main__":

    check_date = '2020-07-01'
    stack = stack_features(check_date)
    print("stack: ", stack.bandNames().getInfo())