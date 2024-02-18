
import pandas as pd
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
    return image.addBands(opticalBands, None, True)\
                .addBands(thermalBands, None, True)\
                .updateMask(qaMask)\
                .updateMask(saturationMask)



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
    numbers = ee.List.sequence(0, 7).map(lambda n: ee.Number(n).format())
    result = segment.select('.*_coefs').bandNames().iterate(
        lambda name, result: ee.Image(result).addBands(segment.select([name]).arrayFlatten([[name], numbers])),
        ee.Image().select()
    )
    return ee.Image(result)

def millis_to_fractionOfYear(t):
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
        n = millis_to_fractionOfYear(img.date().millis())
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
        n = millis_to_fractionOfYear(img.date())  # Assuming millis_to_fractionOfYear is defined
        
        img = img.toFloat().select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'])\
                           .rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'])
        
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
        fraction_of_year = millis_to_fractionOfYear(img.date())
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
