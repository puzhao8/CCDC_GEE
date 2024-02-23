
import ee
import geemap
import pandas as pd
from ccdc import get_preprocessed_Sentinel2, add_ccdc_lambda
from pathlib import Path
from retry import retry
from requests.exceptions import HTTPError 

ee.Initialize()

meta = {
 'B11': 'float',
 'B11_l005': 'float',
 'B12': 'float',
 'B12_l005': 'float',
 'B2': 'float',
 'B2_l005': 'float',
 'B3': 'float',
 'B3_l005': 'float',
 'B4': 'float',
 'B4_l005': 'float',
 'B5': 'float',
 'B5_l005': 'float',
 'B6': 'float',
 'B6_l005': 'float',
 'B7': 'float',
 'B7_l005': 'float',
 'B8': 'float',
 'B8A': 'float',
 'B8A_l005': 'float',
 'B8_l005': 'float',
 'B9': 'float',
 'FoY': 'float',
 'lon': 'float',
 'lat': 'float',
 'ndvi': 'float',
 'ndvi_l005': 'float',
 'ndwi': 'float',
 'ndwi_l005': 'float',
 'water': 'float',
 'water_l005': 'float'}

# FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
# country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

""" Sample time series over a given point """
# point = ee.Geometry.Point([-72.19594365263514, 4.556298580150745])

@retry(HTTPError, tries=10, delay=2, max_delay=60)
def sample_location(row):
    print(row['system:index'], row.longitude, row.latitude)
    point = ee.Geometry.Point([row.longitude, row.latitude])

    def sample_time_series(image):
        return image.reduceRegions(collection=ee.FeatureCollection([point]), reducer=ee.Reducer.mean(), scale=10)\
                    .map(lambda x: x.set({
                                            'FoY': image.get('FoY'),
                                            'lon': row.longitude,
                                            'lat': row.latitude
                                        })
                        )
    
    s2ImgCol = get_preprocessed_Sentinel2({
        'region': point,
        'start_date': '2017-01-01',
        'end_date': '2025-01-01',
        'maskCloud': True
    })

    add_s2_l005 = add_ccdc_lambda('s2-sr', 'l005')
    s2ImgCol_ccdc = s2ImgCol.map(add_s2_l005)

    time_series = ee.FeatureCollection(s2ImgCol_ccdc.map(sample_time_series).flatten())
    
    try:
        df = geemap.ee_to_df(time_series)
        return df
    except Exception as e:
        print("----> ", e)
        if str(e) == "User memory limit exceeded.": return 
        else: return geemap.ee_to_df(time_series)


# df = sample_location(row)

# # create meta data for dask 
# meta = {}
# for col in df.columns:
#     meta[col] = 'float'




if __name__ == "__main__":
    pntfc_worldCover = pd.read_csv("data/WorldCover_Stratified_1k_per_cls.csv")
    pntfc_wetMask = pd.read_csv("data/wetland_mask_Stratified_5k_per_cls.csv")
    # pntfc = pntfc_worldCover
    
    # pntfc = pntfc_worldCover[pntfc_worldCover['wetland_label']==4]
    # rows = pntfc_worldCover.loc[137]

   
    index_names = ['pnt_idx', 'seq_idx']
    col_names = [
            # 'pnt_idx', 'seq_idx',
            'B11', 'B11_l005', 'B12', 'B12_l005', 'B2', 'B2_l005', 'B3', 'B3_l005',
            'B4', 'B4_l005', 'B5', 'B5_l005', 'B6', 'B6_l005', 'B7', 'B7_l005',
            'B8', 'B8A', 'B8A_l005', 'B8_l005', 'FoY', 'lat', 'lon', 'ndvi',
            'ndvi_l005', 'ndwi', 'ndwi_l005', 'water', 'water_l005'
        ]


    csv_restart = False
    save_url = Path("outputs/sampled_ts_data_WorldCover_stratified_V1_2626_2726.csv")
    if csv_restart or (not save_url.exists()):
        # create an empty csv
        df0 = pd.concat([pd.DataFrame([], columns=index_names + col_names)], axis=0)
        df0 = df0.set_index(index_names)
        df0.to_csv(save_url)


    pntfc = pntfc_worldCover
    # pntfc = pntfc.loc[:2]
    print('pntfc \n', pntfc)


    step = 10 # step > 1
    # for i in range(0, pntfc.shape[0], step):
    for i in range(4310, pntfc.shape[0], step):
        print()
        print(f"-------------------------------- [i = {i}] -------------------------------")

        rows = pntfc.loc[i:i+step-1]
        print(rows)

        df = rows.apply(sample_location, axis=1) 
        df = pd.concat(df.to_list(), axis=0, keys=df.index.values)
        df.index.names = index_names
        
        df_tmp =df.reindex(columns=col_names, fill_value=pd.NA)
        df_tmp.to_csv(save_url, mode='a', header=False, index=True)

        # if 0 == i: df.to_csv(save_url, header=True, index=True)
        # else: df.to_csv(save_url, mode='a', header=False, index=True)

