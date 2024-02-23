
import ee
import geemap
import pandas as pd
from ccdc import get_preprocessed_Sentinel2, add_ccdc_lambda
from pathlib import Path
from retry import retry
from requests.exceptions import HTTPError 

ee.Initialize()


# FAO = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
# country = FAO.filter(ee.Filter.eq("ADM0_NAME", "Colombia"))

""" Sample time series over a given point """
# point = ee.Geometry.Point([-72.19594365263514, 4.556298580150745])

def fc_to_gdf(fc):
    try:
        df = geemap.ee_to_gdf(fc)
        return df
    except Exception as e:
        print("----> ", e)
        if str(e) == "User memory limit exceeded.": return 
        else: return geemap.ee_to_gdf(fc)

@retry(HTTPError, tries=10, delay=2, max_delay=60)
def sample_location(row):
    # print(row['system:index'], row.longitude, row.latitude)
    print(row.Index, row.longitude, row.latitude)
    point = ee.Geometry.Point([row.longitude, row.latitude])


    fc = stack.toInt16().reduceRegions(
            collection=ee.FeatureCollection([point]), 
            reducer=ee.Reducer.mean(), 
            scale=10).map(
                        lambda x: x.set({
                            'idx': row.Index,
                            'lon': row.longitude,
                            'lat': row.latitude
                    }))
    
    return fc_to_gdf(fc)




if __name__ == "__main__":
    pntfc_worldCover = pd.read_csv("data/WorldCover_Stratified_1k_per_cls.csv")
    pntfc_wetMask = pd.read_csv("data/wetland_mask_Stratified_5k_per_cls.csv")


    from ccdc import get_stacked_ccdc_features
    stack = get_stacked_ccdc_features(check_date="2020-07-01")
    # bandList = stack.bandNames().getInfo()
    from band_names import bandList # includes .geo. lat, lon


    csv_restart = True
    save_url = Path("outputs/sampled_training_points_wetland_mask_stratified.csv")
    if csv_restart or (not save_url.exists()):
        # create an empty csv
        df0 = pd.concat([pd.DataFrame([], columns=bandList)], axis=0)
        df0.set_index('idx').to_csv(save_url)


    pntfc = pntfc_wetMask
    for row in pntfc.itertuples():
        df = sample_location(row)
        df.set_index('idx').to_csv(save_url, mode='a', header=False, index=True)




