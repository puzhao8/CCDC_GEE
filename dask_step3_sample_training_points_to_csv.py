
#%%
import ee
import geemap
import dask
import dask.dataframe as dd

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



#%%
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    import io
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=4, dashboard_address=':38787')
    client = Client(cluster)#timeout


    #%%
    from ccdc import stack_features
    stack = stack_features(check_date="2020-07-01")

    # pntfc_world_cover = pd.read_csv("data/WorldCover_Stratified_1k_per_cls.csv")
    # pntfc_wetland_mask = pd.read_csv("data/wetland_mask_Stratified_5k_per_cls.csv")

    filename = "wetland_label_Stratified_1k_per_cls_mrg"
    pntfc = dd.read_csv(f"data/{filename}.csv")

    save_dir = Path("outputs/dask_outputs") / f"{filename}_V0"
    save_dir.mkdir(exist_ok=True, parents=True)

    ddf = pntfc.repartition(npartitions=100)

    def sample_partition(partition, partition_info):
        partition_number = partition_info["number"]
        save_url = save_dir / f"training_points_partition_{partition_number}.csv"

        for row_idx, row in enumerate(partition.itertuples()):
            df = sample_location(row)
            if (not Path(save_url).exists()) & (0 == row_idx):  df.set_index('idx').to_csv(save_url)
            else: df.set_index('idx').to_csv(save_url, mode='a', header=False, index=True)
            
    ddf.map_partitions(sample_partition, meta=(None, object)).compute()

    #%%




        







