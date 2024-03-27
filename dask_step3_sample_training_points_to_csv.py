
# fused_label: https://code.earthengine.google.com/3f6bf7ace36115f827d78c21270ee483

#%%
import ee
import geemap
import dask
import dask.dataframe as dd

import pandas as pd
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

    fc = stack.reduceRegions(
                    collection=ee.FeatureCollection([point]), 
                    reducer=ee.Reducer.mean(), 
                    scale=10
                ).map(lambda x: x.set({
                                'idx': row.Index,
                                'lon': row.longitude,
                                'lat': row.latitude,
                            })
                        )
    
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

    filename = "fused_label_stratified_1k_per_cls_seed5"

    pntfc = dd.read_csv(f"data/{filename}.csv")
    pntfc = pntfc.rename(columns={'system:index': 'idx'})
    pntfc = pntfc[(pntfc.idx >=0) & (pntfc.idx < 100)]

    save_dir = Path("outputs/dask_outputs") / f"{filename}"
    save_dir.mkdir(exist_ok=True, parents=True)

    ddf = pntfc.repartition(npartitions=10)

    def sample_partition(partition, partition_info):
        partition_number = partition_info["number"]
        save_url = save_dir / f"training_points_partition_{partition_number}.csv"

        reinit_csv = True
        for row_idx, row in enumerate(partition.itertuples()):
            df = sample_location(row)
            if reinit_csv:  
                df.set_index('idx').to_csv(save_url)
                reinit_csv = False
            else: 
                df.set_index('idx').to_csv(save_url, mode='a', header=False, index=True)
            
    ddf.map_partitions(sample_partition, meta=(None, object)).compute()

    
#%%

""" merge all partitions into a single csv file """
from pathlib import Path
import dask.dataframe as dd
import pandas as pd

filename = "fused_label_stratified_1k_per_cls_seed5"
save_dir = Path("outputs/dask_outputs") / filename

ddf = dd.read_csv(save_dir / f"training_points_partition_*.csv", on_bad_lines='skip', assume_missing=True) 
df = ddf.compute()

df_cls = pd.read_csv("data/fused_class_index_names.csv")

#%%
if 'fused_label_raw' not in df.columns:
    df = df.rename(columns={'fused_label': 'fused_label_raw'})

#%%
df = df.merge(df_cls, on='fused_label_raw', how='left')

df.set_index('idx').to_csv(f"data/training/sampled_points_{filename}.csv")




        







