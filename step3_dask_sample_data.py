

import dask
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

from prettyprinter import pprint
import ee
import geemap
from retry import retry
from requests.exceptions import HTTPError 

# Initialize the Earth Engine library
ee.Initialize()

from ccdc import get_preprocessed_Sentinel2, add_ccdc_lambda

@retry(HTTPError, tries=3, delay=2, max_delay=60)
def sample_location(row):
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




if __name__ == "__main__":
    import io
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=4, dashboard_address=':38787')
    client = Client(cluster)#timeout

    index_names = ['pnt_idx', 'seq_idx']
    col_names = [
            # 'pnt_idx', 'seq_idx',
            'B11', 'B11_l005', 'B12', 'B12_l005', 'B2', 'B2_l005', 'B3', 'B3_l005',
            'B4', 'B4_l005', 'B5', 'B5_l005', 'B6', 'B6_l005', 'B7', 'B7_l005',
            'B8', 'B8A', 'B8A_l005', 'B8_l005', 'FoY', 'lat', 'lon', 'ndvi',
            'ndvi_l005', 'ndwi', 'ndwi_l005', 'water', 'water_l005'
        ]


    """ Sample time series over a given point """
    # point = ee.Geometry.Point([-72.19594365263514, 4.556298580150745])

    # # for  debugging
    # df = pd.read_csv("data/WorldCover_Stratified_1k_per_cls.csv")
    # df = df[df['wetland_label']==4].loc[130:150]
    # print(df)
    
    # ddf = dd.from_pandas(df, 2)

    ddf = dd.read_csv("data/WorldCover_Stratified_1k_per_cls.csv")
    ddf = ddf[ddf['system:index'] != 2677]
    ddf = ddf.repartition(npartitions=100)

    def sample_partition(partition, partition_info):
        df = partition.apply(sample_location, axis=1)

        df = pd.concat(df.to_list(), axis=0, keys=df.index.values)
        df =df.reindex(columns=col_names, fill_value=pd.NA)
        df.index.names = index_names

        partition_number = partition_info["number"]
        save_url = f"outputs/dask_outputs/partition_{partition_number}.csv"
        if not Path(save_url).exists():
            df.to_csv(save_url)

    ddf.map_partitions(sample_partition, meta=(None, object)).compute()
    
    # # For Debugging:
    # x = ddf.map_partitions(sample_partition, meta=(None, object))#.compute()
    # x.get_partition(25).compute()

    # 2626 -> 2726
    # 2677, Exception: User memory limit exceeded.

