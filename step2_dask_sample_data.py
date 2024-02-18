

import dask
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

from prettyprinter import pprint
import ee
import geemap

# Initialize the Earth Engine library
ee.Initialize()

from ccdc import get_preprocessed_Sentinel2, add_ccdc_lambda

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
    df = geemap.ee_to_df(time_series)
    return df


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
    ddf = ddf.repartition(npartitions=100)

    def sample_partition(partition, partition_info):
        df = partition.apply(sample_location, axis=1)

        df = pd.concat(df.to_list(), axis=0, keys=df.index.values)
        df =df.reindex(columns=col_names, fill_value=pd.NA)
        df.index.names = index_names

        partition_number = partition_info["number"]
        df.to_csv(f"outputs/dask_outpus/partition_{partition_number}.csv")

    ddf.map_partitions(sample_partition, meta=(None, object)).compute()


