

#%%

import pandas as pd
def remap_fused_label(df, label_lt_10_flag=False):
    """ convert class labels, making them range from 1 to 20 """
    df['fused_label_raw'] = df.fused_label

    # preprocessing wetland labels (<10)
    if label_lt_10_flag:
        df.loc[df.fused_label_raw < 10, ['fused_label', 'wetland_label']] = df.loc[df.fused_label_raw < 10, ['fused_label', 'wetland_label']] - 1 # (1, 9) -> (0, 8)
        df.loc[(df.fused_label_raw == 10) & (df.wetland_label == 10), ['fused_label', 'wetland_label']] = 9 # for Swamps, 10 -> 9

    # preprocessing world cover [10, 20, 30, ..., 90, 95, 100] -> [10, 12, 13, ..., 19, 11, 20]
    world_cover_index = (df.fused_label_raw > 10) & (df.fused_label_raw % 10 == 0)
    df.loc[world_cover_index, 'fused_label'] = df.loc[world_cover_index, 'fused_label_raw'] // 10 + 10
    df.loc[df.fused_label_raw == 95, 'fused_label'] = 11
    df.loc[(df.fused_label_raw==10) & (df.WorldCover==10), 'fused_label'] = 10

    # print(sorted(df.fused_label.unique()))
    return df



#%%

class_dict = {
    # wetland types
    0: "Transformed wetlands",
    1: "Varzeas and/or Igapós",
    2: "Wetlands in small depressions", # supplied by rain and/or floodable or waterlogged savannas and/or Zurales and/or estuaries
    3: "Flooded forests",
    4: "Overflow Forests",
    5: "Interfluvial flooded forests",
    6: "Floodable grasslands",
    7: "Rivers",
    8: "Wetlands in the process of transformation",
    9: "Swamps",

    # world cover 10m
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

import numpy as np
data_cls = np.array([list(class_dict.keys()), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 11, 20], list(class_dict.values())]).T
df_cls = pd.DataFrame(data_cls, columns=['fused_label_raw', 'fused_label', 'cls_name'])
df_cls.loc[:, 'fused_label'] = df_cls.fused_label_raw.transform(lambda x: int(x))


# df_pnt = pd.read_csv("data/fused_label_stratified_1k_per_cls_V1.csv")
# df_pnt = df_pnt.rename(columns={'system:index': 'idx'})

#%% 
import dask.dataframe as dd
from pathlib import Path 

""" merge all partitions into a single csv file """
folder_name = "fused_label_stratified_1k_per_cls_V0"
ddf = dd.read_csv(Path(f"outputs\dask_outputs/{folder_name}") / f"training_points_partition_*.csv", on_bad_lines='skip', assume_missing=True) 
df0 = ddf.compute()
df0['fused_label'] = 9

df0
#%%

df_mrg = df.merge(df_pnt[['idx', 'fused_label', 'WorldCover']], on='idx', how='left')
print('fused_label' in df.columns)

# df = remap_fused_label(df)

print('fused_label' in df_mrg.columns)


df_mrg.plot(kind='hist', y='fused_label', bins=100)
df_mrg 


#%%
# df.set_index('idx').to_csv(f"data/training/{folder_name}.csv")


#%%

df_ = pd.read_csv("data/training/fused_label_stratified_1k_per_cls.csv")
df_ = df_.drop(columns=['fused_label_raw', 'world_cover'])

df_mrg = pd.concat([df_, df0], axis=0)