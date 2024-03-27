
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/training\sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")

df_cls = pd.read_csv("data/fused_class_index_names.csv")
df_cls_idx = df_cls.set_index('fused_label')
df_cls


#%%
label = 'fused_label'

from group_bands import S1_bands, TOPO_bands, SS_bands, feature_dict
# col_name = 'VH_rmse'
for col in TOPO_bands:

  # Calculate the IQR for each group
  Q1 = df.groupby(label)[col].quantile(0.25)
  Q2 = df.groupby(label)[col].quantile(0.50)
  Q3 = df.groupby(label)[col].quantile(0.75)

  IQR = Q3 - Q1

  # Define upper and lower bounds for each group
  beta = 1.5
  lower_bound = Q1 - beta * IQR
  upper_bound = Q3 + beta * IQR

  cls_idxs = sorted(list(Q1.index))
  cls_names = [df_cls_idx.cls_name.loc[cls] for cls in cls_idxs]
  print(f"----------------------- {col} --------------------------")
  print(cls_idxs)
  print(cls_names)

  #%% 
  """ plot """
  # Filtering outliers
  filtered_df = pd.DataFrame()

  for group in df[label].unique():
      group_df = df[df[label] == group]
      # print(f"[{lower_bound[group]}, {upper_bound[group]}]")
      group_df = group_df[(group_df[col] >= lower_bound[group]) & (group_df[col] <= upper_bound[group])]
      filtered_df = pd.concat([filtered_df, group_df], ignore_index=True)

  # # add tipo name
  # df_mrg = filtered_df.drop(columns=['cls_name']).merge(df_cls, on=label, how='left')
  # print("filtered_df shape: ", df_mrg.shape)
  # print("filtered_df unique label: ", df_mrg.fused_label.unique())
  # print(df_mrg.cls_name.unique())

  #%%
  fig, ax = plt.subplots()
  filtered_df.boxplot(ax=ax, by='cls_name', column=col, ylabel=col)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')  # Adjust here

  if col in TOPO_bands:
    if col.startswith('hand'): plt.ylim(-5, 50)
    elif 'twi' == col: plt.ylim(-5, 20)
    elif 'elevation' == col: plt.ylim(-5, 600)
    elif 'slope' == col: plt.ylim(-5, 20)    
    else: plt.ylim(-20, 100)

  else:
     plt.ylim(-1, 5)
      
  plt.tight_layout()
  fig.savefig(f"outputs/boxplot_by_class_V2/{col}_by_class.png", dpi=200)
  plt.close()

