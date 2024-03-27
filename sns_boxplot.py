
#%%


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from group_bands import feature_dict, TOPO_bands

df = pd.read_csv("data/training\sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")

df_cls = pd.read_csv("data/fused_class_index_names.csv")
df_cls_idx = df_cls.set_index('fused_label')
df_cls


""" pairplot """
label = 'fused_label'

feature_key = 'SS'
cols = feature_dict[feature_key]
# cols = ['VH_apr', 'VH_jul', 'VH_oct']
# cols = ['rem', 'hand30_100', 'elevation', 'twi', 'slope']
# cols = ['rem']

# cols = ['rem', 'hand30_100']
cols_w_label = cols + ['idx', label, 'cls_name']
df = df[cols_w_label].dropna()

beta = 1.5
for col in cols:
  plt.figure()

  # Calculate the IQR for each group
  Q1 = df.groupby(label)[col].quantile(0.25)
  Q2 = df.groupby(label)[col].quantile(0.50)
  Q3 = df.groupby(label)[col].quantile(0.75)

  IQR = Q3 - Q1
  lower_bound = Q1 - beta * IQR
  upper_bound = Q3 + beta * IQR

  print(f"lower bound")
  print(lower_bound)

  print(f"\nupper_bound")
  print(upper_bound)
  print()

  df_flt = pd.DataFrame()
  for group in df[label].unique(): # loop over each class
      group_df = df[df[label] == group]
      # print(f"[{lower_bound[group]}, {upper_bound[group]}]")
      group_df = group_df[(group_df[col] >= lower_bound[group]) & (group_df[col] <= upper_bound[group])]
      df_flt = pd.concat([df_flt, group_df], ignore_index=True)


  sns_plot = sns.boxplot(data=df_flt[[col, 'cls_name']], x=col, y="cls_name", color=".8", linecolor="#137", linewidth=.75, fliersize=0)
  sns_plot.set_title(f"{feature_key}: {col} (N: {df_flt.shape[0]})", size=12, y=1.03)
  
  if col in TOPO_bands:
    if col.startswith('hand'): plt.xlim(-5, 50)
    elif 'twi' == col: plt.xlim(-5, 20)
    elif 'elevation' == col: plt.xlim(-5, 600)
    elif 'slope' == col: plt.xlim(-5, 20)   
    elif 'hillshade' == col: pass 
    else: plt.xlim(-20, 100)

  plt.tight_layout()
  plt.savefig(f"outputs/sns_boxplot/{feature_key}_{col}_N_{df_flt.shape[0]}", dpi=300)
