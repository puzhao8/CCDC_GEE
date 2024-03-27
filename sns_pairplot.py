
#%%


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from group_bands import feature_dict

df = pd.read_csv("data/training\sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")

df_cls = pd.read_csv("data/fused_class_index_names.csv")
df_cls_idx = df_cls.set_index('fused_label')
df_cls


""" pairplot """
label = 'cls_name'
MODE = 'intersection' # union, intersection

feature_key = 'S1'
cols = feature_dict[feature_key]
cols = ['VH_apr', 'VH_jul', 'VH_oct']
# cols = ['rem', 'hand30_100', 'elevation', 'twi', 'slope']

# cols = ['rem', 'hand30_100']
cols_w_label = cols+['idx', 'cls_name']
df = df[cols_w_label].dropna()

Q1 = df[cols].quantile(0.25)
Q2 = df[cols].quantile(0.50)
Q3 = df[cols].quantile(0.75)


beta = 1.5
IQR = Q3 - Q1
lower_bound = Q1 - beta * IQR
upper_bound = Q3 + beta * IQR

print(f"lower bound")
print(lower_bound)

print(f"\nupper_bound")
print(upper_bound)

print()


""" union """
if 'union' == MODE:
  filtered_df = pd.DataFrame()
  for col in cols:
      # print(f"[{lower_bound[group]}, {upper_bound[group]}]")
      df_keep = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]
      print(col, df_keep.shape, 'idx' in df_keep.columns)

      filtered_df = pd.concat([filtered_df, df_keep], ignore_index=True)

  filtered_df = filtered_df.drop_duplicates(subset='idx', keep='first')
  sns_plot = sns.pairplot(data=filtered_df[cols + ['cls_name']], hue='cls_name')
  sns_plot.fig.suptitle(f"{feature_key}_{len(cols)}_bands_{MODE} (N: {df_keep.shape[0]})", size=12, y=1.03)
  sns_plot.savefig(f"outputs/pairplot/{feature_key}_{len(cols)}_bands_{MODE}_N_{filtered_df.shape[0]}", dpi=300)

  print(filtered_df.shape)


""" intersection """
if 'intersection' == MODE:
  df_keep = df
  for col in cols:
      # print(f"[{lower_bound[group]}, {upper_bound[group]}]")
      df_keep = df_keep[(df_keep[col] >= lower_bound[col]) & (df_keep[col] <= upper_bound[col])]
      print(col, df_keep.shape, 'idx' in df_keep.columns)

  # filtered_df = df_keep.drop_duplicates(subset='idx', keep='first')
  sns_plot = sns.pairplot(data=df_keep[cols + ['cls_name']], hue='cls_name', plot_kws={'s': 10})
  sns_plot.fig.suptitle(f"{feature_key}_{len(cols)}_bands_{MODE} (N: {df_keep.shape[0]})", size=12, y=1.03)
  sns_plot.savefig(f"outputs/pairplot/{feature_key}_{len(cols)}_bands_{MODE}_N_{df_keep.shape[0]}.png", dpi=300)


  print(df_keep.shape)
