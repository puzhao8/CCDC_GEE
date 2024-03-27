import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/training\sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")

df_cls = pd.read_csv("data/fused_class_index_names.csv")
df_cls_idx = df_cls.set_index('fused_label')
df_cls
# #%%
# tipo_dict = {
#   0: "non wetlands",
#   1: "Transformed wetlands",
#   2: "Varzeas and/or Igapós",
#   3: "Wetlands in small depressions", # supplied by rain and/or floodable or waterlogged savannas and/or Zurales and/or estuaries
#   4: "Flooded forests",
#   5: "Overflow Forests",
#   6: "Interfluvial flooded forests",
#   7: "Floodable grasslands",
#   8: "Rivers",
#   9: "Wetlands in the process of transformation",
#   10: "Swamps"
# }

# tipo_keys = tipo_dict.keys()
# tipo_names = list(tipo_dict.values())
# df_tipo = pd.DataFrame({'wetland_label': tipo_keys, 'tipo_name': tipo_names})


import numpy as np

label = 'fused_label'

from group_bands import S1_bands, TOPO_bands, SS_bands, feature_dict
col_name = 'VH_rmse'
for col_name in ['VH_rmse', 'VV_rmse', 'HH_rmse', 'HV_rmse']:


  # Calculate the IQR for each group
  Q1 = df.groupby(label)[col_name].quantile(0.25)
  Q2 = df.groupby(label)[col_name].quantile(0.50)
  Q3 = df.groupby(label)[col_name].quantile(0.75)

  IQR = Q3 - Q1

  # Define upper and lower bounds for each group
  beta = 1.5
  lower_bound = Q1 - beta * IQR
  upper_bound = Q3 + beta * IQR

  cls_idxs = sorted(list(Q1.index))
  cls_names = [df_cls_idx.cls_name.loc[cls] for cls in cls_idxs]
  print(f"----------------------- {col_name} --------------------------")
  print(cls_idxs)
  print(cls_names)

  arr = []
  for i in cls_idxs:
      row = []
      for j in cls_idxs:
        L_i, L_j = Q1[i], Q1[j]
        U_i, U_j = Q3[i], Q3[j]
        M_i, M_j = Q2[i], Q2[j]

        bounds = sorted([L_i, L_j, U_i, U_j])
        SI = (bounds[2] - bounds[1]) / (bounds[-1] - bounds[0] + 1e-3)
        if not ((L_i >= U_j) | (L_j >= U_i)): # if overlap exists
            SI = -1 * SI 

        # print(i, j, SI)
        row.append(SI)
      arr.append(row)

  arr = np.around(np.array(arr), decimals=2)
  df_si = pd.DataFrame(arr, index=cls_names, columns=cls_names)

  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap

  # Define the colors, including one for 0 value
  colors = ["darkred", "white", "darkgreen"]  # The first color is for 0

  # Create a new colormap
  cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

  import seaborn as sn
  plt.figure(figsize=(20, 15))
  sn.heatmap(data=df_si, annot=True, annot_kws={"size": 14}, fmt='.1g', vmin=-1, vmax=1, cmap=cmap)
  plt.xticks(fontsize=20, rotation=40, ha='right')
  plt.yticks(fontsize=22)
  plt.title(f"Separation Index (SI) of [{col_name}]\n", fontsize=25)
  plt.tight_layout()
  plt.savefig(f"outputs/separation_index_V1/SI_{col_name}.png")
  plt.close()