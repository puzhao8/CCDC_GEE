import pandas as pd
import matplotlib.pyplot as plt 

from group_bands import which_sensor

def get_band_limits(band, q=0.01):
  lims = df[band].quantile([q, 1-q])
  low = lims.loc[q]
  high = lims.loc[1-q]
  return low, high


version = 'V2'
df = pd.read_csv(f"data/training/sampled_points_wetland_label_Stratified_1k_per_cls_mrg_{version}.csv")

# band = 'VH_INTP'
for band in list(df.columns):
  # if (band not in ['idx', 'lat', 'lon', 'geometry']):# & (band.startswith('V') | band.startswith('H') | band.endswith('_s1') | band.endswith('_ss')):
  if 'rmse' in band:
    low, high = get_band_limits(band)
    print(band, low, high)

    fig, ax = plt.subplots()
    if low < high:
      df_flt = df[(df[band] >= low) & (df[band] <= high)]
      df_flt.plot(ax=ax, kind='hist', y=band, bins=100, logy=True, ylim=(1, 1e4)) #ylim=(0, 1e4)
    else:
      df.plot(ax=ax, kind='hist', y=band, bins=100, logy=True, ylim=(1, 1e4)) #ylim=(0, 1e4)
    
    sensor = which_sensor(band)
    if 'V0' == version: 
      if 'S1' == sensor: title = f"{sensor} (lambda: 1.0)" 
      elif 'SS' == sensor: title = f"{sensor} (lambda: 0.5)" 
      else: title = f"{sensor}"

    if version in ['V1', 'V2']: 
      if 'S1' == sensor: title = f"{sensor} (lambda: 0.3)" 
      elif 'SS' == sensor: title = f"{sensor} (lambda: 0.3)" 
      elif "rmse" in band: title = "RMSE"
      else: title = f"{sensor}"

    ax.set_title(f"{sensor}: {band}")
    fig.tight_layout()
    fig.savefig(f"outputs/feature_histogram_V1/{sensor}_{band}.png")
    plt.close()