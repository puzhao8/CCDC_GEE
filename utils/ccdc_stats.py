
import ee
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ee.Initialize()



def get_and_plot_stats_group_by(stack_raw, band='B2_rmse', group_by_name='world_cover', region=None, groupby_name_dict={}, scale=100, title_note=None):
    # stack_raw: ee.Image, including all bands needed
    # band: the band used to conduct the statical analysis
    # group_by_name: the band used to group statistic by
    # region: the region 
    # groupby_name_dict: code and name for grouping

    save_dir = Path(f"outputs/goup_by_{group_by_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    group_by_name_list = list(groupby_name_dict.values())

    is_sar_band = (band.startswith('V')) or (band.endswith('_s1'))
    print(f"{band} is SAR band? {is_sar_band}")
    
    image = stack_raw.select(band)
    if is_sar_band: image = image.clamp(-10, 10)

    combined = stack_raw.select(band).addBands(stack_raw.select(group_by_name))
    
    meanReducer = ee.Reducer.mean()
    stdDevReducer = ee.Reducer.stdDev()

    reducer = meanReducer.combine(
                    reducer2=stdDevReducer,
                    sharedInputs=True).group(
                                    groupField = 1,  
                                    groupName = 'group_id',
                            )

    band_result = combined.reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=scale, #// Set an appropriate scale, depending on your rasters' resolution
            maxPixels=1e20
    )

    result = band_result.getInfo()['groups']

    df = pd.DataFrame(result)
    df['band'] = band
    df[group_by_name] = group_by_name_list


    if is_sar_band:
            ax = df.plot(kind='bar', x=group_by_name, y=['mean', 'stdDev'], title=f"{band}: {title_note}", figsize=(12, 5))
    else:
            ax = df.plot(kind='bar', x=group_by_name, y=['mean', 'stdDev'], ylim=(0, 0.1), title=f"{band}: {title_note}", figsize=(12, 5))
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')  # Adjust here
    plt.tight_layout()
    plt.savefig(save_dir / f'{band}.png', dpi=100)
    plt.close()

    return df