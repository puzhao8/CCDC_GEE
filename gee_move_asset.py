
import os
import time
import datetime as dt
from datetime import datetime, timedelta
import subprocess
import ee
ee.Initialize()

# feature = "projects/global-wetland-watch/assets"
# for imgCol in ['training-data/global/GWL_FCS30_2020']:
#     response = subprocess.getstatusoutput(f"earthengine ls {feature}/{imgCol}")
#     asset_list = response[1].replace(f"{feature}/", "").split("\n")
#     if len(asset_list) > 0:
#         for asset_id in asset_list:
#             filename = os.path.split(asset_id)[-1]
#             print(f"{filename}: {asset_id}")
#             os.system(f"earthengine rm {asset_id}")


srcImgCol = "projects/global-wetland-watch/assets/training-data/global/GWL_FCS30_2020"
dstImgCol = "projects/global-wetland-watch/assets/labels/global/GWL_FCS30_2020"

response = subprocess.getstatusoutput(f"earthengine ls {srcImgCol}")
fileList = response[1].replace(f"{srcImgCol}/", "").split("\n")
for filename in fileList:
    # filename = os.path.split(asset_id)[-1]
    
    print(f"{filename}: {filename}")
    # os.system(f"earthengine cp {srcImgCol}/{filename} {dstImgCol}/{filename}")

    os.system(f"earthengine rm {srcImgCol}/{filename}")

