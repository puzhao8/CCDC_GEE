import json
import ee

keyfile='private_keys\dhi-gww.json'
key=json.load(open(keyfile))
service_account=key['client_email']
credentials = ee.ServiceAccountCredentials(service_account, keyfile)
ee.Initialize(
    credentials=credentials,
    project='global-wetland-watch',
    opt_url='https://earthengine-highvolume.googleapis.com'
)


#%%
import pandas as pd
cifor = pd.read_csv("outputs\sampled_training_points_wetland_mask_stratified_cifor.csv")

df = pd.read_csv("outputs\sampled_training_points_wetland_mask_stratified.csv")

#%%

for col in ['canopy_height', 'cifor', 'rfw']:
  df[col] = cifor[col]