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


