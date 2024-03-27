

#%%
import pandas as pd

df = pd.read_csv("C:\DHI\CCDC_GEE\outputs_V1/area_by_COD_DEF_en.csv")
df_abv1 = df[df.area_pa >= 1]


for col in ['COD_DEF', 'TIPO', 'TIPO_Code', 'Cid_Cob2', 'Cod_Hum', 'HUMEDALES', 'AGRUP']:

  uniList = df_abv1[col].unique()
  
  print("----------------------------------------------")
  print(col, len(uniList))
  print(uniList)

df_abv1

#%%

df = df_abv1
print("Cid_Cob2 vs. TIPO")
print("---------------------------------------------------------")
for tipo in df['TIPO'].unique():
   print(df[df.TIPO==tipo]['Cid_Cob2'].unique(), tipo)

print()
print("COD_DEF vs. TIPO")
print("---------------------------------------------------------")
for tipo in df['TIPO'].unique():
   print(df[df.TIPO==tipo]['COD_DEF'].unique(), tipo)

#%%

# gee: https://code.earthengine.google.com/e16089fb9fc7c27000866431f45b08be
import ee
ee.Initialize()

colombia_fc = ee.FeatureCollection("projects/global-wetland-watch/assets/labels/COL/colombia")
tipo_name_list = ee.FeatureCollection(colombia_fc).aggregate_array('TIPO').distinct()
Cid_Cob2_list = ee.FeatureCollection(colombia_fc).aggregate_array('Cid_Cob2').distinct()

def get_type_area(tipo_name):
    return colombia_fc.filter(ee.Filter.eq("TIPO", tipo_name)).reduceColumns(ee.Reducer.sum(), ["area_ha"]).get("sum")

area_list = tipo_name_list.map(get_type_area)


#%%

import pandas as pd
import geopandas as gpd
import dask_geopandas as ddg
import dask.dataframe as dd

ddf =ddg.read_file("C:/Users/puzh/Downloads/Colombia/001.shp", npartitions=10)
ddf

#%%


tipo_name_list = [
  "Sin información",
  "Humedales transformados",
  "Bosques inundables",
  "Humedales en proceso de transformación",
  "Ríos",
  "Arbustales inundables",
  "Madreviejas",
  "Lagunas",
  "Herbazales inundables",
  "Varzeas y/o Igapós",
  "Bosques de rebalse",
  "Palmares de Canangucha",
  "Aningales y/o Bajos y/o Esteros",
  "Salados",
  "Playas",
  "Palmares de Moriche y/o Bosques Mixtos",
  "Bosques inundables de interfluvios",
  "Zonas pantanosas",
  "Bajos",
  "Bajos y/o Esteros",
  "Pozos de médanos",
  "Turberas",
  "Bosques de galería",
  "Bijaguales",
  "Sabanas arboladas (Saladillales, Congriales)",
  "Bijaguales y/o Esteros",
  "Cultivos de arroz (arrozales)",
  "Humedales en depresiones pequeñas abastecidas por la lluvia y/o Sabanas inundables o encharcables y/o Zurales y/o  Esteros",
  "Plantas de tratamiento en explotaciones mineras y petroleras o Lagos en canteras y excavaciones abandonadas",
  "Bosques riparios",
  "Aningales y/o Bajos",
  "Humedales pequeños en valles secos de los Andes",
  "Pantanos",
  "Reservorios de agua (represas, embalses) y/o Humedales alrededor de hidroeléctricas",
  "Humedales en suelos podsolicos (Barillales) y/o Humedales ó bajos en depresiones pequeñas abastecidas por la lluvia",
  "Palmares en tepuyes",
  "Aningales",
  "Lagos",
  "Ciénagas",
  "Manglares",
  "Mares y oceanos",
  "Arroyos y quebradas",
  "Helechales y/o Eneales",
  "Helechales y/o Cangrejas",
  "Lagunas costeras estuarinas y canales",
  "Estanques para acuicultura marina e interior",
  "Sedimentos expuestos en bajamar (Litorales rocosos ó riscales y/o Planos lodosos intermareales)",
  "Pantanos riparios a lo largo de ríos de bajo orden fluvial  y/o Lagunas y pantanos en pequeñas depresiones abastecidas por la lluvia",
  "Guandales, Natales, Sajales, Cuangariales, Cativales",
  "Lagunas en pequeñas depresiones abastecidas por la lluvia",
  "Lagunas someras glaciares",
  "Turberas o Juncales",
  "Plantas de tratamiento de aguas urbanas",
  "Canales de riego",
  "Tembladeros",
  "Pantanos residuales",
  "Salitrales",
  "Lagunas costeras de agua dulce",
  "Cativales y/o Corchales",
  "Panganales",
  "Lagunas costeras",
  "Lagunas estuarinas",
  "Humedales construidos para tratamiento de aguas servidas: lagunas de oxidación y/o sedimentación",
  "Dunas",
  "Explotaciones de sal",
  "Sedimentos expuestos en bajamar (Litorales rocosos y calcáreos y/o Planos lodosos intramareales)"
]

""" Calculate Percentage Per COD_DEF """
meta = {
 'OBJECTID_1': 'float',
 'HUMEDALES': 'float',
 'Cod_Hum': 'float',
 'Ambiente': 'str',
 'SubambGeom': 'float',
 'LEYENDA': 'float',
 'COD_DEF': 'float',
 'Cid_Cob2': 'float',
#  'TIPO': 'str',
 'OBSERV': 'str',
 'Area_m2': 'float',
 'ambiente2': 'float',
 'observacio': 'str',
 'AGRUP': 'str',
 'OBJECTID': 'float',
 'Shape_Leng': 'float',
 'area_ha': 'float',
 'concate': 'str',
 'Shape_Le_1': 'float',
 'Shape_Area': 'float',
#  'geometry': 'float',
 'TIPO_Code': 'int'
}
def count_area_by_cls(group):
    tipo_name = group.iloc[0]['TIPO']
    tipo_code = tipo_name_list.index(tipo_name)
    total_area_per_cls = group.area_ha.sum()

    res = pd.DataFrame(group.iloc[0]).T
    res['area_ha'] = total_area_per_cls
    res['TIPO_Code'] = tipo_code
    return res.drop(columns=['TIPO', 'geometry'])
    
df_area_per_tipo = ddf.groupby("TIPO").apply(count_area_by_cls, meta=meta).compute().reset_index().set_index('COD_DEF')
# df_area_per_tipo = ddf.groupby("TIPO").agg({'area_ha': sum}).compute()
df_area_per_tipo


#%%

""" Spanish to English """
import pandas as pd

df_en = pd.read_csv("C:\DHI\CCDC_GEE\outputs_V1/area_by_COD_DEF_english.csv")

#%%
col = 'TIPO'
map_dict = {}
src = df[col].unique()
src

#%%
tgt = ['Wetlands in small depressions supplied by rain and/or flooded or puddled savannahs and/or Zurals and/or Esteros',
        'Flood forests', 'Palmares de Moriche and/or Mixed Forests',
        'Streams and ravines', 'Transformed wetlands', 'Rivers',
        'Swampy areas', 'Flooding grasslands',
        'Wetlands in the process of transformation', 'Lagunas',
        'Rebalse forests', 'Helechales and/or Eneales',
        'Flood-prone bushes', 'No information',
        'Rice crops (rice fields)', 'Coastal lagoons',
        'Pozos de dunes', 'Palmares de Canangucha', 'Beaches',
        'Lows and/or Esteros',
        'Water reservoirs (dams, reservoirs) and/or Wetlands around hydroelectric plants',
        'Estuarine lagoons',
        'Treatment plants in mining and oil operations or lakes in abandoned quarries and excavations',
        'Salitrals', 'Seas and oceans',
        'Sediments exposed at low tide (rocky or cliff coastlines and/or intertidal mud flats)',
        'Irrigation canals', 'Interfluve floodplain forests',
        'Ponds for marine and inland aquaculture', 'Dunes',
        'Salt exploitations', 'Urban water treatment plants',
        'Constructed wetlands for wastewater treatment: oxidation and/or sedimentation lagoons']

for src_, tgt_ in zip(src, tgt):
  map_dict[src_] = tgt_

df.loc[:, col] = df[col].replace(map_dict)
df


#%%

