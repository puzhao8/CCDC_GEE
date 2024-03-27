
#%%

import pandas as pd
from pathlib import Path

version = 'V1'
df = pd.read_csv(f"outputs/training/sampled_points_wetland_label_Stratified_1k_per_cls_mrg_{version}.csv")

save_dir = Path(f"outputs/feature_importance_{version}")
save_dir.mkdir(exist_ok=True, parents=True)

rmse_bands = list(df.filter(regex='.*rmse.*').columns)
drop_bands = ['world_cover', 'GWL_FCS30', 'wetland_label', 'wetland_mask', 'cifor', 'rfw', 'idx', 'geometry', 'lat', 'lon']
drop_bands += rmse_bands

# config
y_col = 'wetland_mask'
feature_key = 'S2_selected'
band_select = True


from band_names import feature_dict

selected_bands = feature_dict[feature_key]
print(f"feature_key: {feature_key}")

if band_select:
    selected_bands += [y_col]
    X = df[selected_bands]
else:
    drop_bands.remove(y_col)
    X = df.drop(drop_bands, axis=1)

X = X.dropna()
y = X[y_col]

# remove labels from X
X = X.drop([y_col], axis=1) 
 

num_bands = X.shape[-1]

print(f"{y_col} ({num_bands} bands)")
print(X.shape, y.shape)
print(list(X.columns))


#%%
# referenece: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image


# from sklearn.datasets import load_breast_cancer
# X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)



#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# from yellowbrick.model_selection import rfecv
from sklearn.feature_selection import RFECV as rfecv
from yellowbrick.datasets import load_credit

# Load classification dataset
X, y = load_credit()

# cv = StratifiedKFold(5)
# visualizer = rfecv(RandomForestClassifier(), X=X, y=y, cv=5, scoring='f1_weighted')

#%%
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

X, y = load_credit()

estimator = RandomForestClassifier()
selector = RFECV(estimator, step=1, cv=5, n_jobs=-1)
selector = selector.fit(X, y)
selector.support_
selector.ranking_
