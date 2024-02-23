
#%%

import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs\sampled_training_points_wetland_mask_stratified.csv")
df = df.dropna()

save_dir = Path("outputs/feature_importance")
save_dir.mkdir(exist_ok=True, parents=True)

rmse_bands = list(df.filter(regex='.*rmse.*').columns)
drop_bands = ['world_cover', 'wetland_label', 'wetland_mask', 'idx', 'geometry', 'lat', 'lon']
drop_bands += rmse_bands

selected_bands = [
  'B11_amp',
  'B11_max',
  'B11_mean',
  'B11_min',
  'B11_phase2',
  'B12_amp',
  'B12_max',
  'B12_mean',
  'B12_min',
  'B2_amp',
  'B2_max',
  'B2_mean',
  'B2_min',
  'B2_phase3',
  'B3_amp',
  'B3_max',
  'B3_mean',
  'B3_phase3',
  'B4_max',
  'B4_mean',
  'B4_min',
  'B5_max',
  'B5_mean',
  'B5_min',
  'B6_amp',
  'B6_max',
  'B6_mean',
  'B6_phase',
  'B7_max',
  'B7_mean',
  'B7_phase',
  'B8A_amp',
  'B8A_max',
  'B8A_phase',
  'B8_amp',
  'B8_max',
  'VH_mean',
  'VH_amp',
  'VH_min',
  'VV_amp',
  'VV_min',
  'VV_mean',
  'elevation',
  'ndvi_mean',
  'ndwi_max',
  'ndwi_mean',
  'ndwi_min',
  'rem',
  'slope',
  'twi',
  'water_mean'
 ]

y_col = 'wetland_label'
y = df[y_col]
X = df.drop(drop_bands, axis=1)
# X = df[selected_bands]

num_bands = X.shape[-1]
print(X.shape, y.shape)


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
print(f"Baseline accuracy on test data: {clf.score(X_test, y_test):.2}")

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return pd.DataFrame(result.importances[perm_sorted_idx].T, columns=X.columns[perm_sorted_idx])


mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance")


pmt_importance = plot_permutation_importance(clf, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score")
fig.suptitle(
    f"{y_col} ({num_bands} bands): Impurity-based vs. permutation importances on multicollinear features (train set)"
)
_ = fig.tight_layout()
fig.savefig(save_dir / f"feature_importance_{y_col}_{num_bands}_bands.png", dpi=300)

#%%

# feature importance 
mrg = pd.concat([mdi_importances, pmt_importance.mean(axis=0)], axis=1)
mrg = mrg.rename(columns={0: 'mdi', 1: 'pmt'})
mrg.to_csv(f'outputs/feature_importance_{y_col}_{num_bands}_bands.csv')

q75 = mrg.quantile(.75)
mrg_flt = mrg[(mrg['mdi'] > q75.mdi) | (mrg['pmt'] > q75.pmt)]
top_25p_bands = list(mrg_flt.index)
mrg_flt
mrg_flt.to_csv(save_dir / f'feature_importance_{y_col}_{num_bands}_bands_top_25p.csv')




#%%
fig, ax = plt.subplots(figsize=(10, 8))
pmt_importance_test = plot_permutation_importance(clf, X_test, y_test, ax)
ax.set_title(f"{y_col} ({num_bands} bands): Permutation Importances on multicollinear features\n(test set)")
ax.set_xlabel("Decrease in accuracy score")
_ = ax.figure.tight_layout()

fig.savefig(save_dir / f"test_feature_importance_{y_col}_{num_bands}_bands.png", dpi=300)


#%%

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 16))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()

fig.savefig(save_dir / f"handle_multicolliner_features_{y_col}_{num_bands}_bands.png", dpi=300)


#%%


from collections import defaultdict

y_col = 'wetland_label'
y = df[y_col]
X = df.drop(drop_bands, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
# cluster_id_to_feature_ids = defaultdict(list)
# for idx, cluster_id in enumerate(cluster_ids):
#     cluster_id_to_feature_ids[cluster_id].append(idx)
# selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# selected_features_names = X.columns[selected_features]

selected_features_names = ['B12_mean', 'B8_mean', 'B4_mean', 'rem', 'elevation', 'twi']
# selected_features_names = ['ndvi_mean', 'ndvi_amp', 'B12_mean', 'B8_mean', 'B4_mean', 'B8_phase', 'B8_slope', 'VH_mean', 'VH_min', 'VH_max', "VV_mean", 'VV_min', 'VV_max', 'twi', 'rem', 'elevation']

X_train_sel = X_train[selected_features_names]
X_test_sel = X_test[selected_features_names]

clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sel.fit(X_train_sel, y_train)
test_acc = clf_sel.score(X_test_sel, y_test)
print(
    "Baseline accuracy on test data with features removed:"
    f" {test_acc:.2f}"
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
ax.set_title(f"{y_col} ({len(selected_features_names)} bands): Permutation Importances on selected subset of features\n(test set: {test_acc:.2f})")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()

fig.savefig(save_dir / f"multicolliner_seleted_features_{y_col}_{len(selected_features_names)}_bands.png", dpi=300)


#%%

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

wetland_type_names = {
  0: "non wetlands",
  1: "Transformed wetlands",
  2: "Varzeas and/or Igapós",
  3: "Wetlands in small depressions", # supplied by rain and/or floodable or waterlogged savannas and/or Zurales and/or estuaries
  4: "Flooded forests",
  5: "Overflow Forests",
  6: "Interfluvial flooded forests",
  7: "Floodable grasslands",
  8: "Rivers",
  9: "Wetlands in the process of transformation",
  10: "Swamps"
}

class_names = list(wetland_type_names.values())
tree = clf_sel.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X_train_sel.columns, class_names=class_names)
plt.show()

#%%
dot_data  = export_graphviz(tree, out_file=None, 
                           feature_names=X_train_sel.columns,
                           class_names=class_names,
                           filled=True)

# Draw graph
import graphviz
graph = graphviz.Source(dot_data, format="pdf") 
graph.render(save_dir / f"{y_col}_{len(selected_features_names)}_bands_decision_tree_class_name")  # Saves the tree diagram to a file decision_tree.png