
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

# feature_dict = {
#   'ALL': bandList,
#   'S1': S1_bands,
#   'SS': SS_bands,
#   'S2': S2_bands,
#   'S2_selected': S2_bands_selected,
#   'TOPO': TOPO_bands,
#   'Climate': climate_bands
# }

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

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Baseline accuracy on train data: {train_acc:.2}")
print(f"Baseline accuracy on test data: {test_acc:.2}")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance")


pmt_importance = plot_permutation_importance(clf, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score")
fig.suptitle(
    f"{feature_key}-{y_col} ({num_bands} bands): Impurity-based vs. permutation importances on multicollinear features (train set: {train_acc:.2f})"
)
_ = fig.tight_layout()
fig.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_feature_importance.png", dpi=300)

#%%

# feature importance 
mrg = pd.concat([mdi_importances, pmt_importance.mean(axis=0)], axis=1)
mrg = mrg.rename(columns={0: 'mdi', 1: 'pmt'})
mrg = mrg.sort_values(by='pmt', ascending=False)
mrg.to_csv(save_dir / f'{feature_key}_{y_col}_{num_bands}_bands_feature_importance.csv')

q75 = mrg.quantile(.75)
mrg_flt = mrg[(mrg['mdi'] > q75.mdi) | (mrg['pmt'] > q75.pmt)]
top_25p_bands = list(mrg_flt.index)
mrg_flt = mrg_flt.sort_values(by='pmt', ascending=False)
mrg_flt.to_csv(save_dir / f'{feature_key}_{y_col}_{num_bands}_bands_feature_importance_top_25p.csv')


#%%
fig, ax = plt.subplots(figsize=(8, 12))
pmt_importance_test = plot_permutation_importance(clf, X_test, y_test, ax)
ax.set_title(f"{feature_key}-{y_col} ({num_bands} bands): Permutation Importances on multicollinear features\n(test set: {test_acc:.2f})")
ax.set_xlabel("Decrease in accuracy score")
_ = ax.figure.tight_layout()

fig.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_feature_importance_test.png", dpi=300)


#%% multicolliner_features

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

fig.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_handle_multicolliner_features.png", dpi=300)


#%%

if False:
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

    fig.savefig(save_dir / f"{y_col}_{len(selected_features_names)}_selected_bands_bands_multicolliner_seleted_features.png", dpi=300)


#%%

if False:
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