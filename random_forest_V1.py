
#%%

import pandas as pd
from pathlib import Path
import numpy as np
from band_names import feature_dict
np.random.seed(42)

# configuration
version = 'V1'
y_col = 'wetland_mask'
band_select = True

df = pd.read_csv(f"outputs/training/sampled_points_wetland_label_Stratified_1k_per_cls_mrg_{version}.csv")

seasonal_bands = feature_dict['Season']
df[seasonal_bands] = df[seasonal_bands] / 1e4

save_dir = Path(f"outputs/feature_importance_{version}/{y_col}")
save_dir.mkdir(exist_ok=True, parents=True)

#%%
# for feature_key in feature_dict.keys():
# for feature_key in [
#     'S1_coefs',
#     'SS_coefs',
#     'S2_coefs',
#     'S1_HM',
#     'SS_HM',
#     'S2_HM',
#     'S2_HM_V2']:
for feature_key in ['TOPO_V2']:

    # remove some label bands 
    rmse_bands = list(df.filter(regex='.*rmse.*').columns)
    drop_bands = ['world_cover', 'GWL_FCS30', 'wetland_label', 'wetland_mask', 'cifor', 'rfw', 'idx', 'geometry', 'lat', 'lon']
    drop_bands += rmse_bands


    if band_select: # select a few bands
        selected_bands = feature_dict[feature_key]
        print(f"feature_key: {feature_key}")

        X = df[selected_bands + [y_col]]

    else:
        drop_bands.remove(y_col)
        X = df.drop(drop_bands, axis=1)

    X = X.dropna()
    y = X[y_col].astype(int)

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
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint

    # Tree Visualisation
    from sklearn.tree import export_graphviz
    from IPython.display import Image


    # from sklearn.datasets import load_breast_cancer
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Baseline accuracy on train data: {train_acc:.2}")
    print(f"Baseline accuracy on test data: {test_acc:.2}")

    # report on test accuracy
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    df_report.to_csv(save_dir / f'{feature_key}_{y_col}_{num_bands}_bands_AccReport.csv')

    # Confusion Matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_.astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=clf.classes_.astype(int))
    disp.plot()
    plt.title(f"{feature_key}_{y_col}_{num_bands}_bands")
    plt.tight_layout()
    plt.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_Confusion_Matrix")
    plt.close()

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

    # plot_permutation_importance
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

    try:
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

    except:
        print(f"{feature_key} multicolliner plot failed !")

