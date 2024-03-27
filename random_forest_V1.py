
#%%

import pandas as pd
from pathlib import Path
import numpy as np
from group_bands import feature_dict, label_bands, fused_class_dict

np.random.seed(42)

# configuration
version = 'V1'
y_col = 'fused_label'
band_select = True

df = pd.read_csv(f"data/training/sampled_points_with_fused_label_stratified_1k_per_cls_V1.csv")


save_dir = Path(f"outputs/feature_importance_{version}/{y_col}")
save_dir.mkdir(exist_ok=True, parents=True)

#%%
# for feature_key in feature_dict.keys():
for feature_key in ['TOPO', 'S1', 'S2', 'SS', 'ALL', 'S1_coefs', 'S2_coefs', 'SS_coefs', 'S1_HM', 'S2_HM', 'SS_HM']:
    print(f"------------------- {feature_key} -------------------------")

    # remove some label bands 
    rmse_bands = list(df.filter(regex='.*rmse.*').columns)
    drop_bands = label_bands + ['idx', 'geometry', 'lat', 'lon']
    drop_bands += rmse_bands

    if band_select: # select a few bands
        selected_bands = feature_dict[feature_key]
        print(f"feature_key: {feature_key}")

        X = df[selected_bands + [y_col]]

    else:
        drop_bands.remove(y_col)
        X = df.drop(drop_bands, axis=1)

    X = X.dropna()
    X.replace([np.inf, -np.inf], 0, inplace=True)

    y = X[y_col].astype(int)

    # class names 
    unique_label_idx = sorted(y.unique())
    cls_name_bank = list(fused_class_dict.values())
    cls_names = [cls_name_bank[cls] for cls in unique_label_idx]
    print(f"unique_label_idx: {unique_label_idx}")

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
    df_cm = pd.DataFrame(data=cm, index=cls_names, columns=cls_names).astype(np.int16)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                         display_labels=clf.classes_.astype(int))
    # disp.plot()
    # plt.title(f"{feature_key}_{y_col}_{num_bands}_bands")
    # plt.tight_layout()
    # plt.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_Confusion_Matrix")
    # plt.close()


    import seaborn as sns
    plt.figure(figsize=(20, 15))
    sns.heatmap(data=df_cm, annot=True, annot_kws={"size": 16}, fmt='g', vmin=0, vmax=50, cmap=sns.color_palette("rocket", as_cmap=True))
    plt.xticks(fontsize=20, rotation=40, ha='right')
    plt.yticks(fontsize=22)
    plt.title(f"Confusion Matrix: [{feature_key} -> {y_col}]\n (Acc: {df_report.loc['accuracy', 'f1-score']:.2f}, wAvgF1: {df_report.loc['weighted avg', 'f1-score']:.2f})", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_Confusion_Matrix.png")
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

    # try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(1, 1)
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.to_list(), ax=ax, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    # ax2.set_yticklabels(dendro["ivl"])
    _ = fig.tight_layout()

    fig.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_multicolliner.png", dpi=300)

    " plot correlation matrix only "
    from matplotlib.colors import LinearSegmentedColormap as lsc
    cmap = lsc.from_list("custom_cmap", ["white", "darkred"])

    fig, ax = plt.subplots(1, 1)
    # , cmap=lsc.from_list("custom_cmap", ["white", "darkred"])
    cmap = sns.color_palette("rocket", as_cmap=True)
    im = ax.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(dendro_idx)
    ax.set_yticks(dendro_idx)
    ax.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax.set_yticklabels(dendro["ivl"])
    ax.set_title(f"{feature_key}_{y_col}_{num_bands}_bands")
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / f"{feature_key}_{y_col}_{num_bands}_bands_correlation.png", dpi=300)

    # except:
    #     print(f"{feature_key} multicolliner plot failed !")

