"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A Recursive Feature Elimination (RFE) example with automatic tuning of the
number of features selected with cross-validation.

"""

# %%
# Data generation
# ---------------
#
# We build a classification task using 3 informative features. The introduction
# of 2 additional redundant (i.e. correlated) features has the effect that the
# selected features vary depending on the cross-validation fold. The remaining
# features are non-informative as they are drawn at random.

# from sklearn.datasets import make_classification

# X, y = make_classification(
#     n_samples=500,
#     n_features=15,
#     n_informative=3,
#     n_redundant=2,
#     n_repeated=0,
#     n_classes=8,
#     n_clusters_per_class=1,
#     class_sep=0.8,
#     random_state=0,
# )

#%%

import pandas as pd
from pathlib import Path
import numpy as np
from prettyprinter import pprint
import logging

logging.basicConfig(filename='outputs/rfecv/rfecv.log', level=logging.INFO)
np.random.seed(42)


def get_wetland_dataset(feature_key='S1', y_col='wetland_mask', version='V1'):
    # # configuration
    # version = 'V1'
    # y_col = 'wetland_label'

    from band_names import feature_dict
    selected_bands = feature_dict[feature_key]

    df = pd.read_csv(f"outputs/training/sampled_points_wetland_label_Stratified_1k_per_cls_mrg_{version}.csv")

    seasonal_bands = feature_dict['Season']
    df[seasonal_bands] = df[seasonal_bands] / 1e4

    selected_bands_w_label = selected_bands + [y_col]
    X = df[selected_bands_w_label]
    X = X.dropna() # drop NaN rows for both features and labels

    y = X[y_col].astype(int)
    X = X[selected_bands]

    return X, y


# %%
# Model training and selection
# ----------------------------
#
# We create the RFE object and compute the cross-validated scores. The scoring
# strategy "accuracy" optimizes the proportion of correctly classified samples.

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 10  # Minimum number of features to consider
clf = RandomForestClassifier()
cv = StratifiedKFold(5)


label = 'wetland_mask'
input_version = 'V1'
scorer = 'f1'

from band_names import feature_dict
for feature_key in feature_dict.keys():

    filename = f"rfecv_{input_version}_{feature_key}_{label}_{scorer}"
    logging.info(f"start {filename}")

    X, y = get_wetland_dataset(feature_key=feature_key, y_col=label, version=input_version)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scorer, # accuracy, weighted_f1
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(X, y)

    logging.info(f"Optimal number of features: {rfecv.n_features_}")
    logging.info(f"Selected features: {rfecv.get_feature_names_out()}")

    # %%
    # In the present case, the model with 3 features (which corresponds to the true
    # generative model) is found to be the most optimal.
    #
    # Plot number of features VS. cross-validation scores
    # ---------------------------------------------------

    import matplotlib.pyplot as plt

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Mean test {scorer}")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
        ecolor='pink',
        capsize=4,
        barsabove=True
    )
    plt.title(f"Recursive Feature Elimination \nwith correlated features\n{filename}")
    # plt.show()

    plt.savefig(f"outputs/rfecv/{filename}.png")
    logging.info(f"figsure saved into: outputs/rfecv/{filename}.png")

    # %%
    # From the plot above one can further notice a plateau of equivalent scores
    # (similar mean value and overlapping errorbars) for 3 to 5 selected features.
    # This is the result of introducing correlated features. Indeed, the optimal
    # model selected by the RFE can lie within this range, depending on the
    # cross-validation technique. The test accuracy decreases above 5 selected
    # features, this is, keeping non-informative features leads to over-fitting and
    # is therefore detrimental for the statistical performance of the models.
